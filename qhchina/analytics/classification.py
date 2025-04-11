import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# from datasets import Dataset  # Removing this dependency
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
import os

# Custom dataset implementation to replace Hugging Face's Dataset
class SimpleDataset:
    """A minimal implementation of a dataset for use with HuggingFace Transformers Trainer."""
    
    def __init__(self, data_dict: Dict[str, List]):
        """
        Initialize the dataset with a dictionary of features.
        
        Args:
            data_dict: Dictionary mapping feature names to lists of values.
                       All lists must have the same length.
        """
        # Validate input
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
        
        # Check all lists have the same length
        lengths = [len(v) for v in data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All feature lists must have the same length")
        
        self.data = data_dict
        self.length = lengths[0] if lengths else 0
        self.features = list(data_dict.keys())
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length
    
    def __getitem__(self, idx):
        """Get a sample from the dataset at the given index."""
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise IndexError(f"Index {idx} out of bounds for dataset of size {self.length}")
            
            # Return a dictionary with features for this sample
            return {feature: self.data[feature][idx] for feature in self.features}
        else:
            # For slicing and other advanced indexing, make a new dataset
            if isinstance(idx, slice):
                # Convert slice to list of indices
                indices = list(range(*idx.indices(self.length)))
            else:
                # Assume idx is an iterable of indices
                indices = list(idx)
            
            # Create a new dataset with the selected samples
            new_data = {feature: [self.data[feature][i] for i in indices] for feature in self.features}
            return SimpleDataset(new_data)
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, List]):
        """
        Create a dataset from a dictionary of lists.
        
        Args:
            data_dict: Dictionary mapping feature names to lists of values
        
        Returns:
            A SimpleDataset instance
        """
        return cls(data_dict)

class SequenceClassifier:
    """
    A simplified text classifier based on pretrained BERT models.
    """
    def __init__(self, 
                 model_name: Optional[str] = None, 
                 num_labels: Optional[int] = None):
        """
        Initialize the classifier with a pretrained model and tokenizer.

        Args:
            model_name: str - The name or path of the pretrained model to use
            num_labels: int - The number of labels for the classification task
        """
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # if provided, initialize the tokenizer and model
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if num_labels:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def _data_collator(self, batch):
        """Standard data collator for all methods"""

        texts = [item["text"] for item in batch]
        encodings = self.tokenizer(texts, 
                                   truncation=True, 
                                   padding=True, 
                                   return_tensors="pt")
        if "label" in batch[0]:
            encodings["labels"] = torch.tensor([item["label"] for item in batch])
            
        return encodings

    # Define compute_metrics function
    def _compute_metrics(self, eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        # Calculate accuracy
        accuracy = accuracy_score(label_ids, preds)
        
        # Calculate precision, recall, f1, support for each class
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, preds, average='weighted')
        
        # Return all metrics in a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def train(self, texts: List[str], labels: List[int], 
              val_split: float = 0.2, epochs: int = 3, 
              batch_size: int = 16, output_dir: str = "./results", learning_rate: float = 2e-5,
              eval_interval: Optional[int] = None):
        """
        Train the classifier on text data.
        
        Args:
            texts: List of input texts for training
            labels: List of corresponding labels
            val_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            output_dir: Directory to save model outputs
            learning_rate: Learning rate for training
            eval_interval: Steps between evaluations (if None, will evaluate per epoch)
        """
        # Input validation
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if not labels:
            raise ValueError("Labels list cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        if not 0 <= val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        # Check if any labels are outside the expected range
        unique_labels = set(labels)
        if max(unique_labels) >= self.model.config.num_labels or min(unique_labels) < 0:
            warnings.warn(f"Labels contain values outside the expected range [0, {self.model.config.num_labels-1}]. "
                          f"This may cause errors during training.")
            
        # Create dataset and split if validation is requested
        if val_split > 0:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=val_split, random_state=42
            )
            
            train_dataset = SimpleDataset.from_dict({"text": train_texts, "label": train_labels})
            val_dataset = SimpleDataset.from_dict({"text": val_texts, "label": val_labels})
        else:
            train_dataset = SimpleDataset.from_dict({"text": texts, "label": labels})
            val_dataset = None
            
        # Set up training arguments
        training_args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "learning_rate": learning_rate,
            "report_to": "none",
            "remove_unused_columns": False
        }
        
        # Set evaluation strategy
        if val_dataset:
            training_args_dict["load_best_model_at_end"] = True
            training_args_dict["save_total_limit"] = 5
            if eval_interval:
                training_args_dict["eval_strategy"] = "steps"
                training_args_dict["eval_steps"] = eval_interval
                training_args_dict["save_strategy"] = "steps"
                training_args_dict["save_steps"] = eval_interval
                training_args_dict["logging_strategy"] = "steps"
                training_args_dict["logging_steps"] = eval_interval
            else:
                training_args_dict["eval_strategy"] = "epoch"
                training_args_dict["save_strategy"] = "epoch"
                training_args_dict["logging_strategy"] = "steps"
                training_args_dict["logging_steps"] = 100
        else:
            training_args_dict["eval_strategy"] = "no" 
            training_args_dict["save_strategy"] = "no"
            training_args_dict["logging_strategy"] = "steps"
            training_args_dict["logging_steps"] = 100
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        # Train the model
        trainer.train()

        # Handle model state after training
        if val_dataset and training_args_dict.get("load_best_model_at_end", False):
            print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
        
        # Save model and tokenizer
        self.model.save_pretrained(f"{output_dir}/final-model")
        self.tokenizer.save_pretrained(f"{output_dir}/final-model")
        print(f"The best model saved to {output_dir}/final-model.")
        
        # Important: we no longer keep a reference to the trainer
        # This avoids confusion between the trainer's model and our model
        del trainer
    
    def predict(self, 
                texts: Union[str, List[str]],
                batch_size: int = 16,
                return_probs: bool = False) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """
        Make predictions on new texts.
        Creates a fresh Trainer for each prediction to avoid state issues.
    
        Args:
            texts: Union[str, List[str]] - The texts to predict on
            batch_size: int - The batch size for prediction
            return_probs: bool - Whether to return the probabilities of the predictions

        Returns:
            Union[List[int], Tuple[List[int], List[List[float]]]] - The predicted labels or the predicted labels and probabilities
        """
            
        # Check batch size
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
            
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Check for empty texts
        if not texts:
            raise ValueError("Cannot predict on empty text list")
            
        # Create dataset with consistent format
        dataset = SimpleDataset.from_dict({"text": texts})
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a fresh Training Arguments optimized for prediction
        predict_args = TrainingArguments(
            per_device_eval_batch_size=batch_size,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Always create a new trainer for prediction
        temp_trainer = Trainer(
            model=self.model,
            args=predict_args,
            data_collator=self._data_collator,
        )
        
        # Run prediction with no_grad to optimize speed
        with torch.no_grad():
            predictions = temp_trainer.predict(test_dataset=dataset)
            
        pred_labels = np.argmax(predictions.predictions, axis=-1).tolist()
            
        if return_probs:
            probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).tolist()
            return pred_labels, probs
            
        return pred_labels
    
    def evaluate(self, texts: List[str], labels: List[int], batch_size: int = 16) -> Dict[str, float]:
        """
        Evaluate the model on a set of texts and labels.
        Creates a fresh Trainer for each evaluation to avoid state issues.

        Args:
            texts: List[str] - The texts to evaluate on
            labels: List[int] - The corresponding labels
            batch_size: int - The batch size for evaluation

        Returns:
            Dict[str, float] - The evaluation metrics
        """
            
        # Input validation
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if not labels:
            raise ValueError("Labels list cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
            
        # Check if any labels are outside the expected range
        unique_labels = set(labels)
        if max(unique_labels) >= self.model.config.num_labels or min(unique_labels) < 0:
            warnings.warn(f"Labels contain values outside the expected range [0, {self.model.config.num_labels-1}]. "
                          f"This may cause errors during evaluation.")
        
        # Create dataset
        eval_dataset = SimpleDataset.from_dict({"text": texts, "label": labels})
        
        # Explicitly set model to evaluation mode
        self.model.eval()
        
        # Create a fresh Training Arguments optimized for evaluation
        eval_args = TrainingArguments(
            per_device_eval_batch_size=batch_size,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Always create a new trainer for each evaluation
        temp_trainer = Trainer(
            model=self.model,  # Use the existing model
            args=eval_args,
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        # Run evaluation with no_grad to optimize speed
        with torch.no_grad():
            eval_results = temp_trainer.evaluate(eval_dataset=eval_dataset)
        
        return eval_results
    
    def save(self, path: str):
        """
        Save the model and tokenizer.
        
        Args:
            path: str - Path to save the model and tokenizer
        """

        if not path:
            raise ValueError("Path cannot be empty")
            
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path: str, num_labels: int):
        """
        Load a previously saved model and tokenizer.
        
        Args:
            path: str - Path to the saved model and tokenizer
            num_labels: int - Number of labels for classification
        
        Returns:
            Classifier instance with the model and tokenizer loaded
        """
        # Check if path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path '{path}' does not exist")
        
        # Check num_labels
        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError("num_labels must be a positive integer")
            
        try:
            # Create a new classifier instance with the loaded model and tokenizer
            return cls(model_name=path, num_labels=num_labels)
        except Exception as e:
            raise ValueError(f"Failed to load model/tokenizer from '{path}': {str(e)}")