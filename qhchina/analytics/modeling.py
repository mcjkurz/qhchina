import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def set_device(device: Optional[str] = None) -> str:
    """
    Determine the appropriate device for computation.
    
    Args:
        device: Optional device specification ('cuda', 'mps', 'cpu', or None)
                If None, will return 'cpu'. Only returns 'cuda' or 'mps' if explicitly specified.
    
    Returns:
        str: The selected device ('cuda', 'mps', or 'cpu')
    """
    if device is not None:
        device = device.lower()

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("CUDA not available, defaulting to CPU")
            return "cpu"
    elif device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            print("MPS not available, defaulting to CPU")
            return "cpu"
    return "cpu"

class TextDataset(Dataset):
    def __init__(self, texts: List[str], 
                 tokenizer: AutoTokenizer, 
                 max_length: Optional[int] = None, 
                 labels: Optional[List[int]] = None):
        """
        Initialize a dataset for text classification.
        
        Args:
            texts: List of raw texts
            tokenizer: Tokenizer to use for encoding texts
            max_length: Maximum sequence length for tokenization
            labels: Optional list of labels
        """
        if labels is not None and len(labels) != len(texts):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of texts ({len(texts)})")
            
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - 'text': Raw text string
                - 'label': Label (if available)
        """
        item = {'text': self.texts[idx]}
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.texts)

def plot_training_curves(
    history: Dict[str, Any],
    save_path: Path,
    title: str = 'Training Loss per Batch',
) -> None:
    """
    Plot training curves and save to file.
    
    Args:
        history: Dictionary containing training history with train_metrics and val_metrics
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Plot training losses
    train_batches = [m['batch'] for m in history['train_metrics']]
    train_losses = [m['loss'] for m in history['train_metrics']]
    plt.plot(train_batches, train_losses, label='Training Loss')
    
    # Plot validation loss if available
    if history['val_metrics'] is not None and len(history['val_metrics']) > 0:
        val_batches = [m['batch'] for m in history['val_metrics']]
        val_losses = [m['loss'] for m in history['val_metrics']]
        plt.plot(val_batches, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_bert_classifier(
    model: AutoModelForSequenceClassification,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
    device: Optional[str] = None,
    warmup_steps: int = 0,
    max_train_batches: Optional[int] = None,
    logging_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    plot_interval: int = 100,  # Plot every N batches
    val_interval: Optional[int] = None,  # Validate every N batches
) -> Dict[str, Any]:
    """
    Train a BERT-based classifier with custom training loop.
    
    Args:
        model: Pre-loaded BERT model for classification
        train_dataset: PyTorch Dataset for training
        val_dataset: Optional PyTorch Dataset for validation. If None, no validation is performed.
        batch_size: Training batch size
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        device: Device to train on ('cuda', 'mps', or 'cpu')
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_train_batches: Maximum number of training batches per epoch (for quick experiments)
        logging_dir: Directory to save training logs
        save_dir: Directory to save model checkpoints
        plot_interval: Number of batches between plot updates
        val_interval: Number of batches between validation runs. If None, only validate at the end of each epoch.
    
    Returns:
        Dictionary containing training history and final model
    """
    print(f"Training set size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation set size: {len(val_dataset)}")
    if val_interval is not None and val_dataset is None:
        raise ValueError("val_dataset must be provided if val_interval is not None")
    
    # Set device
    device = set_device(device)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps for scheduling
    total_steps = len(train_loader) * num_epochs
    if max_train_batches is not None:
        total_steps = min(max_train_batches, total_steps)
    
    if warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})")
    
    # Create warmup scheduler
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create main scheduler
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-6,
            total_iters=total_steps - warmup_steps
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    else:
        # No warmup, just use the main scheduler
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1e-6,
            total_iters=total_steps
        )
    
    # Training history
    history = {
        'train_metrics': [],  # List of training metrics per batch
        'val_metrics': [] if val_dataset is not None else None  # List of validation metrics
    }
    
    # Setup logging directory
    if logging_dir:
        log_path = Path(logging_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    total_batches_processed = 0
    finished_training = False

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batches_processed_in_current_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check if we've reached max_train_batches
            if max_train_batches is not None and batch_idx >= max_train_batches:
                finished_training = True
                break
                
            # Tokenize the batch
            texts = batch['text']
            encodings = train_dataset.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=train_dataset.max_length,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            if 'label' in batch:
                encodings['labels'] = batch['label']
            
            # Move batch to device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            # Forward pass
            outputs = model(**encodings)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Get loss value and clear memory
            current_loss = loss.item()
            del outputs, loss
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Update metrics
            total_loss += current_loss
            batches_processed_in_current_epoch += 1
            total_batches_processed += 1
            
            # Record training metrics
            train_metrics = {
                'batch': total_batches_processed,
                'loss': current_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            history['train_metrics'].append(train_metrics)
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}'
            })
            
            # Periodically update plots and save history
            if logging_dir and total_batches_processed % plot_interval == 0:
                # Save history as JSON
                with open(log_path / 'training_history.json', 'w') as f:
                    json.dump(history, f)
                
                # Update plot
                plot_training_curves(
                    history=history,
                    save_path=log_path / 'training_curves.png',
                    title=f'Training Loss per Batch (Batch {total_batches_processed})'
                )
            
            # Run validation if interval is specified and it's time
            if val_dataset is not None and val_interval is not None and total_batches_processed % val_interval == 0:
                model.eval()
                val_results = evaluate(model, val_dataset, batch_size, device)
                if history['val_metrics'] is None:
                    history['val_metrics'] = []
                val_results['batch'] = total_batches_processed
                history['val_metrics'].append(val_results)
                # print val loss
                print(f'Validation Loss: {val_results["loss"]:.6f}')
                model.train()  # Set back to training mode
        
        if finished_training:
            break
            
        # Calculate epoch metrics
        epoch_loss = total_loss / batches_processed_in_current_epoch
        
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # Run validation at the end of each epoch
        if val_dataset is not None:
            model.eval()
            val_results = evaluate(model, val_dataset, batch_size, device)
            if history['val_metrics'] is None:
                history['val_metrics'] = []
            val_results['batch'] = total_batches_processed
            history['val_metrics'].append(val_results)
            # print val loss
            print(f'Validation Loss: {val_results["loss"]:.6f}')
            model.train()  # Set back to training mode
        
        # Save checkpoint if save_dir is provided
        if save_dir:
            save_path = Path(save_dir) / f'checkpoint_epoch_{epoch+1}'
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
    
    # Final plot and save if logging_dir is provided
    if logging_dir:
        # Save final history as JSON
        with open(log_path / 'training_history.json', 'w') as f:
            json.dump(history, f)
        
        # Create final plot
        plot_training_curves(
            history=history,
            save_path=log_path / 'training_curves.png',
            title=f'Training Loss per Batch (Final, Batch {total_batches_processed})'
        )
    
    return {
        'model': model,
        'history': history,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset) if val_dataset is not None else None
    }

def evaluate(
    model: AutoModelForSequenceClassification,
    dataset: Dataset,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a BERT-based classifier on a dataset.
    
    Args:
        model: Pre-loaded BERT model for classification
        dataset: PyTorch Dataset to evaluate
        batch_size: Batch size for evaluation
        device: Device to evaluate on ('cuda', 'mps', or 'cpu')
    
    Returns:
        Dictionary containing evaluation metrics and statistics, including average loss
    """
    # Input validation
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Set device
    device = set_device(device)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Process batches
    all_labels = []
    all_predictions = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Tokenize the batch
            texts = batch['text']
            encodings = dataset.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=dataset.max_length,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            if 'label' in batch:
                labels = batch['label'].numpy()
                all_labels.extend(labels)
                encodings['labels'] = batch['label']
            
            # Move batch to device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            # Forward pass
            outputs = model(**encodings)
            
            # Get predictions and loss
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Clear GPU memory if needed
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    unique_labels = np.unique(all_labels)
    
    # Calculate metrics with different averaging methods
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Weighted average (takes class imbalance into account)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted',
        zero_division=0
    )
    
    # Macro average (treats all classes equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='macro',
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in unique_labels:
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels == label,
            all_predictions == label,
            average='binary',
            zero_division=0
        )
        class_metrics[f'class_{label}'] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'support': np.sum(all_labels == label)
        }
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Create results dictionary
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'class_metrics': class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_predictions.tolist(),
        'true_labels': all_labels.tolist()
    }
    
    return results

def predict(
    model: AutoModelForSequenceClassification,
    texts: Optional[List[str]] = None,
    dataset: Optional[Dataset] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    return_probs: bool = False,
) -> Union[List[int], Dict[str, Any]]:
    """
    Make predictions on new texts using a trained BERT classifier.
    
    Args:
        model: Pre-loaded BERT model for classification
        texts: List of texts to predict (required if dataset is None)
        dataset: PyTorch Dataset to use for inference (required if texts is None)
        tokenizer: Pre-loaded tokenizer corresponding to the model (required if texts is provided)
        batch_size: Batch size for inference
        device: Device to run inference on ('cuda', 'mps', or 'cpu')
        return_probs: Whether to return prediction probabilities
    
    Returns:
        If return_probs is False:
            List of predicted labels
        Otherwise:
            Dictionary containing:
                - 'labels': List of predicted labels
                - 'probabilities': List of probability distributions
    """
    # Validate input arguments
    if texts is None and dataset is None:
        raise ValueError("Either texts or dataset must be provided")
    if texts is not None and dataset is not None:
        raise ValueError("Cannot provide both texts and dataset")
    if texts is not None and tokenizer is None:
        raise ValueError("tokenizer must be provided when using texts")
    
    # Set device
    device = set_device(device)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    if texts is not None:
        # Create dataset from texts
        dataset = TextDataset(texts, tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize results
    all_predictions = []
    all_probabilities = [] if return_probs else None
    
    # Inference loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            # Tokenize the batch
            texts = batch['text']
            encodings = dataset.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=dataset.max_length,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            # Move batch to device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            # Forward pass
            outputs = model(**encodings)
            
            # Get predictions and probabilities
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            
            if return_probs:
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probabilities.extend(probs.cpu().numpy())
    
    # Convert numpy arrays to lists
    all_predictions = [int(x) for x in all_predictions]
    if return_probs:
        all_probabilities = [x.tolist() for x in all_probabilities]
    
    # Return results
    if return_probs:
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
    else:
        return all_predictions

def make_datasets(
    data: List[Tuple[str, int]],
    tokenizer: AutoTokenizer,
    split: Union[Tuple[float, float], Tuple[float, float, float]],
    max_length: Optional[int] = None,
    random_seed: int = None,
) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
    """
    Create train/val/test datasets from a list of (text, label) tuples with stratification.
    
    Args:
        data: List of tuples where each tuple contains (text, label)
        tokenizer: Tokenizer to use for text encoding
        split: Tuple of proportions for splits:
            - (train_prop, val_prop) for train/val split
            - (train_prop, val_prop, test_prop) for train/val/test split
        max_length: Maximum sequence length for tokenization
        random_seed: Random seed for reproducible splits
    
    Returns:
        If split has length 2:
            Tuple of (train_dataset, val_dataset)
        If split has length 3:
            Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if len(split) not in [2, 3]:
        raise ValueError("split must be a tuple of length 2 or 3")
    
    if not all(0 < p < 1 for p in split):
        raise ValueError("All proportions must be between 0 and 1")
    
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.0")
    
    # Unzip the data into texts and labels
    texts, labels = zip(*data)
    
    # Print class distribution in original data
    print("\nOriginal dataset class distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} examples ({count/total:.2%})")
    
    # Calculate split sizes
    total_size = len(texts)
    if len(split) == 2:
        train_prop, val_prop = split
        train_size = int(total_size * train_prop)
        val_size = total_size - train_size
        test_size = 0
    else:
        train_prop, val_prop, test_prop = split
        train_size = int(total_size * train_prop)
        val_size = int(total_size * val_prop)
        test_size = total_size - train_size - val_size
    
    # Create train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_size + test_size,
        random_state=random_seed,
        stratify=labels
    )
    
    # Create test split if needed
    if len(split) == 3:
        # Further split validation data into val and test
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            val_texts, val_labels,
            test_size=test_size,
            random_state=random_seed,
            stratify=val_labels
        )
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length, train_labels)
    val_dataset = TextDataset(val_texts, tokenizer, max_length, val_labels)
    
    # Print split sizes
    print(f"\nTotal samples: {total_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Print class distribution for each split
    print("\nTraining set class distribution:")
    print_class_distribution(train_dataset)
    
    print("\nValidation set class distribution:")
    print_class_distribution(val_dataset)
    
    if len(split) == 3:
        test_dataset = TextDataset(test_texts, tokenizer, max_length, test_labels)
        print(f"\nTest set size: {len(test_dataset)}")
        
        print("\nTest set class distribution:")
        print_class_distribution(test_dataset)
        
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, val_dataset

def bert_encode(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    pooling_strategy: str = 'cls',
    device: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Extract embeddings from a transformer model for given text(s).
    
    Args:
        model: Pre-loaded transformer model (e.g., BERT, RoBERTa, etc.)
        tokenizer: Pre-loaded tokenizer corresponding to the model
        texts: List of texts to encode
        batch_size: If None, process texts individually. If provided, process in batches
        max_length: Maximum sequence length for tokenization
        pooling_strategy: Strategy for pooling embeddings ('cls' or 'mean')
        device: Device to run inference on ('cuda', 'mps', or 'cpu')
    
    Returns:
        List of numpy arrays, each of shape (hidden_size,)
        
    Raises:
        ValueError: If texts list is empty or pooling_strategy is invalid
    """
    # Input validation
    if not texts:
        raise ValueError("Texts list cannot be empty")
    if pooling_strategy not in ['cls', 'mean']:
        raise ValueError("pooling_strategy must be either 'cls' or 'mean'")
    
    # Set device
    device = set_device(device)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # If max_length is not provided, use model's max length
    if max_length is None:
        max_length = getattr(model.config, 'max_position_embeddings', 512)
    
    # Process texts
    all_embeddings = []
    
    if batch_size is None:
        # Process one text at a time
        for text in tqdm(texts, desc="Processing texts"):
            # Tokenize single text
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors='pt',
                return_token_type_ids=False
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Apply pooling strategy
                if pooling_strategy == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                else:  # mean pooling
                    attention_mask = attention_mask.unsqueeze(-1)
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy and add to list
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings[0])  # Remove batch dimension
                
                # Clear memory
                del outputs
                if device == "cuda":
                    torch.cuda.empty_cache()
    
    else:
        # Process in batches
        dataset = TextDataset(texts, tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = batch['attention_mask']
                
                # Apply pooling strategy
                if pooling_strategy == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                else:  # mean pooling
                    attention_mask = attention_mask.unsqueeze(-1)
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy and add to list
                embeddings = embeddings.cpu().numpy()
                all_embeddings.extend(embeddings)
                
                # Clear memory
                del outputs
                if device == "cuda":
                    torch.cuda.empty_cache()
    
    return all_embeddings

def print_class_distribution(dataset):
    """Print the distribution of classes in a dataset."""
    if dataset.labels is None:
        print("Dataset has no labels")
        return
    
    labels = dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} examples ({count/total:.2%})")