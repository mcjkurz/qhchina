import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def set_device(device: Optional[str] = None) -> str:
    """
    Determine the appropriate device for computation.
    
    Args:
        device: Optional device specification ('cuda', 'mps', 'cpu', or None)
                If None, will automatically select the best available device
    
    Returns:
        str: The selected device ('cuda', 'mps', or 'cpu')
    """
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512, labels: Optional[List[int]] = None):
        self.encodings = tokenizer(texts, 
                                   truncation=True, 
                                   padding=True, 
                                   max_length=max_length, 
                                   return_tensors='pt', 
                                   return_token_type_ids=False)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def plot_training_curves(
    history: Dict[str, Any],
    save_path: Path,
    title: str = 'Training Loss per Batch',
) -> None:
    """
    Plot training curves and save to file.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    
    # Plot batch losses
    batch_numbers, losses = zip(*history['batch_losses'])
    plt.plot(batch_numbers, losses, label='Batch Loss')
    
    plt.title(title)
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_bert_classifier(
    model: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    device: Optional[str] = None,
    scheduler_type: str = "linear",
    warmup_steps: int = 0,
    max_train_batches: Optional[int] = None,
    logging_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    plot_interval: int = 100,  # Plot every N batches
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
        scheduler_type: Type of learning rate scheduler ('linear' or 'cosine')
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_train_batches: Maximum number of training batches per epoch (for quick experiments)
        logging_dir: Directory to save training logs
        save_dir: Directory to save model checkpoints
        plot_interval: Number of batches between plot updates
    
    Returns:
        Dictionary containing training history and final model
    """
    print(f"Training set size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation set size: {len(val_dataset)}")
    
    # Set device
    device = set_device(device)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps for scheduling
    total_steps = len(train_loader) * num_epochs
    if max_train_batches is not None:
        total_steps = min(max_train_batches * num_epochs, total_steps)
    
    # Create warmup scheduler
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create main scheduler
        if scheduler_type == "linear":
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - warmup_steps
            )
        elif scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    else:
        # No warmup, just use the main scheduler
        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    # Training history
    history = {
        'batch_losses': [],  # List of (batch_number, loss) tuples
        'val_loss': [] if val_dataset is not None else None
    }
    
    # Setup logging directory
    if logging_dir:
        log_path = Path(logging_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    batch_number = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check if we've reached max_train_batches
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            current_loss = loss.item()
            total_loss += current_loss
            num_batches += 1
            
            # Record batch loss
            history['batch_losses'].append((batch_number, current_loss))
            batch_number += 1
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}'
            })
            
            # Periodically update plots and save history
            if logging_dir and batch_number % plot_interval == 0:
                # Save history as JSON
                with open(log_path / 'training_history.json', 'w') as f:
                    json.dump(history, f)
                
                # Update plot
                plot_training_curves(
                    history=history,
                    save_path=log_path / 'training_curves.png',
                    title=f'Training Loss per Batch (Batch {batch_number})'
                )
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        
        # Validation if val_dataset is provided
        if val_dataset is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f'Validation')
                for batch in val_progress:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    current_val_loss = outputs.loss.item()
                    val_loss += current_val_loss
                    val_progress.set_postfix({
                        'loss': f'{current_val_loss:.4f}'
                    })
            
            val_loss = val_loss / len(val_loader)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Train Loss: {epoch_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}:')
            print(f'Train Loss: {epoch_loss:.4f}')
        
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
            title=f'Training Loss per Batch (Final, Batch {batch_number})'
        )
    
    return {
        'model': model,
        'history': history,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset) if val_dataset is not None else None
    }


def evaluate(
    model: PreTrainedModel,
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
        Dictionary containing evaluation metrics and statistics
    """
    # Set device
    device = set_device(device)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Get true labels from dataset and verify number of classes
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        all_labels.extend(batch['labels'].numpy())
    all_labels = np.array(all_labels)
    
    # Sanity check: verify number of unique labels matches model's num_labels
    unique_labels = np.unique(all_labels)
    if len(unique_labels) != model.config.num_labels:
        raise ValueError(
            f"Number of unique labels ({len(unique_labels)}) does not match "
            f"model's expected number of labels ({model.config.num_labels})"
        )
    
    # Get predictions using classify function
    predictions = classify(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device
    )
    all_predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted'
    )
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in unique_labels:
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels == label,
            all_predictions == label,
            average='binary'
        )
        class_metrics[f'class_{label}'] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_predictions.tolist(),
        'true_labels': all_labels.tolist()
    }
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nPer-class Metrics:")
    for label, metrics in class_metrics.items():
        print(f"\n{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    return results

def classify(
    model: PreTrainedModel,
    texts: Optional[Union[str, List[str]]] = None,
    dataset: Optional[Dataset] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    return_probs: bool = False,
) -> Union[List[int], Dict[str, Any]]:
    """
    Make predictions on new texts using a trained BERT classifier.
    
    Args:
        model: Pre-loaded BERT model for classification
        texts: Single text or list of texts to predict (required if dataset is None)
        dataset: PyTorch Dataset to use for inference (required if texts is None)
        tokenizer: Pre-loaded tokenizer corresponding to the model (required if texts is provided)
        batch_size: Batch size for inference
        max_length: Maximum sequence length for tokenization (only used if texts is provided)
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
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Create dataset from texts
        dataset = TextDataset(texts, tokenizer, max_length)
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize results
    all_predictions = []
    all_probabilities = [] if return_probs else None
    
    # Inference loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
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

def create_datasets(
    texts: List[str],
    labels: List[int],
    tokenizer: AutoTokenizer,
    split: Union[Tuple[float, float], Tuple[float, float, float]],
    max_length: int = 512,
    random_seed: int = 42,
) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
    """
    Create train/val/test datasets from texts and labels with stratification.
    
    Args:
        texts: List of texts
        labels: List of labels
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
    print(f"Total samples: {total_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    if len(split) == 3:
        test_dataset = TextDataset(test_texts, tokenizer, max_length, test_labels)
        print(f"Test set size: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, val_dataset

def bert_encode(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    pooling_strategy: str = 'cls',
    device: Optional[str] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Extract embeddings from a BERT model for given text(s).
    
    Args:
        model: Pre-loaded BERT model
        tokenizer: Pre-loaded tokenizer corresponding to the model
        texts: List of texts to encode
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        pooling_strategy: Strategy for pooling embeddings ('cls' or 'mean')
        device: Device to run inference on ('cuda', 'mps', or 'cpu')
    
    Returns:
        If single text:
            numpy array of shape (hidden_size,)
        If list of texts:
            list of numpy arrays, each of shape (hidden_size,)
    """    
    # Set device
    device = set_device(device)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize list to store embeddings
    all_embeddings = []
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch, output_hidden_states=True)
            
            # Get last hidden states
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = batch['attention_mask']
            
            # Apply pooling strategy
            if pooling_strategy == 'cls':
                # Use CLS token embeddings
                embeddings = last_hidden_state[:, 0, :]
            elif pooling_strategy == 'mean':
                # Calculate mean pooling with attention mask
                # Expand attention_mask to same dims as hidden states
                attention_mask = attention_mask.unsqueeze(-1)
                # Sum up vectors for each sequence
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                # Count number of non-padding tokens
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                # Calculate mean
                embeddings = sum_embeddings / sum_mask
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
            
            # Convert to numpy and add to list
            embeddings = embeddings.cpu().numpy()
            all_embeddings.extend(embeddings)
    
    # Convert to numpy arrays
    all_embeddings = [np.array(emb) for emb in all_embeddings]
    
    return all_embeddings
