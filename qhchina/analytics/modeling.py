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

def add_corpus_tags(corpora, labels, target_words):
    """
    Add corpus-specific tags to target words in all corpora at once.
    
    Args:
        corpora: List of corpora (each corpus is list of tokenized sentences)
        labels: List of corpus labels
        target_words: List of words to tag
    
    Returns:
        List of processed corpora where target words have been tagged with their corpus label
    """
    processed_corpora = []
    target_words_set = set(target_words)
    
    for corpus, label in zip(corpora, labels):
        processed_corpus = []
        for sentence in corpus:
            processed_sentence = []
            for token in sentence:
                if token in target_words_set:
                    processed_sentence.append(f"{token}_{label}")
                else:
                    processed_sentence.append(token)
            processed_corpus.append(processed_sentence)
        processed_corpora.append(processed_corpus)
    
    return processed_corpora

def semantic_change(corpora_data, target_words, method='full', min_freq=5, top_n=None, 
                   stopwords=None, window=5, min_length=1, vector_size=256, epochs=3, 
                   anchor_words=1000, limit_most_similar=None, return_models=False, seed=None,
                   batch_size=32, calculate_loss=False):
    """
    Analyze semantic change of target words across consecutive corpora using various methods.
    
    Args:
        corpora_data: List of (label, corpus) tuples, where each corpus is a list of tokenized sentences
                     The order of tuples determines the order of analysis
        target_words: List of words to track for semantic change
        method: Analysis method ('window', 'procrustes', or 'full')
                - window: Train one model with tagged words for each consecutive pair
                - procrustes: Train separate models and align with Procrustes
                - full: (default) Train one model on a combined corpus with balanced token counts
        min_freq: Minimum frequency threshold for words
        top_n: Number of top similar/dissimilar words to return (if None, return all)
        stopwords: Set of stopwords to exclude from final results
        window: Context window size for Word2Vec
        min_length: Minimum length of words to include in analysis
        vector_size: Size of word vectors
        epochs: Number of training epochs
        anchor_words: Number of most frequent words to use for alignment (default=1000)
        limit_most_similar: If provided, only include words that are among the N most
                          similar words to the target in either time period
        return_models: Whether to return trained Word2Vec models along with results
        seed: Random seed for reproducibility
        batch_size: Batch size for Word2Vec training
        calculate_loss: Whether to calculate and return loss during Word2Vec training
    Returns:
        If return_models=False:
            Dict of format:
            {
                'corpus1->corpus2': {
                    'target_word1': [(word1, change1), (word2, change2), ...],
                    'target_word2': [(word1, change1), (word2, change2), ...]
                },
                ...
            }
        If return_models=True:
            Tuple (results_dict, models_dict)
    """
    # Extract labels and corpora from the input tuples, maintaining order
    labels = [label for label, _ in corpora_data]
    corpora = [corpus for _, corpus in corpora_data]
    transformations = None
    
    if method == 'window':
        results, models = _semantic_change_window(
            corpora, labels, target_words, min_freq, top_n,
            stopwords, window, min_length, vector_size, epochs,
            limit_most_similar, seed, batch_size, calculate_loss
        )
    elif method == 'procrustes':
        results, models, transformations = _semantic_change_procrustes(
            corpora, labels, target_words, min_freq, top_n,
            stopwords, window, min_length, vector_size, epochs,
            anchor_words, limit_most_similar, seed, batch_size, calculate_loss
        )
    elif method == 'full':
        results, models = _semantic_change_full(
            corpora, labels, target_words, min_freq, top_n,
            stopwords, window, min_length, vector_size, epochs,
            limit_most_similar, seed, batch_size, calculate_loss
        )
    else:
        raise ValueError("Method must be 'window', 'procrustes', or 'full'")
    
    if return_models:
        if transformations is not None:
            return results, models, transformations
        return results, models
    return results

def _semantic_change_window(corpora, labels, target_words, min_freq=5, top_n=None,
                           stopwords=None, window=5, min_length=1, vector_size=256, 
                           epochs=5, limit_most_similar=None, seed=None, batch_size=32, calculate_loss=False):
    """
    Analyze semantic change using window training approach.
    
    This approach uses TemporalWord2Vec with a sliding window over pairs of corpora.
    For each consecutive pair of corpora, a single model is trained where target words
    are tagged with their respective time periods.
    
    Internal helper function for semantic_change().
    """
    from qhchina.analytics.word2vec import TemporalWord2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    import numpy as np
    import random
    from collections import Counter
    from qhchina.helpers.texts import sample_sentences_to_token_count
    random.seed(seed)
    
    results = {}
    models = {}
    
    # Process all corpora to add tags to target words
    processed_corpora = add_corpus_tags(corpora, labels, target_words)
    
    # Create list of consecutive pairs for the progress bar
    corpus_pairs = list(zip(labels[:-1], labels[1:]))
    
    # Analyze each consecutive pair of corpora
    pbar = tqdm(enumerate(corpus_pairs), total=len(corpus_pairs), 
               desc="Analyzing corpus pairs (window)")
    for i, (label1, label2) in pbar:
        label_pair = f"{label1}->{label2}"
        pbar.set_description(f"Analyzing: {label1}->{label2}")
        
        # Count tokens in both corpora using Counter
        tokens_prev = sum(len(sent) for sent in processed_corpora[i])
        tokens_next = sum(len(sent) for sent in processed_corpora[i+1])
        min_tokens = min(tokens_prev, tokens_next)
        
        # Sample sentences to balance token counts
        corpus_prev = sample_sentences_to_token_count(processed_corpora[i], min_tokens)
        corpus_next = sample_sentences_to_token_count(processed_corpora[i+1], min_tokens)
        combined_corpus = corpus_prev + corpus_next
        
        # Create temporal word map for this pair of corpora
        # Map each target word to its temporal variants for this pair
        temporal_word_map = {target: [f"{target}_{label1}", f"{target}_{label2}"] for target in target_words}
        
        # Train TemporalWord2Vec model on combined corpus
        model = TemporalWord2Vec(
            temporal_word_map=temporal_word_map,
            vector_size=vector_size,
            window=window,
            min_count=min_freq,
            epochs=epochs,
            seed=seed,
            sg=1,  # Use Skip-gram architecture for better results with temporal words
            batch_size=batch_size
        )
        model.train(sentences=combined_corpus, calculate_loss=calculate_loss)
        
        models[label_pair] = model
        
        # Get vocabulary that appears in both corpora with minimum frequency using Counter
        vocab_counts_prev = Counter(word for sent in corpus_prev for word in sent)
        vocab_counts_next = Counter(word for sent in corpus_next for word in sent)
        
        # Find common vocabulary meeting frequency requirements (without min_length filter)
        # Exclude temporal variants from the common vocabulary
        temporal_variants = set()
        for variants in temporal_word_map.values():
            temporal_variants.update(variants)
            
        common_vocab = [word for word in vocab_counts_prev if 
                      word in vocab_counts_next and
                      vocab_counts_prev[word] >= min_freq and
                      vocab_counts_next[word] >= min_freq and
                      word in model.vocab and
                      word not in temporal_variants]  # Exclude temporal variants
        
        # Get vectors for all common words
        vocab_vectors = np.array([model.get_vector(word) for word in common_vocab])
        results[label_pair] = {}
        
        # Analyze each target word
        for target in target_words:
            target_prev = f"{target}_{label1}"
            target_next = f"{target}_{label2}"
            
            # Skip if either temporal variant is not in the model's vocabulary
            if target_prev not in model.vocab or target_next not in model.vocab:
                continue
            
            # If limit_most_similar is provided, filter common_vocab to only include
            # words that are among the most similar to either target state
            if limit_most_similar is not None:
                similar_prev = set(word for word, _ in model.most_similar(target_prev, topn=limit_most_similar))
                similar_next = set(word for word, _ in model.most_similar(target_next, topn=limit_most_similar))
                similar_words = similar_prev.union(similar_next)
                
                # Filter common_vocab and vocab_vectors
                filtered_indices = [i for i, word in enumerate(common_vocab) if word in similar_words]
                if not filtered_indices:  # Skip if no similar words found
                    continue
                    
                filtered_vocab = [common_vocab[i] for i in filtered_indices]
                filtered_vectors = vocab_vectors[filtered_indices]
                
                # Update for the rest of the analysis
                common_vocab = filtered_vocab
                vocab_vectors = filtered_vectors
            
            # Get the temporal variant vectors
            target_prev_vector = model.get_vector(target_prev).reshape(1, -1)
            target_next_vector = model.get_vector(target_next).reshape(1, -1)
            
            # Calculate cosine similarities between each temporal variant and the common vocabulary
            sims_prev = cosine_similarity(
                target_prev_vector,
                vocab_vectors
            )[0]
            
            sims_next = cosine_similarity(
                target_next_vector,
                vocab_vectors
            )[0]
            
            # Calculate changes in similarity
            changes = [(word, sim_next - sim_prev) 
                      for word, sim_prev, sim_next 
                      in zip(common_vocab, sims_prev, sims_next)]
            
            # Filter results by min_length and stopwords
            filtered_changes = []
            for word, change in changes:
                if len(word) >= min_length and (not stopwords or word not in stopwords):
                    filtered_changes.append((word, change))
            
            # Sort by change magnitude (most positive first)
            filtered_changes.sort(key=lambda x: x[1], reverse=True)
            
            # Optionally limit to top_n most changed in each direction
            if top_n is not None and len(filtered_changes) > top_n * 2:
                moved_towards = filtered_changes[:top_n]  # Words with greatest positive change
                moved_away_from = filtered_changes[-top_n:]  # Words with greatest negative change
                filtered_changes = moved_towards + moved_away_from
            
            results[label_pair][target] = filtered_changes
    
    return results, models

def _semantic_change_procrustes(corpora, labels, target_words, min_freq=5, top_n=None,
                             stopwords=None, window=5, min_length=1, vector_size=256, 
                             epochs=5, anchor_words=1000, limit_most_similar=None, seed=None,
                             batch_size=32, calculate_loss=False):
    """
    Analyze semantic change using procrustes training approach with Procrustes alignment.
    Internal helper function for semantic_change().
    """
    from qhchina.analytics.word2vec import Word2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    import numpy as np
    from collections import Counter
    import random
    random.seed(seed)
    
    results = {}
    models = {}
    transformations = {}  # Store transformation matrices for each transition
    
    # Train separate models for each corpus and count words
    word_counts = []  # Store word counts for each corpus
    corpus_sizes = []  # Store total words in each corpus
    pbar = tqdm(enumerate(corpora), total=len(corpora), 
               desc="Training individual models")
    
    for i, corpus in pbar:
        pbar.set_description(f"Training model for corpus: {labels[i]}")
        
        # Count word frequencies using Counter
        counts = Counter(word for sent in corpus for word in sent)
        total_words = sum(counts.values())
        word_counts.append(counts)
        corpus_sizes.append(total_words)
        
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_freq,
            epochs=epochs,
            seed=seed,
            batch_size=batch_size
        )
        model.train(sentences=corpus, calculate_loss=calculate_loss)
        models[labels[i]] = model
    
    # Analyze consecutive pairs
    corpus_pairs = list(zip(labels[:-1], labels[1:]))
    pbar = tqdm(enumerate(corpus_pairs), 
               total=len(corpus_pairs), desc="Analyzing model pairs (procrustes)")
    
    for i, (label1, label2) in pbar:
        model1 = models[label1]
        model2 = models[label2]
        label_pair = f"{label1}->{label2}"
        pbar.set_description(f"Analyzing: {label1}->{label2}")
        
        # Get word counts and corpus sizes
        counts1 = word_counts[i]
        counts2 = word_counts[i+1]
        size1 = corpus_sizes[i]
        size2 = corpus_sizes[i+1]
        
        # Find common vocabulary between models (excluding target words)
        common_vocab = [word for word in model1.vocab if 
                      word in model2.vocab and
                      counts1[word] >= min_freq and
                      counts2[word] >= min_freq and
                      word not in target_words]
        
        # Calculate relative frequencies and select anchor words
        # Note: We use ALL top frequent words (including stopwords) for alignment
        rel_freqs = {}
        for word in common_vocab:
            freq1 = counts1[word] / size1
            freq2 = counts2[word] / size2
            # Use geometric mean of relative frequencies as stability measure
            rel_freqs[word] = (freq1 * freq2) ** 0.5
        
        # Sort by frequency and select top anchor_words
        anchor_vocab = sorted(rel_freqs.items(), key=lambda x: x[1], reverse=True)
        anchor_vocab = [word for word, _ in anchor_vocab[:anchor_words]]
        
        if len(anchor_vocab) < 2:
            print(f"Not enough anchor words found for {label1}->{label2}")
            anchor_vocab = common_vocab  # Fallback to all common words if not enough anchors
        
        # Get vectors for alignment using anchor words
        anchor_vecs1 = np.array([model1.get_vector(word) for word in anchor_vocab])
        anchor_vecs2 = np.array([model2.get_vector(word) for word in anchor_vocab])
        
        # Align second space to first space using anchor words
        _, transformation = align_vectors(anchor_vecs2, anchor_vecs1)
        transformations[label_pair] = transformation  # Store the transformation matrix
        
        # Get vectors for all common words (for similarity computation)
        vecs1 = np.array([model1.get_vector(word) for word in common_vocab])
        
        results[label_pair] = {}
        
        # Analyze each target word
        for target in target_words:
            if target not in model1.vocab or target not in model2.vocab:
                continue
            
            # If limit_most_similar is provided, filter common_vocab
            if limit_most_similar is not None:
                similar1 = set(word for word, _ in model1.most_similar(target, topn=limit_most_similar))
                similar2 = set(word for word, _ in model2.most_similar(target, topn=limit_most_similar))
                similar_words = similar1.union(similar2)
                
                # Filter common_vocab and vectors
                filtered_indices = [i for i, word in enumerate(common_vocab) if word in similar_words]
                if not filtered_indices:  # Skip if no similar words found
                    continue
                    
                filtered_vocab = [common_vocab[i] for i in filtered_indices]
                filtered_vecs = vecs1[filtered_indices]
                
                # Update for the rest of the analysis
                common_vocab = filtered_vocab
                vecs1 = filtered_vecs
            
            # Get target vectors from both spaces
            target_vec1 = model1.get_vector(target)
            target_vec2 = model2.get_vector(target)
            
            # Align target vector from second space using the same transformation
            aligned_target_vec2 = np.dot(target_vec2.reshape(1, -1), transformation.T)
            
            # Calculate similarities with common vocabulary
            sims1 = cosine_similarity(
                target_vec1.reshape(1, -1),
                vecs1
            )[0]
            
            sims2 = cosine_similarity(
                aligned_target_vec2,
                vecs1
            )[0]
            
            changes = [(word, sim2 - sim1) 
                      for word, sim1, sim2 
                      in zip(common_vocab, sims1, sims2)]
            
            # Filter results by min_length and stopwords
            filtered_changes = []
            for word, change in changes:
                if len(word) >= min_length and (not stopwords or word not in stopwords):
                    filtered_changes.append((word, change))
            
            filtered_changes.sort(key=lambda x: x[1], reverse=True)
            
            if top_n is not None and len(filtered_changes) > top_n * 2:
                moved_towards = filtered_changes[:top_n]
                moved_away_from = filtered_changes[-top_n:]
                filtered_changes = moved_towards + moved_away_from
            
            results[label_pair][target] = filtered_changes
    
    return results, models, transformations

def _semantic_change_full(corpora, labels, target_words, min_freq=5, top_n=None,
                           stopwords=None, window=5, min_length=1, vector_size=256, 
                           epochs=5, limit_most_similar=None, seed=None, batch_size=32, calculate_loss=False):
    """
    Analyze semantic change using a full training approach.
    This method builds a single combined corpus with balanced token counts from all time periods,
    then trains a single TemporalWord2Vec model on this combined corpus.
    
    Internal helper function for semantic_change().
    """
    from qhchina.analytics.word2vec import TemporalWord2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    import numpy as np
    import random
    from collections import Counter
    from qhchina.helpers.texts import sample_sentences_to_token_count
    random.seed(seed)
    
    results = {}
    models = {}
    
    # Process all corpora to add tags to target words
    processed_corpora = add_corpus_tags(corpora, labels, target_words)
    
    # Count tokens in each corpus
    token_counts = [sum(len(sent) for sent in corpus) for corpus in processed_corpora]
    min_tokens = min(token_counts)
    
    # Sample each corpus to have roughly the same number of tokens
    sampled_corpora = []
    for corpus in processed_corpora:
        sampled_corpus = sample_sentences_to_token_count(corpus, min_tokens)
        sampled_corpora.append(sampled_corpus)
    
    # Combine all sampled corpora into one
    combined_corpus = []
    for corpus in sampled_corpora:
        combined_corpus.extend(corpus)
    
    # Create temporal word map for TemporalWord2Vec
    temporal_word_map = {target: [f"{target}_{label}" for label in labels] for target in target_words}
    
    # Train a single TemporalWord2Vec model on the combined corpus
    model = TemporalWord2Vec(
        temporal_word_map=temporal_word_map,
        vector_size=vector_size,
        window=window,
        min_count=min_freq,
        epochs=epochs,
        seed=seed,
        sg=1,  # Use Skip-gram architecture
        batch_size=batch_size
    )
    model.train(sentences=combined_corpus, calculate_loss=calculate_loss)
    
    # Store the trained model
    models['full'] = model
    
    # Create list of consecutive pairs for the results
    corpus_pairs = list(zip(labels[:-1], labels[1:]))
    
    # Analyze each consecutive pair of corpora using the same model
    pbar = tqdm(enumerate(corpus_pairs), total=len(corpus_pairs), 
               desc="Analyzing corpus pairs (full)")
    
    for i, (label1, label2) in pbar:
        label_pair = f"{label1}->{label2}"
        pbar.set_description(f"Analyzing: {label1}->{label2}")
        
        # Get vocabulary counts in both corpora using Counter
        vocab_counts_prev = Counter(word for sent in sampled_corpora[i] for word in sent)
        vocab_counts_next = Counter(word for sent in sampled_corpora[i+1] for word in sent)
        
        # Find common vocabulary meeting frequency requirements (without min_length filter)
        common_vocab = [word for word in vocab_counts_prev if 
                      word in vocab_counts_next and
                      vocab_counts_prev[word] >= min_freq and
                      vocab_counts_next[word] >= min_freq and
                      word in model.vocab]
        
        # Get vectors for all common words
        vocab_vectors = np.array([model.get_vector(word) for word in common_vocab])
        results[label_pair] = {}
        
        # Analyze each target word
        for target in target_words:
            target_prev = f"{target}_{label1}"
            target_next = f"{target}_{label2}"
            
            # Check if target variants exist in the model
            if model.get_vector(target_prev) is None or model.get_vector(target_next) is None:
                continue
            
            # If limit_most_similar is provided, filter common_vocab to only include
            # words that are among the most similar to either target state
            if limit_most_similar is not None:
                similar_prev = set(word for word, _ in model.most_similar(target_prev, topn=limit_most_similar))
                similar_next = set(word for word, _ in model.most_similar(target_next, topn=limit_most_similar))
                similar_words = similar_prev.union(similar_next)
                
                # Filter common_vocab and vocab_vectors
                filtered_indices = [i for i, word in enumerate(common_vocab) if word in similar_words]
                if not filtered_indices:  # Skip if no similar words found
                    continue
                    
                filtered_vocab = [common_vocab[i] for i in filtered_indices]
                filtered_vectors = vocab_vectors[filtered_indices]
                
                # Update for the rest of the analysis
                common_vocab = filtered_vocab
                vocab_vectors = filtered_vectors
            
            target_prev_vector = np.array(model.get_vector(target_prev)).reshape(1, -1)
            target_next_vector = np.array(model.get_vector(target_next)).reshape(1, -1)
            
            sims_prev = cosine_similarity(
                target_prev_vector,
                vocab_vectors
            )[0]
            
            sims_next = cosine_similarity(
                target_next_vector,
                vocab_vectors
            )[0]
            
            changes = [(word, sim_next - sim_prev) 
                      for word, sim_prev, sim_next 
                      in zip(common_vocab, sims_prev, sims_next)]
            
            # Filter results by min_length and stopwords (keeping corpus tags intact)
            filtered_changes = []
            for word, change in changes:
                if len(word) >= min_length and (not stopwords or word not in stopwords):
                    filtered_changes.append((word, change))
            
            filtered_changes.sort(key=lambda x: x[1], reverse=True)
            
            if top_n is not None and len(filtered_changes) > top_n * 2:
                moved_towards = filtered_changes[:top_n]
                moved_away_from = filtered_changes[-top_n:]
                filtered_changes = moved_towards + moved_away_from
            
            results[label_pair][target] = filtered_changes
    
    return results, models

def align_vectors(source_vectors, target_vectors):
    """
    Align source vectors with target vectors using Procrustes analysis.
    
    Args:
        source_vectors: numpy array of vectors to be aligned
        target_vectors: numpy array of vectors to align to
        
    Returns:
        Tuple of (aligned_vectors, transformation_matrix)
        - aligned_vectors: The aligned source vectors
        - transformation_matrix: The orthogonal transformation matrix that can be used to align other vectors
    """
    # Center the vectors
    source_centered = source_vectors - np.mean(source_vectors, axis=0)
    target_centered = target_vectors - np.mean(target_vectors, axis=0)
    
    # Compute the covariance matrix
    covariance = np.dot(target_centered.T, source_centered)
    
    # Compute SVD
    U, _, Vt = np.linalg.svd(covariance)
    
    # Compute the rotation matrix
    rotation = np.dot(U, Vt)
    
    # Apply the rotation to the source vectors
    aligned_vectors = np.dot(source_vectors, rotation.T)
    
    return aligned_vectors, rotation

def visualize_semantic_trajectory(
    semantic_change_results, target_word, word2vec_models, 
    method='pca', n_neighbors=5, perplexity=30, figsize=(12, 8),
    filename=None, transformations=None, adjust_text_labels=False,
    target_position_method='neighbor_mean', umap_n_neighbors=15, umap_min_dist=0.1, 
    seed=None, fontsize=12, arrowstyle='-|>', arrow_color='blue', arrow_mutation_scale=15):
    """
    Visualize the semantic trajectory of a target word across corpora by showing its movement
    relative to its most significant neighbors in each transition period.
    
    Args:
        semantic_change_results: Results dictionary from semantic_change function
        target_word: The target word to visualize
        word2vec_models: Dictionary mapping corpus pairs to Word2Vec models (consecutive approach)
                       or corpus labels to models (independent approach)
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        n_neighbors: Number of most significant neighbors to show for each transition
        perplexity: Perplexity parameter for t-SNE (if method='tsne')
        figsize: Figure size tuple
        filename: If provided, save the plot to this file
        transformations: Dictionary mapping corpus pairs to transformation matrices (independent approach)
        adjust_text_labels: Whether to use adjustText to prevent label overlap
        target_position_method: How to determine target word position ('embedding' or 'neighbor_mean')
                              'embedding': Project target word vector along with neighbor vectors
                              'neighbor_mean': Use mean of neighbor vectors after projection (default)
        umap_n_neighbors: Number of neighbors to consider for UMAP (if method='umap')
        umap_min_dist: Minimum distance parameter for UMAP (if method='umap')
        seed: Random seed for reproducibility
        fontsize: Font size for text labels
        arrowstyle: Arrow style for the movement arrow
        arrow_color: Color for the movement arrow
        arrow_mutation_scale: Mutation scale for the movement arrow
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    if target_position_method not in ['embedding', 'neighbor_mean']:
        raise ValueError("target_position_method must be either 'embedding' or 'neighbor_mean'")
    
    if method not in ['pca', 'tsne', 'umap']:
        raise ValueError("method must be one of 'pca', 'tsne', or 'umap'")
    
    if method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP method requires the 'umap-learn' package. Install it with 'pip install umap-learn'.")
    
    # Get all corpus labels in order
    corpus_pairs = list(semantic_change_results.keys())

    # Determine if we're using consecutive or independent approach
    is_consecutive = transformations is None
    
    for idx, pair in enumerate(corpus_pairs):
        # Create a new figure for each transition
        plt.figure(figsize=figsize)
        label1, label2 = pair.split('->')

        target_prev = f"{target_word}_{label1}"
        target_next = f"{target_word}_{label2}"
        
        # Get the appropriate model(s) based on approach
        if is_consecutive: # for consecutive approach, we have one model per transition
            model = word2vec_models[pair]
            if target_prev not in model.vocab or target_next not in model.vocab:
                plt.text(0.5, 0.5, f"Word not found in {pair}", 
                       ha='center', va='center')
                plt.close()
                continue
        else: # for independent approach, we have two models per transition
            model1 = word2vec_models[label1]
            model2 = word2vec_models[label2]
            if target_word not in model1.vocab or target_word not in model2.vocab:
                plt.text(0.5, 0.5, f"Word not found in {pair}", 
                       ha='center', va='center')
                plt.close()
                continue
        
        # Get the most significant neighbors from semantic change results
        changes = semantic_change_results[pair][target_word]
        if not changes:
            plt.text(0.5, 0.5, f"No significant changes in {pair}", 
                   ha='center', va='center')
            plt.close()
            continue
        
        # Get top words moved towards and away from
        if len(changes) < n_neighbors * 2:
            print(f"Not enough changes in {pair} for {target_word}; might produce inaccurate results.")
        moved_towards = [word for word, change in changes[:n_neighbors]]
        moved_away = [word for word, change in changes[-n_neighbors:]]
        neighbor_words = moved_towards + moved_away
        
        # Get vectors for neighbor words and target words if using embedding method
        if is_consecutive: 
            # for consecutive approach, we have one vector space per transition
            neighbor_vectors = np.array([model.get_vector(word) for word in neighbor_words])
            if target_position_method == 'embedding':
                all_vectors = np.vstack([
                    neighbor_vectors,
                    model.get_vector(target_prev).reshape(1, -1),
                    model.get_vector(target_next).reshape(1, -1)
                ])
            else:
                all_vectors = neighbor_vectors # we will add target vectors later by averaging
        else: # independent approach
            # for independent approach, we have two vector spaces per transition
            # we use the first space for all neighbors & target vector pre-transition
            # and the second space only for the target vector post-transition, which we need to align to the first space
            neighbor_vectors = np.array([model1.get_vector(word) for word in neighbor_words])
            if target_position_method == 'embedding':
                # For independent models, align the target vector from second model
                target_vec1 = model1.get_vector(target_word).reshape(1, -1)
                target_vec2_aligned = np.dot(model2.get_vector(target_word).reshape(1, -1), 
                                   transformations[pair].T)
                all_vectors = np.vstack([
                    neighbor_vectors, 
                    target_vec1, 
                    target_vec2_aligned])
            else:
                all_vectors = neighbor_vectors # we will add target vectors later by averaging
        
        # Project vectors to 2D
        if method == 'pca':
            projector = PCA(n_components=2)
            projected = projector.fit_transform(all_vectors)
            explained_var = projector.explained_variance_ratio_
            x_label = f"PC1 ({explained_var[0]:.1%} variance)"
            y_label = f"PC2 ({explained_var[1]:.1%} variance)"
        elif method == 'tsne':
            projector = TSNE(n_components=2, perplexity=min(perplexity, len(all_vectors) - 1))
            projected = projector.fit_transform(all_vectors)
            x_label = "t-SNE Dimension 1"
            y_label = "t-SNE Dimension 2"
        elif method == 'umap':
            # Use UMAP for dimensionality reduction
            projector = umap.UMAP(
                n_components=2,
                n_neighbors=min(umap_n_neighbors, len(all_vectors) - 1),
                min_dist=umap_min_dist,
                random_state=seed,
                n_jobs=None  # Use None instead of 1 to avoid the warning
            )
            projected = projector.fit_transform(all_vectors)
            x_label = "UMAP Dimension 1"
            y_label = "UMAP Dimension 2"
        
        # Split projected points back into neighbors and targets
        if target_position_method == 'embedding':
            neighbor_points = projected[:-2]  # All but last two points are neighbors
            target_prev_point = projected[-2]  # Second-to-last point is target_prev
            target_next_point = projected[-1]  # Last point is target_next
        elif target_position_method == 'neighbor_mean':
            # we need to average the embeddings of the neighbors to get the target words' position
            neighbor_points = projected
            n_away = len(moved_away)
            n_towards = len(moved_towards)
            # Calculate centers of each group
            # First n_towards points are the "moved towards" neighbors
            # Last n_away points are the "moved away" neighbors
            target_prev_point = np.mean(neighbor_points[-n_away:], axis=0)  # Away from center
            target_next_point = np.mean(neighbor_points[:n_towards], axis=0)  # Towards center
        
        # Plot all neighbor words in the same color
        plt.scatter(neighbor_points[:, 0], neighbor_points[:, 1],
                   c='gray', alpha=0.6, label='Context Words')
        
        # Plot target word positions
        plt.scatter([target_prev_point[0], target_next_point[0]], 
                   [target_prev_point[1], target_next_point[1]],
                   c='blue', s=100, label='Target Word')
        
        # Add arrow showing movement
        arrow = FancyArrowPatch(
            target_prev_point, target_next_point,
            arrowstyle=arrowstyle,
            color=arrow_color,
            mutation_scale=arrow_mutation_scale
        )
        plt.gca().add_patch(arrow)
        
        # Add labels with adjustText if enabled
        texts = []
        # Add labels for neighbors
        for i, word in enumerate(neighbor_words):
            texts.append(plt.text(neighbor_points[i, 0], neighbor_points[i, 1], word,
                                fontsize=fontsize*0.8, alpha=0.7))
            
        texts.append(plt.text(target_prev_point[0], target_prev_point[1], target_prev,
                            fontsize=fontsize, fontweight='bold', c='red'))
        texts.append(plt.text(target_next_point[0], target_next_point[1], target_next,
                            fontsize=fontsize, fontweight='bold', c='red'))
        
        if adjust_text_labels:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        plt.title(f"Transition: {label1} → {label2}", fontsize=fontsize*1.2, pad=10)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=fontsize*0.8)
        
        # Adjust tick label sizes
        plt.xticks(fontsize=fontsize*0.8)
        plt.yticks(fontsize=fontsize*0.8)
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Save the plot
        if filename:
            transition_filename = f"{filename}_{pair}.png"
            plt.savefig(transition_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot for transition {pair} to {transition_filename}")
        
        plt.close()

def visualize_semantic_trajectory_full(semantic_change_results, target_word, model, 
                                        method='pca', n_neighbors=5, perplexity=30, figsize=(15, 10),
                                        filename=None, adjust_text_labels=False,
                                        target_position_method='neighbor_mean', umap_n_neighbors=15, 
                                        umap_min_dist=0.1, seed=None, fontsize=12, arrowstyle='-|>', 
                                        arrow_color='blue', arrow_mutation_scale=15):
    """
    Create visualization of a target word's semantic trajectory across time periods
    specifically designed for the 'full' method of semantic_change.
    
    This function works only with the output of semantic_change with method='full', 
    which trains a single model with all temporal variants in the same vector space.
    
    Args:
        semantic_change_results: Results dictionary from semantic_change function
        target_word: The target word to visualize
        model: The TemporalWord2Vec model from semantic_change with method='full'
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        n_neighbors: Number of most significant neighbors to show for each transition
        perplexity: Perplexity parameter for t-SNE (if method='tsne')
        figsize: Figure size tuple
        filename: If provided, save the plot to this file
        adjust_text_labels: Whether to use adjustText to prevent label overlap
        target_position_method: How to determine target word position ('embedding' or 'neighbor_mean')
                              'embedding': Project target word vector along with neighbor vectors
                              'neighbor_mean': Use mean of neighbor vectors after projection
        umap_n_neighbors: Number of neighbors to consider for UMAP (if method='umap')
        umap_min_dist: Minimum distance parameter for UMAP (if method='umap')
        seed: Random seed for reproducibility
        fontsize: Base font size for text elements
        arrowstyle: Style of arrows for matplotlib FancyArrowPatch
        arrow_color: Color of trajectory arrows
        arrow_mutation_scale: Size factor for arrows
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    if target_position_method not in ['embedding', 'neighbor_mean']:
        raise ValueError("target_position_method must be either 'embedding' or 'neighbor_mean'")
    
    if method not in ['pca', 'tsne', 'umap']:
        raise ValueError("method must be one of 'pca', 'tsne', or 'umap'")
    
    if method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP method requires the 'umap-learn' package. Install it with 'pip install umap-learn'.")
    
    # Get all corpus labels in order
    corpus_pairs = list(semantic_change_results.keys())
    
    # Extract all labels from the pairs (e.g., "1960s->1980s" gives "1960s", "1980s")
    all_labels = []
    for pair in corpus_pairs:
        labels = pair.split("->")
        if len(all_labels) == 0:
            all_labels.append(labels[0])  # Add the first label from the first pair
        all_labels.append(labels[1])  # Always add the second label
    
    # Create dictionaries to track context words by corpus transition
    context_words_dict = {}  # {corpus_pair: {"towards": [words], "away": [words]}}
    
    # Process the semantic change results to gather context words
    for pair_idx, pair in enumerate(corpus_pairs):
        if target_word not in semantic_change_results[pair]:
            print(f"Warning: '{target_word}' not found in transition {pair}")
            continue
            
        changes = semantic_change_results[pair][target_word]
        if not changes:
            print(f"Warning: No changes for '{target_word}' in transition {pair}")
            continue
            
        # Get top words moved towards and away from
        words_towards = [word for word, _ in changes[:n_neighbors]]
        words_away = [word for word, _ in changes[-n_neighbors:]]
        
        context_words_dict[pair] = {
            "towards": words_towards,
            "away": words_away
        }
    
    # Get all unique context words
    all_context_words = set()
    for pair_data in context_words_dict.values():
        all_context_words.update(pair_data["towards"])
        all_context_words.update(pair_data["away"])
    all_context_words = list(all_context_words)
    
    # Check if we have the temporal variants of the target word
    target_variants = []
    for label in all_labels:
        target_variant = f"{target_word}_{label}"
        if target_variant in model.vocab:
            target_variants.append(target_variant)
        else:
            print(f"Warning: '{target_variant}' not found in model vocabulary")
            return
    
    # Get vectors for all context words
    context_vectors = []
    valid_context_words = []
    for word in all_context_words:
        if word in model.vocab:
            context_vectors.append(model.get_vector(word))
            valid_context_words.append(word)
        else:
            print(f"Warning: Context word '{word}' not found in model vocabulary")
    
    if not valid_context_words:
        print(f"No valid context words found for '{target_word}'. Cannot create visualization.")
        return
    
    # Prepare vectors for projection based on the target_position_method
    if target_position_method == 'embedding':
        # Include target vectors in the projection
        target_vectors = [model.get_vector(variant) for variant in target_variants]
        all_vectors = np.vstack([context_vectors, target_vectors])
    else:
        # Only project context words, target positions will be calculated later
        all_vectors = np.array(context_vectors)
    
    # Project vectors to 2D
    if method == 'pca':
        projector = PCA(n_components=2, random_state=seed)
        projected = projector.fit_transform(all_vectors)
        explained_var = projector.explained_variance_ratio_
        x_label = f"PC1 ({explained_var[0]:.1%} variance)"
        y_label = f"PC2 ({explained_var[1]:.1%} variance)"
    elif method == 'tsne':
        projector = TSNE(n_components=2, perplexity=min(perplexity, len(all_vectors) - 1), random_state=seed)
        projected = projector.fit_transform(all_vectors)
        x_label = "t-SNE Dimension 1"
        y_label = "t-SNE Dimension 2"
    elif method == 'umap':
        projector = umap.UMAP(
            n_components=2,
            n_neighbors=min(umap_n_neighbors, len(all_vectors) - 1),
            min_dist=umap_min_dist,
            random_state=seed
        )
        projected = projector.fit_transform(all_vectors)
        x_label = "UMAP Dimension 1"
        y_label = "UMAP Dimension 2"
    
    # Get context positions from projection
    context_positions = projected
    
    # Create mapping from words to positions for neighbor_mean method
    word_to_position = {}
    for i, word in enumerate(valid_context_words):
        word_to_position[word] = context_positions[i]
    
    # Determine target positions based on the method
    if target_position_method == 'embedding':
        # Get target positions directly from projection
        # They're the last len(target_variants) vectors
        target_positions = projected[-len(target_variants):]
        # Adjust context_positions to exclude target positions
        context_positions = projected[:-len(target_variants)]
    else:  # neighbor_mean
        # Calculate target positions based on the mean of relevant context words
        target_positions = []
        
        for i, label in enumerate(all_labels):
            if i == 0:
                # For the first position, use the mean of "away from" words from the first transition
                first_pair = corpus_pairs[0]
                if first_pair in context_words_dict:
                    away_words = context_words_dict[first_pair]["away"]
                    away_positions = [word_to_position[w] for w in away_words if w in word_to_position]
                    
                    if away_positions:
                        target_pos = np.mean(away_positions, axis=0)
                        target_positions.append(target_pos)
                    else:
                        print(f"Warning: No valid 'away from' words for first position. Using center.")
                        target_positions.append(np.mean(context_positions, axis=0))
                else:
                    print(f"Warning: No data for transition {first_pair}. Using center.")
                    target_positions.append(np.mean(context_positions, axis=0))
            else:
                # For later positions, use the mean of "towards" words from the previous transition
                prev_pair = corpus_pairs[i-1]
                if prev_pair in context_words_dict:
                    towards_words = context_words_dict[prev_pair]["towards"]
                    towards_positions = [word_to_position[w] for w in towards_words if w in word_to_position]
                    
                    if towards_positions:
                        target_pos = np.mean(towards_positions, axis=0)
                        target_positions.append(target_pos)
                    else:
                        print(f"Warning: No valid 'towards' words for position {i}. Using center.")
                        target_positions.append(np.mean(context_positions, axis=0))
                else:
                    print(f"Warning: No data for transition {prev_pair}. Using center.")
                    target_positions.append(np.mean(context_positions, axis=0))
        
        target_positions = np.array(target_positions)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot context words
    plt.scatter(context_positions[:, 0], context_positions[:, 1], c='gray', alpha=0.6, label='Context Words')
    
    # Plot target positions
    plt.scatter(target_positions[:, 0], target_positions[:, 1], c='blue', s=100, label='Target Word')
    
    # Add trajectory arrows
    for i in range(len(target_positions) - 1):
        arrow = FancyArrowPatch(
            target_positions[i], target_positions[i + 1],
            arrowstyle=arrowstyle,
            color=arrow_color,
            mutation_scale=arrow_mutation_scale
        )
        plt.gca().add_patch(arrow)
    
    # Add labels
    texts = []
    
    # Add context word labels
    for i, word in enumerate(valid_context_words):
        texts.append(plt.text(context_positions[i, 0], context_positions[i, 1], word,
                            fontsize=fontsize*0.8, alpha=0.7))
    
    # Add target word labels with corpus tags
    for i, label in enumerate(all_labels):
        target_label = f"{target_word}_{label}"
        texts.append(plt.text(target_positions[i, 0], target_positions[i, 1],
                            target_label, fontsize=fontsize, fontweight='bold', c='red'))
    
    if adjust_text_labels and len(texts) > 0:
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        except ImportError:
            print("Warning: adjustText package not found. Install with 'pip install adjustText' for better label placement.")
    
    plt.title(f"Semantic Trajectory of '{target_word}' Across Time Periods", fontsize=fontsize*1.2)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=fontsize*0.8)
    
    # Adjust tick label sizes
    plt.xticks(fontsize=fontsize*0.8)
    plt.yticks(fontsize=fontsize*0.8)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save the plot if filename provided
    if filename:
        plt.savefig(f"{filename}_complete.png", dpi=300, bbox_inches='tight')
        print(f"Saved complete trajectory plot to {filename}_complete.png")
    
    plt.close()

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