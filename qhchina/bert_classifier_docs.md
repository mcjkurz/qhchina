# BERT Text Classification

A comprehensive toolkit for fine-tuning BERT models for text classification tasks with PyTorch.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [Dataset Creation](#dataset-creation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Making Predictions](#making-predictions)
  - [Text Encoding](#text-encoding)
- [API Reference](#api-reference)
  - [TextDataset](#textdataset)
  - [make_datasets()](#make_datasets)
  - [train_bert_classifier()](#train_bert_classifier)
  - [evaluate()](#evaluate)
  - [predict()](#predict)
  - [bert_encode()](#bert_encode)
  - [set_device()](#set_device)
- [Examples](#examples)
- [Visualizations](#visualizations)
- [Performance Tips](#performance-tips)

## Overview

This package provides tools for fine-tuning pre-trained BERT models for text classification tasks. Key features include:

- Easy dataset preparation with stratified train/validation/test splits
- A customizable training loop with learning rate scheduling
- Comprehensive evaluation metrics
- Simple prediction interface
- Text encoding with different pooling strategies
- Training visualizations
- Multi-device support (CPU, CUDA, MPS)

## Installation

```bash
pip install transformers torch numpy scikit-learn matplotlib tqdm
```

## Quick Start

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from analytics.modeling import make_datasets, train_bert_classifier, predict

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

# Prepare data
data = [
    ("This movie is great!", 1),
    ("I hated this movie.", 0),
    # Add more examples...
]

# Create datasets
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    max_length=128
)

# Train model
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    device="cuda",  # or "cpu" or "mps"
    logging_dir="./logs"
)

# Make predictions
new_texts = ["A fantastic film!", "Terrible acting."]
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer,
    return_probs=True
)
```

## Core Components

### Dataset Creation

The package includes a `TextDataset` class that inherits from PyTorch's `Dataset` class, making it compatible with PyTorch's `DataLoader`. The `make_datasets()` function creates stratified train/validation/test splits from raw text data.

### Model Training

The `train_bert_classifier()` function provides a customizable training loop with:

- Learning rate scheduling with warmup
- Training and validation metrics tracking
- Batch-level and epoch-level progress tracking
- Visualization of training curves
- Model checkpointing

### Model Evaluation

The `evaluate()` function computes comprehensive evaluation metrics including:

- Accuracy
- Precision, recall, and F1 score (weighted and macro averages)
- Per-class metrics
- Confusion matrix

### Making Predictions

The `predict()` function provides an easy-to-use interface for making predictions on new texts, with options to return class labels or probability distributions.

### Text Encoding

The `bert_encode()` function extracts vector representations from texts using different pooling strategies (CLS token or mean pooling).

## API Reference

### TextDataset

```python
TextDataset(texts, tokenizer, max_length=None, labels=None)
```

A PyTorch Dataset for text classification tasks.

**Parameters:**
- `texts` (List[str]): List of raw texts
- `tokenizer` (AutoTokenizer): Tokenizer to use for encoding texts
- `max_length` (Optional[int]): Maximum sequence length for tokenization
- `labels` (Optional[List[int]]): Optional list of labels

### make_datasets()

```python
make_datasets(data, tokenizer, split, max_length=None, random_seed=None)
```

Create train/val/test datasets from a list of (text, label) tuples with stratification.

**Parameters:**
- `data` (List[Tuple[str, int]]): List of tuples where each tuple contains (text, label)
- `tokenizer` (AutoTokenizer): Tokenizer to use for text encoding
- `split` (Union[Tuple[float, float], Tuple[float, float, float]]): Tuple of proportions for splits
  - (train_prop, val_prop) for train/val split
  - (train_prop, val_prop, test_prop) for train/val/test split
- `max_length` (Optional[int]): Maximum sequence length for tokenization
- `random_seed` (Optional[int]): Random seed for reproducible splits

**Returns:**
- If split has length 2: Tuple of (train_dataset, val_dataset)
- If split has length 3: Tuple of (train_dataset, val_dataset, test_dataset)

### train_bert_classifier()

```python
train_bert_classifier(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=1,
    device=None,
    warmup_steps=0,
    max_train_batches=None,
    logging_dir=None,
    save_dir=None,
    plot_interval=100,
    val_interval=None
)
```

Train a BERT-based classifier with custom training loop.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Pre-loaded BERT model for classification
- `train_dataset` (Dataset): PyTorch Dataset for training
- `val_dataset` (Optional[Dataset]): Optional PyTorch Dataset for validation
- `batch_size` (int): Training batch size
- `learning_rate` (float): Learning rate for training
- `num_epochs` (int): Number of training epochs
- `device` (Optional[str]): Device to train on ('cuda', 'mps', or 'cpu')
- `warmup_steps` (int): Number of warmup steps for learning rate scheduler
- `max_train_batches` (Optional[int]): Maximum number of training batches per epoch
- `logging_dir` (Optional[str]): Directory to save training logs
- `save_dir` (Optional[str]): Directory to save model checkpoints
- `plot_interval` (int): Number of batches between plot updates
- `val_interval` (Optional[int]): Number of batches between validation runs

**Returns:**
Dictionary containing:
- `model`: Trained model
- `history`: Training history (loss and metrics)
- `train_size`: Number of training examples
- `val_size`: Number of validation examples (if applicable)

### evaluate()

```python
evaluate(model, dataset, batch_size=32, device=None)
```

Evaluate a BERT-based classifier on a dataset.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Pre-loaded BERT model for classification
- `dataset` (Dataset): PyTorch Dataset to evaluate
- `batch_size` (int): Batch size for evaluation
- `device` (Optional[str]): Device to evaluate on ('cuda', 'mps', or 'cpu')

**Returns:**
Dictionary containing evaluation metrics:
- `loss`: Average loss
- `accuracy`: Overall accuracy
- `weighted_avg`: Weighted precision, recall, and F1 scores
- `macro_avg`: Macro precision, recall, and F1 scores
- `class_metrics`: Per-class metrics
- `confusion_matrix`: Confusion matrix
- `predictions`: All predictions
- `true_labels`: All true labels

### predict()

```python
predict(
    model,
    texts=None,
    dataset=None,
    tokenizer=None,
    batch_size=32,
    device=None,
    return_probs=False
)
```

Make predictions on new texts using a trained BERT classifier.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Pre-loaded BERT model for classification
- `texts` (Optional[List[str]]): List of texts to predict
- `dataset` (Optional[Dataset]): PyTorch Dataset to use for inference
- `tokenizer` (Optional[AutoTokenizer]): Pre-loaded tokenizer (required if texts is provided)
- `batch_size` (int): Batch size for inference
- `device` (Optional[str]): Device to run inference on ('cuda', 'mps', or 'cpu')
- `return_probs` (bool): Whether to return prediction probabilities

**Returns:**
- If `return_probs` is False: List of predicted labels
- If `return_probs` is True: Dictionary with 'predictions' (labels) and 'probabilities'

### bert_encode()

```python
bert_encode(
    model,
    tokenizer,
    texts,
    batch_size=None,
    max_length=None,
    pooling_strategy='cls',
    device=None
)
```

Extract embeddings from a transformer model for given text(s).

**Parameters:**
- `model` (AutoModelForSequenceClassification): Pre-loaded transformer model
- `tokenizer` (AutoTokenizer): Pre-loaded tokenizer
- `texts` (List[str]): List of texts to encode
- `batch_size` (Optional[int]): If None, process texts individually. If provided, process in batches
- `max_length` (Optional[int]): Maximum sequence length for tokenization
- `pooling_strategy` (str): Strategy for pooling embeddings ('cls' or 'mean')
- `device` (Optional[str]): Device to run inference on ('cuda', 'mps', or 'cpu')

**Returns:**
- List of numpy arrays, each of shape (hidden_size,)

### set_device()

```python
set_device(device=None)
```

Determine the appropriate device for computation.

**Parameters:**
- `device` (Optional[str]): Optional device specification ('cuda', 'mps', 'cpu', or None)

**Returns:**
- str: The selected device ('cuda', 'mps', or 'cpu')

## Examples

### Binary Sentiment Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from analytics.modeling import make_datasets, train_bert_classifier, evaluate, predict

# Prepare data: (text, label) pairs
data = [
    ("Absolutely loved this movie!", 1),
    ("The acting was terrible", 0),
    # More examples...
]

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2  # Binary classification
)

# Create datasets with 80/20 split
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    max_length=128
)

# Train model
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    device="cuda",
    warmup_steps=100,
    logging_dir="./logs",
    save_dir="./checkpoints"
)

# Evaluate final model
eval_results = evaluate(
    model=results["model"],
    dataset=val_dataset,
    batch_size=32
)

print(f"Accuracy: {eval_results['accuracy']:.4f}")
print(f"F1 Score: {eval_results['weighted_avg']['f1']:.4f}")

# Make predictions
new_texts = [
    "I would recommend this movie to everyone!",
    "Don't waste your time on this film."
]

predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer,
    return_probs=True
)

for text, pred, probs in zip(new_texts, 
                             predictions["predictions"], 
                             predictions["probabilities"]):
    print(f"Text: {text}")
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Confidence: {max(probs)*100:.1f}%\n")
```

### Multi-class Classification

```python
# For a multi-class scenario (e.g., topic classification)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=4  # 4 classes
)

# Then follow similar steps as above
```

## Visualizations

The training process automatically generates visualizations when `logging_dir` is provided:

- **Training curves**: Shows training loss and validation loss over time
- **Learning rate**: Visualizes the learning rate schedule with warmup

## Performance Tips

1. **Hardware acceleration**: Use CUDA or MPS for faster training
2. **Batch size optimization**: Use the largest batch size that fits in your GPU memory
3. **Learning rate tuning**: Start with 2e-5 and adjust based on validation performance
4. **Warmup steps**: For large datasets, use ~10% of total steps as warmup
5. **Sequence length**: Use the shortest sequence length that captures the necessary context
6. **Memory optimization**: Set `max_train_batches` for quick experiments 