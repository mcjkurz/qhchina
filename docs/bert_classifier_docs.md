---
layout: default
title: BERT Classification - qhChina Documentation
---

# BERT Text Classification with qhChina

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

The qhChina package provides tools for fine-tuning pre-trained BERT models for text classification tasks. Key features include:

- Easy dataset preparation with stratified train/validation/test splits
- A customizable training loop with learning rate scheduling
- Comprehensive evaluation metrics
- Simple prediction interface
- Text encoding with different pooling strategies
- Training visualizations
- Multi-device support (CPU, CUDA, MPS)

## Installation

```bash
pip install qhchina
```

Or install from source:

```bash
git clone https://github.com/mcjkurz/qhchina.git
cd qhchina
pip install -e .
```

## Quick Start

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qhchina.analytics import make_datasets, train_bert_classifier, predict

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=2
)

# Prepare data
data = [
    ("这部电影非常精彩！", 1),  # This movie is excellent!
    ("我讨厌这部电影。", 0),    # I hate this movie.
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
new_texts = ["一部精彩的影片！", "演技很差劲。"]  # A fantastic film!, Terrible acting.
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
    val_interval=None,
    collate_fn=None
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
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.

**Returns:**
- `dict`: Dictionary containing:
  - `model`: Fine-tuned model
  - `history`: Training history (loss, metrics)
  - `metrics`: Final validation metrics
  - `device`: Device used for training

### evaluate()

```python
evaluate(model, test_dataset, batch_size=16, device=None, collate_fn=None)
```

Evaluate a trained model on a test dataset.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Trained model
- `test_dataset` (Dataset): PyTorch Dataset for testing
- `batch_size` (int): Batch size for evaluation
- `device` (Optional[str]): Device to use for evaluation
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.

**Returns:**
- `dict`: Dictionary containing evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

### predict()

```python
predict(model, texts, tokenizer, max_length=128, batch_size=16, device=None, return_probs=False, collate_fn=None)
```

Make predictions with a trained model on new texts.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Trained model
- `texts` (List[str]): List of texts to classify
- `tokenizer` (AutoTokenizer): Tokenizer corresponding to the model
- `max_length` (int): Maximum sequence length for tokenization
- `batch_size` (int): Batch size for prediction
- `device` (Optional[str]): Device to use for prediction
- `return_probs` (bool): Whether to return probability distributions
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.

**Returns:**
- If `return_probs=True`: List of tuples (probabilities, predicted_class)
- If `return_probs=False`: List of predicted classes

### bert_encode()

```python
bert_encode(texts, model, tokenizer, pooling="cls", max_length=128, batch_size=16, device=None, collate_fn=None)
```

Encode texts into BERT embeddings.

**Parameters:**
- `texts` (List[str]): List of texts to encode
- `model`: BERT model (can be a standard BERT model or a classification model)
- `tokenizer` (AutoTokenizer): Tokenizer corresponding to the model
- `pooling` (str): Pooling strategy ('cls' for CLS token, 'mean' for mean pooling)
- `max_length` (int): Maximum sequence length for tokenization
- `batch_size` (int): Batch size for encoding
- `device` (Optional[str]): Device to use for encoding
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.

**Returns:**
- `numpy.ndarray`: Array of shape (len(texts), embedding_dim) with text embeddings

### set_device()

```python
set_device(device=None)
```

Determine the appropriate device for computation.

**Parameters:**
- `device` (Optional[str]): Device specification ('cuda', 'mps', 'cpu', or None)

**Returns:**
- `str`: The selected device ('cuda', 'mps', or 'cpu')

## Examples

### Binary Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qhchina.analytics import make_datasets, train_bert_classifier, evaluate, predict
import torch

# Prepare data: (text, label) pairs
data = [
    ("这部电影非常精彩！", 1),  # Positive
    ("演员的表演很出色。", 1),  # Positive
    ("故事情节有趣。", 1),      # Positive
    ("我讨厌这部电影。", 0),    # Negative
    ("演技很差劲。", 0),        # Negative
    ("浪费时间的电影。", 0),    # Negative
]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=2
)

# Create datasets with stratification
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    max_length=128
)

# Optional: Define a custom collate function
def custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    return encodings

# Train model
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=2,
    learning_rate=2e-5,
    num_epochs=3,
    collate_fn=custom_collate_fn  # Optional: Use custom collate function
)

# Make predictions
new_texts = ["这是一部非常好看的电影", "这个故事很无聊"]
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer,
    collate_fn=custom_collate_fn  # Optional: Use same custom collate function
)
print(f"Predicted classes: {predictions}")
```

### Multi-class Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qhchina.analytics import make_datasets, train_bert_classifier

# Prepare multi-class data: (text, label) pairs
data = [
    ("这是一篇关于体育的新闻。", 0),  # Sports
    ("足球比赛昨天结束了。", 0),     # Sports
    ("经济增长率达到新高。", 1),     # Economy
    ("股市今天上涨了。", 1),        # Economy
    ("新技术改变了我们的生活。", 2), # Technology
    ("人工智能正在迅速发展。", 2),   # Technology
]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=3  # 3 classes
)

# Create datasets
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.7, 0.3)
)

# Train model
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=5
    # If collate_fn is not provided, a default one will be created using the dataset's tokenizer
)
```

### Text Encoding

```python
from transformers import AutoModel, AutoTokenizer
from qhchina.analytics import bert_encode
import torch

# Load base model (not classification model)
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Texts to encode
texts = [
    "中国经济持续发展。",
    "北京是中国的首都。"
]

# Optional: Define a custom collate function for batch processing
def custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

# Encode texts (get BERT embeddings) with batch processing
embeddings = bert_encode(
    texts=texts,
    model=model,
    tokenizer=tokenizer,
    pooling="cls",  # Use CLS token embedding
    batch_size=16,
    collate_fn=custom_collate_fn  # Optional: Use custom collate function
)

print(f"Embedding shape: ({len(embeddings)}, {embeddings[0].shape[0]})")  # (2, 768) for bert-base
```

### Evaluation with Custom Collate Function

```python
from qhchina.analytics import evaluate
import torch

# Define a custom collate function
def custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    return encodings

# Evaluate model
eval_results = evaluate(
    model=model,
    dataset=val_dataset,
    batch_size=16,
    device="cuda",  # Use GPU if available
    collate_fn=custom_collate_fn  # Use custom collate function
)

print(f"Accuracy: {eval_results['accuracy']:.4f}")
print(f"F1 Score (Weighted): {eval_results['weighted_avg']['f1']:.4f}")
```

## Visualizations

The `train_bert_classifier()` function automatically generates training visualizations:

1. **Loss curve**: Shows training loss (and validation loss if available) over time
2. **Metrics curve**: Shows evaluation metrics (accuracy, F1) over time

These visualizations are saved to the specified `logging_dir` if provided.

## Performance Tips

### Batch Size

Choose an appropriate batch size based on your hardware:

- For high-end GPUs: 16-32 (or higher)
- For consumer GPUs: 8-16
- For CPU: 4-8

### Sequence Length

Limit sequence length based on your data needs:

```python
# For efficiency with short texts
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    max_length=64  # Shorter sequences
)

# For long documents that need more context
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    max_length=256  # Longer sequences
)
```

### Learning Rate

Typically best values for BERT fine-tuning:

```python
# Smaller learning rate for stable training
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    learning_rate=1e-5
)

# Default learning rate
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    learning_rate=2e-5
)

# Larger learning rate for faster convergence (may be less stable)
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    learning_rate=5e-5
)
```

### Warmup Steps

Using warmup can improve training stability:

```python
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    warmup_steps=100  # Gradual learning rate warmup
)
``` 