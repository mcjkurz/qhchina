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
  - [get_device()](#get_device)
- [Examples](#examples)
- [Visualizations](#visualizations)
- [Performance Tips](#performance-tips)
- [How Tokenization and Batching Works](#how-tokenization-and-batching-works)
  - [Default Collate Function Behavior](#default-collate-function-behavior)
  - [How max_length is Determined](#how-max_length-is-determined)
  - [Tokenizer Sourcing](#tokenizer-sourcing)
  - [Custom Collate Functions](#custom-collate-functions)

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

# Prepare data (using dictionary format)
data = {
    'text': [
        "这部电影非常精彩！",  # This movie is excellent!
        "我讨厌这部电影。",    # I hate this movie.
        # Add more examples...
    ],
    'label': [1, 0]
}

# Create datasets
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    max_length=128,
    verbose=True  # Print dataset statistics (default behavior)
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
new_texts = [
    "一部精彩的影片！",   # A fantastic film!
    "演技很差劲。",      # Terrible acting.
    "情节发展太慢了。"    # The plot develops too slowly.
]
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

#### Alternative: Using Huggingface's Dataset Methods

While `make_datasets()` provides stratified splitting by default, you can also use Huggingface's `datasets` library methods for dataset creation and splitting:

```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# Example comments data with 1-5 star ratings
comments = [
    {"Comment": "这部电影非常精彩！", "Label": 5},  # Excellent movie! (5 stars)
    {"Comment": "演员的表演很出色。", "Label": 5},   # The actors' performances were outstanding. (5 stars)
    {"Comment": "剧情还不错，但有些地方可以改进。", "Label": 4},  # The plot is good, but some parts could be improved. (4 stars)
    {"Comment": "整体表现一般，不算特别出彩。", "Label": 3},  # Overall performance is average, not particularly impressive. (3 stars)
    {"Comment": "有些无聊，故事发展太慢。", "Label": 2},  # A bit boring, story develops too slowly. (2 stars)
    {"Comment": "我讨厌这部电影。", "Label": 1},     # I hate this movie. (1 star)
    {"Comment": "演技很差劲。", "Label": 1},        # The acting is terrible. (1 star)
    # ... more examples
]

# For binary sentiment analysis, we'll focus on clearly positive (5) vs clearly negative (1) ratings
# One-step filtering and transformation to create dictionary for Dataset
filtered_comments_dict = {
    "text": [elem["Comment"] for elem in comments if elem["Label"] in [1, 5]],
    "label": [1 if elem["Label"] == 5 else 0 for elem in comments if elem["Label"] in [1, 5]]
}

# Create Huggingface Dataset
full_dataset = Dataset.from_dict(filtered_comments_dict)

# Split using Huggingface's train_test_split method (non-stratified)
split_dataset = full_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Train model using qhChina's train_bert_classifier
from qhchina.analytics import train_bert_classifier

results = train_bert_classifier(
    model=bert_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=4,
    device="mps",  # or "cuda" or "cpu"
    num_epochs=1,
    tokenizer=tokenizer,  # Need to provide tokenizer explicitly
    max_length=128
)
```

**Key Differences from `make_datasets()`:**

1. **Stratification**: Huggingface's `train_test_split()` does not provide stratification by default, while `make_datasets()` does. This is important for imbalanced datasets.

2. **Tokenizer Integration**: When using Huggingface's Dataset directly, you need to provide the tokenizer explicitly to `train_bert_classifier()`, as the dataset object doesn't have a tokenizer attribute.

3. **Convenience**: `make_datasets()` handles the creation of `TextDataset` objects with appropriate attributes for you, while with Huggingface's approach you need to handle more details explicitly.

Choose the approach that best suits your workflow and requirements.

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
make_datasets(data, tokenizer, split, max_length=None, random_seed=None, verbose=True)
```

Create train/val/test datasets from input data with stratification.

**Parameters:**
- `data` (Union[List[Tuple[str, int]], Dict[str, List]]): Input data in one of these formats:
  - List of tuples where each tuple contains (text, label)
  - Dictionary with keys 'text' and 'label', where each is a list
- `tokenizer` (AutoTokenizer): Tokenizer to use for text encoding
- `split` (Union[Tuple[float, float], Tuple[float, float, float]]): Tuple of proportions for splits
  - (train_prop, val_prop) for train/val split
  - (train_prop, val_prop, test_prop) for train/val/test split
- `max_length` (Optional[int]): Maximum sequence length for tokenization
- `random_seed` (Optional[int]): Random seed for reproducible splits
- `verbose` (bool): Whether to print dataset statistics and class distributions (default: True)

**Returns:**
- If split has length 2: Tuple of (train_dataset, val_dataset)
- If split has length 3: Tuple of (train_dataset, val_dataset, test_dataset)

**Example using list of tuples:**
```python
data = [
    ("这部电影非常精彩！", 1),  # This movie is excellent!
    ("我讨厌这部电影。", 0),    # I hate this movie.
    # Add more examples...
]

train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    verbose=True
)
```

**Example using dictionary format:**
```python
data = {
    'text': [
        "这部电影非常精彩！",    # This movie is excellent!
        "演员表演令人印象深刻。",  # The actors' performance was impressive.
        "我讨厌这部电影。"       # I hate this movie.
    ],
    'label': [1, 1, 0]
}

train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    split=(0.8, 0.2),
    verbose=False  # Suppress printing of statistics
)
```

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
    collate_fn=None,
    tokenizer=None,
    max_length=None
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
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created using the tokenizer.
- `tokenizer` (Optional[AutoTokenizer]): Tokenizer to use for encoding texts. If None, the train_dataset's tokenizer will be used.
- `max_length` (Optional[int]): Maximum sequence length for tokenization. If None, will try to get from train_dataset.

**Default Collate Function Behavior:**
If `collate_fn` is not provided, a default function is created that:
- Uses the provided `tokenizer` (or from dataset)
- Sets `truncation=True` to handle texts longer than the maximum length
- Sets `padding=True` to pad batches to the same length
- Uses the provided `max_length` (or from dataset, if available)
- Returns tensors with labels if present in the batch

**Returns:**
- `dict`: Dictionary containing:
  - `model`: Fine-tuned model
  - `history`: Training history (loss, metrics)
  - `train_size`: Size of the training dataset
  - `val_size`: Size of the validation dataset (if provided)

### evaluate()

```python
evaluate(model, dataset, batch_size=16, device=None, collate_fn=None, tokenizer=None, max_length=None)
```

Evaluate a trained model on a test dataset.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Trained model
- `dataset` (Dataset): PyTorch Dataset for testing
- `batch_size` (int): Batch size for evaluation
- `device` (Optional[str]): Device to use for evaluation
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.
- `tokenizer` (Optional[AutoTokenizer]): Tokenizer to use for encoding texts. If None, the dataset's tokenizer will be used.
- `max_length` (Optional[int]): Maximum sequence length for tokenization. If None, will try to get from dataset.

**Default Collate Function Behavior:**
If `collate_fn` is not provided, a default function is created that:
- Uses the provided `tokenizer` (or from dataset)
- Sets `truncation=True` to handle texts longer than the maximum length
- Sets `padding=True` to pad batches to the same length
- Uses the provided `max_length` (or from dataset, if available)
- Returns tensors with labels if present in the batch

**Returns:**
- `dict`: Dictionary containing evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

### predict()

```python
predict(model, texts=None, dataset=None, tokenizer=None, batch_size=16, device=None, return_probs=False, collate_fn=None, max_length=None)
```

Make predictions with a trained model on new texts.

**Parameters:**
- `model` (AutoModelForSequenceClassification): Trained model
- `texts` (Optional[List[str]]): List of texts to classify (required if dataset is None)
- `dataset` (Optional[Dataset]): PyTorch Dataset for inference (required if texts is None)
- `tokenizer` (Optional[AutoTokenizer]): Tokenizer for encoding texts
  - Required if texts is provided and no collate_fn is provided
  - If dataset is provided and has a tokenizer attribute, that tokenizer will be used if this is None
- `batch_size` (int): Batch size for prediction
- `device` (Optional[str]): Device to use for prediction
- `return_probs` (bool): Whether to return probability distributions
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.
- `max_length` (Optional[int]): Maximum sequence length for tokenization. If None, will try to get from dataset.

**Default Collate Function Behavior:**
If `collate_fn` is not provided, a default function is created that:
- Uses the provided `tokenizer` (or from dataset)
- Sets `truncation=True` to handle texts longer than the maximum length
- Sets `padding=True` to pad batches to the same length
- Uses the provided `max_length` (or from dataset, if available)
- Returns tensors with labels if present in the batch

**Returns:**
- If `return_probs=False`: List of predicted labels
- If `return_probs=True`: Dictionary containing:
  - 'predictions': List of predicted labels
  - 'probabilities': List of probability distributions

### bert_encode()

```python
bert_encode(model, texts, tokenizer=None, batch_size=None, max_length=None, pooling_strategy="cls", device=None, collate_fn=None)
```

Encode texts into BERT embeddings.

**Parameters:**
- `model`: BERT model (can be a standard BERT model or a classification model)
- `texts` (List[str]): List of texts to encode
- `tokenizer` (Optional[AutoTokenizer]): Tokenizer to use for encoding texts. If None, must provide a collate_fn
- `batch_size` (Optional[int]): Batch size for encoding. If None, process texts individually.
- `max_length` (Optional[int]): Maximum sequence length for tokenization. If None, will first try to get from dataset, then from model config.
- `pooling_strategy` (str): Pooling strategy ('cls' for CLS token, 'mean' for mean pooling)
- `device` (Optional[str]): Device to use for encoding
- `collate_fn` (Optional[Callable]): Custom collation function for DataLoader. If None, a default function will be created.

**Default Collate Function Behavior:**
If `collate_fn` is not provided, a default function is created that:
- Uses the provided `tokenizer` (required for this function)
- Sets `truncation=True` to handle texts longer than the maximum length
- Sets `padding=True` to pad batches to the same length
- Uses the provided `max_length` (or model's max position embeddings if not provided)
- Returns tensors ready for embedding extraction

**Returns:**
- List of numpy arrays, each of shape (embedding_dim)

### get_device()

```python
get_device(device=None)
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

# Prepare data using dictionary format
data = {
    'text': [
        "这部电影非常精彩！",  # Positive
        "演员的表演很出色。",  # Positive
        "故事情节有趣。",      # Positive
        "我讨厌这部电影。",    # Negative
        "演技很差劲。",        # Negative
        "浪费时间的电影。",    # Negative
    ],
    'label': [1, 1, 1, 0, 0, 0]
}

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
    max_length=128,
    verbose=True  # Print statistics (default)
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
new_texts = [
    "这是一部非常好看的电影",    # This is a very good movie
    "这个故事很无聊",          # This story is boring
    "画面很美，但剧情一般"      # Beautiful visuals, but average plot
]
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

### Binary Classification with External Tokenizer

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

# Create datasets without specifying tokenizer in the datasets
train_texts, train_labels = zip(*data[:4])
val_texts, val_labels = zip(*data[4:])

# Create TextDataset without tokenizer
from qhchina.analytics import TextDataset
train_dataset = TextDataset(train_texts, labels=train_labels)
val_dataset = TextDataset(val_texts, labels=val_labels)

# Train model with explicitly provided tokenizer
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=2,
    learning_rate=2e-5,
    num_epochs=3,
    tokenizer=tokenizer  # Explicitly provide tokenizer
)

# Make predictions with external tokenizer
new_texts = [
    "这是一部非常好看的电影",    # This is a very good movie
    "这个故事很无聊",          # This story is boring
    "画面很美，但剧情一般"      # Beautiful visuals, but average plot
]
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer  # Explicitly provide tokenizer
)

print(f"Predicted classes: {predictions}")
```

### Custom Collate Function with External Tokenizer

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qhchina.analytics import TextDataset, train_bert_classifier

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=2
)

# Create a dataset without tokenizer
train_texts = ["这部电影很好看", "这个故事很无趣", "演员表演出色"]
train_labels = [1, 0, 1]
train_dataset = TextDataset(train_texts, labels=train_labels)

# Define a custom collate function that uses the tokenizer
def custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    # Add special tokens and other custom processing
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # Add labels
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    
    return encodings

# Train the model with custom collate function
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    batch_size=2,
    num_epochs=2,
    collate_fn=custom_collate_fn,  # Use custom collate function
    tokenizer=tokenizer  # Also provide tokenizer for evaluation
)
```

### Text Encoding with Explicit Tokenizer

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

# Option 1: Use tokenizer explicitly
embeddings = bert_encode(
    model=model,
    texts=texts,
    tokenizer=tokenizer,  # Explicitly provide tokenizer
    pooling_strategy="cls",
    batch_size=16
)

# Option 2: Use custom collate function
def custom_embedding_collate_fn(batch):
    texts = [item['text'] for item in batch]
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

embeddings = bert_encode(
    model=model,
    texts=texts,
    batch_size=16,
    collate_fn=custom_embedding_collate_fn  # Use custom collate function
)

print(f"Embedding shape: ({len(embeddings)}, {embeddings[0].shape[0]})")  # (2, 768) for bert-base
```

### Binary Classification with Custom Max Length

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
    max_length=64  # Explicitly set maximum sequence length to 64 tokens
)

# Train model with explicit max_length
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=2,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=64  # Explicitly provide max_length for tokenization in default collate_fn
)

# Make predictions with explicit max_length
new_texts = [
    "这是一部非常好看的电影",    # This is a very good movie
    "这个故事很无聊",          # This story is boring
    "画面很美，但剧情一般"      # Beautiful visuals, but average plot
]
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer,
    max_length=64  # Explicitly provide max_length for consistent tokenization
)
print(f"Predicted classes: {predictions}")
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

Limit sequence length based on your data needs using the `max_length` parameter:

```python
# For efficiency with short texts
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    max_length=64  # Shorter sequences save memory and compute time
)

# For long documents that need more context
train_dataset, val_dataset = make_datasets(
    data=data,
    tokenizer=tokenizer,
    max_length=256  # Longer sequences preserve more context
)
```

You can pass the same `max_length` value to the training, evaluation, and prediction functions to ensure consistent tokenization:

```python
# Train with consistent max_length across all operations
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    max_length=128  # Uses this value in default collate_fn
)

# Evaluate with the same max_length
eval_results = evaluate(
    model=results["model"],
    dataset=test_dataset,
    max_length=128  # Keep consistent with training
)

# Predict with the same max_length
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer,
    max_length=128  # Keep consistent with training and evaluation
)
```

**Note**: If you don't provide `max_length`, the system will:
1. Try to get it from the dataset's `max_length` attribute
2. For `bert_encode`, if still not found, fall back to the model's configuration (`max_position_embeddings`)
3. Default to using the tokenizer's default behavior if no value is found

### Custom Collation Functions

For advanced use cases, you can provide your own `collate_fn` to control exactly how batches are created:

```python
def my_custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    
    # Custom preprocessing can be done here
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',  # Always pad to max_length instead of longest in batch
        max_length=128,
        return_tensors='pt',
        return_token_type_ids=True  # Include token_type_ids
    )
    
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    
    return encodings
}

# Use custom collate function
results = train_bert_classifier(
    model=model,
    train_dataset=train_dataset,
    collate_fn=my_custom_collate_fn
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

## How Tokenization and Batching Works

### Default Collate Function Behavior

When using the `train_bert_classifier()`, `evaluate()`, `predict()`, or `bert_encode()` functions without providing a custom `collate_fn`, a default collate function is automatically created. This function handles the conversion of raw text to tokenized tensors needed by the BERT model.

The default collate function:

1. **Extracts text from batch**: Gathers all text items from the current batch
2. **Tokenizes with specific settings**:
   - Always applies `truncation=True` to handle texts longer than the maximum length
   - Always applies `padding=True` to pad all texts in a batch to the same length
   - Uses the `max_length` parameter if provided (see next section for details)
   - Returns tensors in PyTorch format with `return_tensors='pt'`
   - Excludes token type IDs with `return_token_type_ids=False`
3. **Adds labels if present**: If the batch contains labels, they are added to the tokenized output

Here's how the default collate function is created in all functions:

```python
def default_collate_fn(batch):
    texts = [item['text'] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
        return_token_type_ids=False
    )
    
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    
    return encodings
```

### How max_length is Determined

The `max_length` parameter determines the maximum sequence length for tokenization. This is important as it affects both memory usage and model performance. All functions follow a consistent approach to determining the `max_length`:

1. **Explicitly provided value**: If you set `max_length` when calling the function, this value is used
2. **Dataset attribute**: If `max_length` is not provided but the dataset has a `max_length` attribute, that value is used
3. **Model configuration**: For `bert_encode()`, if neither of the above are available, it falls back to the model's `max_position_embeddings` (typically 512 for BERT)
4. **Tokenizer default**: If none of the above are available, the tokenizer's default behavior is used

Example processing flow:

```python
# Explicit value takes precedence
if max_length is provided:
    use the provided max_length
# Otherwise, try the dataset
elif hasattr(dataset, 'max_length'):
    max_length = dataset.max_length
# For bert_encode, also try the model config
elif function is bert_encode:
    max_length = getattr(model.config, 'max_position_embeddings', 512)
# Otherwise, rely on tokenizer defaults
```

### Tokenizer Sourcing

Similarly, a tokenizer is necessary for the default collate function. The functions look for a tokenizer in the following order:

1. **Explicitly provided tokenizer**: If you pass a `tokenizer` parameter to the function
2. **Dataset attribute**: If the dataset has a `tokenizer` attribute (e.g., when created with `make_datasets()`)

If neither source is available and no custom `collate_fn` is provided, the functions will raise a `ValueError` with a clear message.

### Custom Collate Functions

You can override the default behavior by providing your own `collate_fn`. This gives you complete control over how batches are processed. Your custom function should:

1. Accept a batch of data items from the dataset
2. Convert the texts to a format suitable for the model
3. Return a dictionary with the appropriate tensors

Here's a custom collate function example with different tokenization settings:

```python
def my_custom_collate_fn(batch):
    texts = [item['text'] for item in batch]
    
    # Custom preprocessing can be done here
    processed_texts = [text.lower() for text in texts]  # Example: lowercase all texts
    
    encodings = tokenizer(
        processed_texts,
        truncation=True,
        padding='max_length',  # Always pad to max_length instead of longest in batch
        max_length=128,        # Hard-coded max length
        return_tensors='pt',
        return_token_type_ids=True  # Include token_type_ids
    )
    
    if 'label' in batch[0]:
        encodings['labels'] = torch.tensor([item['label'] for item in batch])
    
    return encodings
``` 