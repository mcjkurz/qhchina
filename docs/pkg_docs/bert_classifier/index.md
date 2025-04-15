---
layout: default
title: BERT Classification
permalink: /pkg_docs/bert_classifier/
---

# BERT Text Classification with qhChina

A comprehensive toolkit for fine-tuning BERT models for text classification tasks with PyTorch.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [SequenceClassifier](#sequenceclassifier)
  - [Dataset Creation](#dataset-creation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Making Predictions](#making-predictions)
- [API Reference](#api-reference)
  - [SequenceClassifier](#sequenceclassifier-api)
- [Examples](#examples)
- [Performance Tips](#performance-tips)

## Overview

The qhChina package provides tools for fine-tuning pre-trained BERT models for text classification tasks. Key features include:

- A simple, unified `SequenceClassifier` class for all classification needs
- Easy-to-use API for training, evaluation, and prediction
- Multi-device support (CPU, CUDA)
- Comprehensive evaluation metrics
- Ability to save and load trained models

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
from qhchina.analytics import SequenceClassifier
from transformers import AutoTokenizer
from datasets import Dataset

# Create a classifier with a Chinese BERT model
classifier = SequenceClassifier(
    model_name="bert-base-chinese", 
    num_labels=2
)

# Prepare example data
texts = [
    "这部电影非常精彩！",  # This movie is excellent!
    "剧情太无聊了。",      # The plot is too boring.
    "演员的表演很出色。",  # The actors' performances were outstanding.
    "故事情节很有趣。",    # The storyline is interesting.
    "导演的处理手法很棒。", # The director's approach is great.
    "我讨厌这部电影。",    # I hate this movie.
    "演技非常差。",        # The acting is very poor.
]
labels = [1, 0, 1, 1, 1, 0, 0]  # 1 for positive, 0 for negative

# Train the classifier
classifier.train(
    texts=texts,
    labels=labels,
    val_split=0.2,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    output_dir="./model_output"
)

# Make predictions
new_texts = [
    "一部精彩的影片！",   # A fantastic film!
    "演技很差劲。",      # Terrible acting.
    "情节发展太慢了。"    # The plot develops too slowly.
]
predictions = classifier.predict(new_texts)
print("Predictions:", predictions)

# Get prediction probabilities
predictions, probabilities = classifier.predict(new_texts, return_probs=True)
print("Predictions with probabilities:", list(zip(predictions, probabilities)))

# Evaluate on a test set
test_texts = ["这个故事非常有趣", "电影不够吸引人"]
test_labels = [1, 0]
metrics = classifier.evaluate(test_texts, test_labels)
print("Evaluation metrics:", metrics)

# Save the trained model
classifier.save("./saved_classifier")
```

## Core Components

### SequenceClassifier

The `SequenceClassifier` class is the main component for text classification tasks. It wraps Hugging Face's transformers library and provides a simplified API for training, evaluation, and prediction.

```python
from qhchina.analytics.classification import SequenceClassifier

# Initialize with a pre-trained BERT model
classifier = SequenceClassifier(
    model_name="bert-base-chinese",
    num_labels=2,
    max_length=128,
    batch_size=16
)
```

### Dataset Creation

The `SequenceClassifier` automatically handles dataset creation internally. You only need to provide the texts and labels for training:

```python
# Training with automatic validation split
classifier.train(
    texts=your_texts,
    labels=your_labels,
    val_split=0.2  # 20% of data used for validation
)
```

### Training

The training process is handled by the `train` method, which includes:

- Automatic dataset creation with validation splitting
- Training with customizable parameters
- Evaluation after each epoch or at specified intervals
- Saving the best model based on validation performance

```python
classifier.train(
    texts=training_texts,
    labels=training_labels,
    val_split=0.2,
    epochs=3,
    batch_size=16,
    output_dir="./model_output",
    learning_rate=2e-5,
    eval_interval=100  # Evaluate every 100 steps instead of per epoch
)
```

### Evaluation

The `evaluate` method provides detailed metrics for model performance:

```python
metrics = classifier.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

### Making Predictions

The `predict` method allows you to make predictions on new texts:

```python
# Basic predictions (returns class indices)
predictions = classifier.predict(new_texts)

# Get prediction probabilities
predictions, probabilities = classifier.predict(
    new_texts,
    return_probs=True
)

# Process predictions with class names
class_names = ["Negative", "Positive"]
for text, pred, probs in zip(new_texts, predictions, probabilities):
    print(f"Text: {text}")
    print(f"Prediction: {class_names[pred]} ({probs[pred]:.4f})")
    print("---")
```

## API Reference

### SequenceClassifier API

```python
class SequenceClassifier:
    def __init__(self, model_name=None, num_labels=None):
        """Initialize the classifier with a pretrained model and tokenizer.
        
        Args:
            model_name: str - The name or path of the pretrained model to use
            num_labels: int - The number of labels for the classification task
        """
        
    def train(self, texts, labels, val_split=0.2, epochs=3, batch_size=16, 
              output_dir="./results", learning_rate=2e-5, eval_interval=None):
        """Train the classifier on text data.
        
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
        
    def predict(self, texts, batch_size=16, return_probs=False):
        """Make predictions on new texts.
        
        Args:
            texts: String or list of strings to classify
            batch_size: Batch size for prediction
            return_probs: Whether to return the probabilities of the predictions
            
        Returns:
            If return_probs=False: List of predicted label indices
            If return_probs=True: Tuple of (predictions, probabilities)
        """
        
    def evaluate(self, texts, labels, batch_size=16):
        """Evaluate the model on a set of texts and labels.
        
        Args:
            texts: List of texts to evaluate on
            labels: List of corresponding labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics (accuracy, precision, recall, f1)
        """
        
    def save(self, path):
        """Save the model and tokenizer.
        
        Args:
            path: Path to save the model and tokenizer
        """
        
    @classmethod
    def load(cls, path):
        """Load a model and tokenizer from a path.
        
        Args:
            path: Path where the model and tokenizer are saved
            
        Returns:
            SequenceClassifier instance with loaded model and tokenizer
        """
```

## Examples

### Multilingual Classification

```python
# Create a classifier with a multilingual model
classifier = SequenceClassifier(
    model_name="xlm-roberta-base",
    num_labels=2
)

# Prepare data in multiple languages
texts = [
    "这部电影非常精彩！",  # Chinese: This movie is excellent!
    "This movie is excellent!",  # English
    "Ce film est excellent !",  # French
    "Diese Film ist ausgezeichnet!",  # German
    "剧情太无聊了。",      # Chinese: The plot is too boring.
    "The plot is too boring.",  # English
    "L'intrigue est trop ennuyeuse.",  # French
    "Die Handlung ist zu langweilig.",  # German
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for positive, 0 for negative

# Train the multilingual classifier
classifier.train(
    texts=texts,
    labels=labels,
    epochs=3
)

# Test on new languages
test_texts = [
    "Aquesta pel·lícula és excel·lent!",  # Catalan
    "Esta película es excelente!",  # Spanish
    "Denne filmen er utmerket!",  # Norwegian
    "Questa pellicola è noiosa.",  # Italian: This film is boring
]
predictions = classifier.predict(test_texts)
```

### Multi-class Classification

```python
# Create a multi-class classifier
classifier = SequenceClassifier(
    model_name="bert-base-chinese",
    num_labels=3
)

# Prepare example data for 3 classes (0: Negative, 1: Neutral, 2: Positive)
texts = [
    "这部电影非常精彩！",     # This movie is excellent! (Positive)
    "剧情太无聊了。",        # The plot is too boring. (Negative)
    "这部电影一般般。",      # This movie is average. (Neutral)
    "电影还行，不好不坏。",   # The movie is okay, not good or bad. (Neutral)
    "演员的表演很出色。",     # The actors' performances were outstanding. (Positive)
    "我讨厌这部电影。",      # I hate this movie. (Negative)
]
labels = [2, 0, 1, 1, 2, 0]  # 0: Negative, 1: Neutral, 2: Positive

# Train the classifier
classifier.train(
    texts=texts,
    labels=labels,
    epochs=3
)

# Make predictions
new_texts = [
    "一部精彩的影片！",   # A fantastic film!
    "电影不好也不坏。",   # The movie is neither good nor bad.
    "演技很差劲。"       # Terrible acting.
]

# Get predictions with probabilities
preds, probs = classifier.predict(new_texts, return_probs=True)

# Map predictions to meaningful labels
class_names = ["Negative", "Neutral", "Positive"]
for text, pred, prob in zip(new_texts, preds, probs):
    print(f"Text: {text}")
    print(f"Prediction: {class_names[pred]}")
    print(f"Probabilities: Negative: {prob[0]:.2f}, Neutral: {prob[1]:.2f}, Positive: {prob[2]:.2f}")
    print("---")
```

## Performance Tips

- For Chinese text, use specialized models like `bert-base-chinese` or `chinese-roberta-wwm-ext`
- Use a smaller learning rate (1e-5 to 5e-5) for more stable training
- Adjust batch size based on GPU memory (larger batches can improve training stability)
- For longer texts, consider increasing max_length (default is typically 512 tokens for BERT)
- For very imbalanced datasets, consider using weighted loss or data augmentation
- To improve performance on low-resource languages, start with a multilingual model like XLM-RoBERTa