---
layout: default
title: BERT Classification
permalink: /qhchina_docs/bert_classifier/
---

# BERT Text Classification with qhChina

A comprehensive toolkit for fine-tuning BERT models for text classification tasks with PyTorch.

## Overview

The qhChina package provides tools for fine-tuning pre-trained BERT models for text classification tasks. Key features include:

- Easy dataset preparation with stratified train/validation/test splits
- A customizable training loop with learning rate scheduling
- Comprehensive evaluation metrics
- Simple prediction interface
- Text encoding with different pooling strategies
- Training visualizations
- Multi-device support (CPU, CUDA, MPS)

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
    num_epochs=3
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

[Full Documentation](/qhchina_docs/bert_classifier/full) 