---
layout: default
title: BERT Text Classification
---

# BERT Text Classification

A comprehensive toolkit for fine-tuning BERT models for text classification tasks with PyTorch.

[![GitHub stars](https://img.shields.io/github/stars/mcjkurz/qhchina.svg?style=social&label=Star)](https://github.com/mcjkurz/qhchina)
[![GitHub forks](https://img.shields.io/github/forks/mcjkurz/qhchina.svg?style=social&label=Fork)](https://github.com/mcjkurz/qhchina/fork)

## Features

- Easy dataset preparation with stratified train/validation/test splits
- A customizable training loop with learning rate scheduling
- Comprehensive evaluation metrics
- Simple prediction interface
- Text encoding with different pooling strategies
- Training visualizations
- Multi-device support (CPU, CUDA, MPS)

[View the full documentation](bert_classifier_docs.html)

## Quick Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from analytics.modeling import make_datasets, train_bert_classifier, predict

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
    device="cuda"  # or "cpu" or "mps"
)

# Make predictions
new_texts = ["一部精彩的影片！", "演技很差劲。"]  # A fantastic film!, Terrible acting.
predictions = predict(
    model=results["model"],
    texts=new_texts,
    tokenizer=tokenizer
)
```

## Installation

```bash
pip install transformers torch numpy scikit-learn matplotlib tqdm
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 