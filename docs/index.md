---
layout: default
title: qhChina - Quantitative Humanities China Lab
---

# Overview

A comprehensive toolkit for Chinese NLP research and educational purposes, developed by the Quantitative Humanities China Lab. The package is particularly suited for research in humanities and social sciences, offering tools that balance technical capabilities with ease of use.

[![GitHub stars](https://img.shields.io/github/stars/mcjkurz/qhchina.svg?style=social&label=Star)](https://github.com/mcjkurz/qhchina)
[![GitHub forks](https://img.shields.io/github/forks/mcjkurz/qhchina.svg?style=social&label=Fork)](https://github.com/mcjkurz/qhchina/fork)


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

## Documentation

Detailed documentation for each component:

- [BERT Classification Documentation](bert_classifier_docs.html)
- [Word Embeddings Documentation](word_embeddings_docs.html)
- [Corpus Analysis Documentation](corpora_docs.html)
- [Collocation Analysis Documentation](collocations_docs.html)
- [Topic Modeling Documentation](topic_modeling_docs.html)

## Core Components

### 🔍 Corpus Analysis

- Collocation analysis with various statistical methods
- Corpus comparison tools using Fisher's exact test or chi-square test
- Co-occurrence matrix generation for network analysis

```python
from qhchina.analytics import find_collocates, compare_corpora, cooc_matrix

# Find words that frequently appear near target words
collocates = find_collocates(
    sentences=[["这部", "电影", "非常", "精彩"], ["我", "讨厌", "这部", "电影"]], 
    target_words=["电影"], 
    method="window",
    horizon=3
)

# Compare word usage between two corpora
comparison = compare_corpora(
    corpusA=["这部", "电影", "非常", "精彩"], 
    corpusB=["我", "讨厌", "这部", "电影"]
)
```

### 📊 Word Embeddings

- Custom Word2Vec implementation with CBOW and Skip-gram architectures
- Specialized temporal reference Word2Vec for tracking semantic change
- Vector projection, similarity, and bias direction analysis

```python
from qhchina.analytics import Word2Vec, project_2d, calculate_bias

# Train Word2Vec model on Chinese text
model = Word2Vec(
    vector_size=100,
    window=5,
    min_count=5,
    sg=1  # Use Skip-gram architecture
)
model.build_vocab(sentences)
model.train(sentences, epochs=5)

# Visualize word vectors
project_2d(
    vectors={word: model.get_vector(word) for word in ["电影", "戏剧", "书籍", "电视"]},
    method="pca"
)
```

### 🤖 BERT-based Models

- Fine-tuning BERT models for text classification
- Text encoding with different pooling strategies
- Comprehensive evaluation and visualization tools

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

# Train model
train_dataset, val_dataset = make_datasets(data, tokenizer, split=(0.8, 0.2))
results = train_bert_classifier(model, train_dataset, val_dataset)
```

### 🔧 Utility Tools

- Text preprocessing utilities
- Font management for Chinese text visualization
- Helper functions for data manipulation

## Dependencies

- transformers
- torch
- numpy
- scikit-learn
- matplotlib
- tqdm
- pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details. 