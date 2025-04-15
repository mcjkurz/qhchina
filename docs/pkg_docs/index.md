---
layout: default
title: Documentation
permalink: /pkg_docs/
---

# qhChina Package Documentation

Welcome to the documentation for the qhChina Python package, a toolkit designed for computational analysis of Chinese texts in humanities research.

## Package Components

Our package includes several modules, each focusing on specific computational approaches to Chinese text analysis:

- [Word Embeddings]({{ site.baseurl }}/pkg_docs/word_embeddings/) - Tools for creating and analyzing word embeddings for Chinese texts
- [Topic Modeling]({{ site.baseurl }}/pkg_docs/topic_modeling/) - Methods for topic modeling with Chinese-specific preprocessing
- [BERT Classifier]({{ site.baseurl }}/pkg_docs/bert_classifier/) - BERT-based text classification for Chinese documents
- [Collocations]({{ site.baseurl }}/pkg_docs/collocations/) - Analysis of word collocations in Chinese texts
- [Corpora]({{ site.baseurl }}/pkg_docs/corpora/) - Tools for managing and processing Chinese text corpora
- [Preprocessing]({{ site.baseurl }}/pkg_docs/preprocessing/) - Text segmentation and tokenization for Chinese texts

## Package Structure

The qhChina package is organized into several key modules:

- `qhchina.analytics` - Core analytical tools including word embeddings, topic modeling, and collocation analysis
- `qhchina.preprocessing` - Text segmentation and preprocessing utilities
- `qhchina.helpers` - Utility functions for file loading, font handling, and more
- `qhchina.educational` - Educational visualization tools

## Getting Started

### Installation

```python
pip install qhchina
```

### Basic Usage

```python
import qhchina

# Load fonts for visualization
qhchina.load_fonts()

# Example with sample Chinese sentences
texts = [
    ["我", "今天", "去", "公园", "散步"],                     # Walking in the park
    ["她", "在", "图书馆", "学习", "汉语"],                   # Studying Chinese at the library
    ["他们", "周末", "喜欢", "做", "中国", "菜"],           # Cooking Chinese food on weekends
    ["这个", "城市", "的", "交通", "很", "方便"],          # City transportation
    ["我的", "家人", "明天", "要", "去", "北京", "旅游"]    # Family travel to Beijing
]

# Example of using topic modeling
from qhchina.analytics import LDAGibbsSampler

lda = LDAGibbsSampler(n_topics=10)
lda.fit(texts)
```

## API Reference

For detailed information about each module, please refer to the specific documentation pages linked above. 