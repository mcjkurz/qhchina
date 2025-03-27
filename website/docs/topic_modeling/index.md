---
layout: default
title: Topic Modeling
permalink: /docs/topic_modeling/
---

# Topic Modeling with qhChina

This module provides specialized topic modeling tools designed for Chinese text analysis in humanities research.

## Overview

The qhChina topic modeling module includes:

- LDA (Latent Dirichlet Allocation) with custom preprocessing for Chinese text
- BERTopic implementation with multilingual support
- Topic coherence evaluation metrics
- Topic visualization tools

## Basic Usage

```python
from qhchina.analytics import ChineseTopicModel

# Prepare corpus
documents = [
    "中国经济发展迅速，国内生产总值连年增长。",
    "科技创新是推动经济发展的重要引擎。",
    "教育改革对提高国民素质具有重要意义。",
    # More documents...
]

# Initialize and train the model
model = ChineseTopicModel(
    n_topics=10,
    method='lda',  # or 'bertopic'
    min_word_length=2,
    stopwords=['的', '了', '和', '是', '在']
)

# Train the model
model.fit(documents)

# Get topics
topics = model.get_topics(n_words=10)
for topic_id, words in topics.items():
    print(f"Topic #{topic_id}: {', '.join(words)}")

# Get document-topic distribution
doc_topics = model.get_document_topics(documents[0])

# Visualize topics
model.visualize_topics()
```

## Advanced Features

- Time-based topic modeling
- Hierarchical topic modeling
- Topic similarity analysis
- Custom preprocessing options for Classical and Modern Chinese

[Full Documentation](/docs/topic_modeling/full) 