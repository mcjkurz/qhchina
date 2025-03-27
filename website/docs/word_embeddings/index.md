---
layout: default
title: Word Embeddings
permalink: /docs/word_embeddings/
---

# Word Embeddings in qhChina

This page documents the word embeddings functionality in the qhChina package, with a focus on the customized Word2Vec implementation.

## Word2Vec Implementation

qhChina provides a custom implementation of Word2Vec with both CBOW (Continuous Bag of Words) and Skip-gram architectures, designed specifically for research in humanities and social sciences with Chinese text.

### Basic Usage

```python
from qhchina.analytics import Word2Vec

# Initialize a Word2Vec model
model = Word2Vec(
    vector_size=100,  # Dimensionality of word vectors
    window=5,         # Context window size
    min_count=5,      # Minimum word frequency threshold
    sg=1,             # 1 for Skip-gram; 0 for CBOW
    negative=5,       # Number of negative samples
    alpha=0.025,      # Initial learning rate
    seed=42           # Random seed for reproducibility
)

# Prepare tokenized sentences
sentences = [
    ["我", "喜欢", "这部", "电影"],
    ["这", "是", "一个", "有趣", "的", "故事"],
    # More sentences...
]

# Train the model
model.train(sentences, epochs=5)

# Get word vector
vector = model.get_vector("电影")

# Find similar words
similar_words = model.most_similar("电影", topn=10)
```

## Temporal Reference Word2Vec

qhChina provides a specialized implementation called `TempRefWord2Vec` for tracking semantic change over time. This model does not require training separate models for each time period. Instead, it creates temporal variants of target words in a single vector space using a specialized training approach.

### Basic Usage

```python
from qhchina.analytics import TempRefWord2Vec

# Prepare corpus data from different time periods
time_labels = ["1980", "1990", "2000", "2010"]
corpora = [corpus_1980, corpus_1990, corpus_2000, corpus_2010]

# Target words to track for semantic change
target_words = ["改革", "经济", "科技", "人民"]

# Initialize and train the model
model = TempRefWord2Vec(
    corpora=corpora,          # List of corpora for different time periods
    labels=time_labels,       # Labels for each time period
    targets=target_words,     # Words to track for semantic change
    vector_size=256,
    window=5,
    min_count=5,
    sg=1                      # Use Skip-gram model
)

# Access temporal variants of words
reform_1980s = model.get_vector("改革_1980")
reform_2010s = model.get_vector("改革_2010")

# Find words similar to a target in a specific time period
similar_to_reform_1980s = model.most_similar("改革_1980", topn=10)
```

[Full Documentation](/docs/word_embeddings/full) 