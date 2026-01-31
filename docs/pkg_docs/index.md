---
layout: docs_with_sidebar
title: Documentation
permalink: /pkg_docs/
functions:
  - name: Quick Start
    anchor: quick-start
  - name: Global Config
    anchor: global-configuration
---

# qhChina Package Documentation

A Python toolkit for computational analysis of Chinese texts in humanities research.

## Installation

```bash
pip install qhchina
```

## Modules

- **Preprocessing**: Chinese text segmentation with multiple backends (spaCy, Jieba, BERT, LLM)
- **Analytics**: Core analysis tools including:
  - *Word2Vec*: Word embedding training and temporal semantic change analysis
  - *Topic Modeling*: LDA with Gibbs sampling for topic discovery
  - *Stylometry*: Authorship attribution and document clustering
  - *Collocations*: Statistical collocation analysis and co-occurrence matrices
- **Educational**: Interactive learning tools and visualizations for basic NLP concepts
- **Helpers**: Font management, text loading, and utility functions

---

## Quick Start

```python
import qhchina
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.analytics.stylometry import Stylometry
from qhchina.helpers import load_fonts, load_stopwords

# Load fonts for visualization
load_fonts()

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter
segmenter = create_segmenter(
    backend="spacy",
    strategy="sentence",
    filters={"stopwords": stopwords, "min_word_length": 2}
)

# Segment text
text = "深度学习正在改变自然语言处理。机器学习模型变得越来越强大。"
sentences = segmenter.segment(text)

# Topic modeling
lda = LDAGibbsSampler(n_topics=5, iterations=20)
lda.fit(sentences)

# Get topics
topics = lda.get_topics(n_words=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```

For detailed information about each module, please refer to the specific documentation pages in the sidebar.

---

## Global Configuration

The package provides centralized configuration for logging and random number generation.

### Random Seed

Set a global random seed for reproducibility across all qhchina modules:

```python
import qhchina

# Set seed for reproducible results
qhchina.set_random_seed(42)

# Check current seed
print(qhchina.get_random_seed())  # 42

# Reset to random behavior
qhchina.set_random_seed(None)
```

The global seed affects all modules that use `get_rng()` internally. Individual functions may also accept a `random_state` or `seed` parameter which overrides the global seed for that specific operation.

For advanced use cases requiring an isolated random number generator:

```python
from qhchina.config import get_rng

# Get numpy RandomState (doesn't affect global numpy.random)
rng = get_rng(42)
rng.random()
rng.randint(0, 100)
rng.shuffle(my_list)  # Works on lists too
```

### Logging

Control the verbosity of package output with `set_log_level()`:

```python
import qhchina

qhchina.set_log_level('WARNING')  # Only show warnings and errors
qhchina.set_log_level('DEBUG')    # Show all messages including debug info
qhchina.set_log_level('SILENT')   # Suppress all messages
qhchina.set_log_level('INFO')     # Default level - informational messages
```
