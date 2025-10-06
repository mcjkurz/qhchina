---
layout: docs_with_sidebar
title: Documentation
permalink: /pkg_docs/
---

# qhChina Package Documentation

A Python toolkit for computational analysis of Chinese texts in humanities research.

## Installation

```bash
pip install qhchina
```

## Quick Start

```python
import qhchina
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_fonts, load_stopwords

# Load fonts for visualization
load_fonts()

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Create segmenter
segmenter = create_segmenter(
    backend="spacy",
    strategy="sentence",
    filters={"stopwords": stopwords, "min_length": 2}
)

# Segment text
text = "深度学习正在改变自然语言处理。机器学习模型变得越来越强大。"
sentences = segmenter.segment(text)

# Topic modeling
lda = LDAGibbsSampler(n_topics=5, iterations=1000)
lda.fit(sentences)

# Get topics
topics = lda.get_topics(n_words=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```

For detailed information about each module, please refer to the specific documentation pages linked above.
