---
layout: default
title: Collocation Analysis
permalink: /qhchina_docs/collocations/
---

# Collocation Analysis with qhChina

This module provides tools for identifying and analyzing word collocations in Chinese texts, with specialized features for humanities research.

## Overview

The collocation analysis tools in qhChina include:

- Advanced statistical measures for collocation identification
- Customized window and distance settings
- Visualization options for collocation networks
- Tools for comparing collocations across different time periods or text collections

## Basic Usage

```python
from qhchina.analytics import CollocationAnalyzer

# Prepare tokenized sentences
sentences = [
    ["中国", "经济", "发展", "迅速"],
    ["改革", "开放", "推动", "经济", "发展"],
    # More sentences...
]

# Initialize analyzer
analyzer = CollocationAnalyzer(
    min_count=5,       # Minimum frequency of words
    window_size=5,     # Context window size
    scoring='pmi'      # Scoring method: 'pmi', 't_score', 'chi_sq', etc.
)

# Train the analyzer
analyzer.fit(sentences)

# Find collocations for a word
economy_collocations = analyzer.get_collocations("经济")
for word, score in economy_collocations:
    print(f"{word}: {score:.4f}")

# Get the most significant collocations in the corpus
top_collocations = analyzer.get_top_collocations(n=20)

# Visualize collocation network
analyzer.visualize_network("经济", min_edge_weight=3.0)
```

## Advanced Features

- Directional collocation analysis (left/right context)
- Part-of-speech filtered collocations
- Temporal collocation comparison
- Collocation significance testing

[Full Documentation](/qhchina_docs/collocations/full) 