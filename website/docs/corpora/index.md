---
layout: default
title: Corpus Analysis
permalink: /docs/corpora/
---

# Corpus Analysis with qhChina

This module provides specialized tools for managing and analyzing Chinese text corpora, with features designed for humanities research.

## Overview

The corpus analysis tools in qhChina include:

- Corpus loading and preprocessing for Chinese texts
- Statistical analysis of corpus characteristics
- Text segmentation options for Classical and Modern Chinese
- Concordance and keyword-in-context (KWIC) functionality
- Corpus comparison tools

## Basic Usage

```python
from qhchina.corpora import ChineseCorpus

# Create a corpus from files
corpus = ChineseCorpus.from_files(
    file_paths=['path/to/file1.txt', 'path/to/file2.txt'],
    encoding='utf-8'
)

# Or create from a list of texts
texts = [
    "中国是一个拥有悠久历史的国家。",
    "经济发展对国家繁荣至关重要。",
    # More texts...
]
corpus = ChineseCorpus(texts)

# Get corpus statistics
stats = corpus.get_statistics()
print(f"Total documents: {stats['document_count']}")
print(f"Total characters: {stats['character_count']}")
print(f"Unique characters: {stats['unique_character_count']}")

# Search for a term in the corpus
results = corpus.search("经济")
for doc_id, positions in results.items():
    print(f"Document {doc_id}: found at positions {positions}")

# Get keyword in context
kwic_results = corpus.kwic("经济", window=15)
for result in kwic_results:
    print(f"{result['left_context']} [经济] {result['right_context']}")
```

## Advanced Features

- Document metadata management
- Term frequency and distribution analysis
- Sub-corpus creation and comparison
- Time-based corpus analysis
- Support for different Chinese text segmentation methods

[Full Documentation](/docs/corpora/full) 