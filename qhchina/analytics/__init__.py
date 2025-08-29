"""Analytics module for text and vector operations.

This module provides tools for:
- Collocation analytics
- Corpus comparison
- Vector operations and projections
- Topic modeling

To use specific functionality, import directly from the appropriate submodule:
- from qhchina.analytics.word2vec import Word2Vec
- from qhchina.analytics.vectors import project_2d, cosine_similarity
- from qhchina.analytics.collocations import find_collocates, cooc_matrix
- from qhchina.analytics.topicmodels import LDAGibbsSampler
"""

# Define what should be available when using wildcard imports (import *)
# This is empty to prevent unwanted imports when using `from qhchina.analytics import *`
__all__ = []