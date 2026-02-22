"""Analytics module for text and vector operations.

This module provides tools for:
- Word embeddings (Word2Vec, TempRefWord2Vec)
- Topic modeling (LDAGibbsSampler)
- Stylometry and authorship attribution (Stylometry, compare_corpora)
- Collocation analytics (find_collocates, cooc_matrix, plot_collocates)
- Vector operations and projections (project_2d, cosine_similarity)

Convenience imports:
    from qhchina.analytics import Word2Vec, LDAGibbsSampler, Stylometry
    from qhchina.analytics import find_collocates, cooc_matrix
"""

# Word embeddings
from .word2vec import Word2Vec
from .tempref_word2vec import TempRefWord2Vec

# Topic modeling
from .topicmodels import LDAGibbsSampler

# Stylometry
from .stylometry import Stylometry, compare_corpora

# Collocations
from .collocations import find_collocates, cooc_matrix, plot_collocates

__all__ = [
    # Word embeddings
    'Word2Vec',
    'TempRefWord2Vec',
    # Topic modeling
    'LDAGibbsSampler',
    # Stylometry
    'Stylometry',
    'compare_corpora',
    # Collocations
    'find_collocates',
    'cooc_matrix',
    'plot_collocates',
]