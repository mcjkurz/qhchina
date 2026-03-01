"""Analytics module for text and vector operations.

This module provides tools for:
- Word embeddings (Word2Vec, TempRefWord2Vec, DynamicWord2Vec)
- Topic modeling (LDAGibbsSampler)
- Stylometry and authorship attribution (Stylometry, compare_corpora)
- Collocation analytics (find_collocates, cooc_matrix, plot_collocates)
- Vector operations and projections (project_2d, cosine_similarity)

Convenience imports:
    from qhchina.analytics import Word2Vec, LDAGibbsSampler, Stylometry
    from qhchina.analytics import find_collocates, cooc_matrix
"""

# Corpus streaming
from ..utils import LineSentenceFile

# Word embeddings
from .word2vec import Word2Vec, TempRefWord2Vec, DynamicWord2Vec

# Topic modeling
from .topicmodels import LDAGibbsSampler

# Stylometry
from .stylometry import Stylometry, compare_corpora, type_token_ratio, mattr

# Collocations
from .collocations import find_collocates, cooc_matrix, plot_collocates, kwic, compare_collocates

# Text reuse
from .textreuse import find_shared_sequences

__all__ = [
    # Corpus streaming
    'LineSentenceFile',
    # Word embeddings
    'Word2Vec',
    'TempRefWord2Vec',
    'DynamicWord2Vec',
    # Topic modeling
    'LDAGibbsSampler',
    # Stylometry
    'Stylometry',
    'compare_corpora',
    'type_token_ratio',
    'mattr',
    # Collocations
    'find_collocates',
    'cooc_matrix',
    'plot_collocates',
    'kwic',
    'compare_collocates',
    # Text reuse
    'find_shared_sequences',
]