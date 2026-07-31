"""
Embeddings sub-package for learning word vectors from text.

Provides model variants:
- Word2Vec: Standard CBOW/Skip-gram word embeddings.
- TempRefWord2Vec: Temporal Referencing for tracking semantic change.
- DynamicWord2Vec: Time-sliced embeddings with temporal regularization.
- GloVe: Global vectors trained from weighted co-occurrence statistics.

Also exports utility classes for temporal sentence iteration.

Example:
    from qhchina.analytics.embeddings import Word2Vec, TempRefWord2Vec, DynamicWord2Vec, GloVe
"""

from .word2vec import (
    Word2Vec,
    TempRefWord2Vec,
    DynamicWord2Vec,
    BalancedSentenceIterator,
    SingleCorpusTemporalIterator,
    TemporalSentence,
    TemporalSentenceIterator,
    CYTHON_AVAILABLE,
    word2vec_c,
)
from .glove import GloVe

__all__ = [
    'Word2Vec',
    'TempRefWord2Vec',
    'DynamicWord2Vec',
    'GloVe',
    'BalancedSentenceIterator',
    'SingleCorpusTemporalIterator',
    'TemporalSentence',
    'TemporalSentenceIterator',
    'CYTHON_AVAILABLE',
    'word2vec_c',
]
