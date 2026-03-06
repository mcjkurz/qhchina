"""
Word2Vec sub-package for learning word embeddings from text.

Provides three model variants:
- Word2Vec: Standard CBOW/Skip-gram word embeddings.
- TempRefWord2Vec: Temporal Referencing for tracking semantic change.
- DynamicWord2Vec: Time-sliced embeddings with temporal regularization.

Also exports utility classes for temporal sentence iteration.

Example:
    from qhchina.analytics.word2vec import Word2Vec, TempRefWord2Vec, DynamicWord2Vec
"""

from .base import Word2Vec
from .tempref import TempRefWord2Vec
from .dynamic import DynamicWord2Vec
from .utils import (
    BalancedSentenceIterator,
    SingleCorpusTemporalIterator,
    TemporalSentence,
    TemporalSentenceIterator,
    CYTHON_AVAILABLE,
    word2vec_c,
)

__all__ = [
    'Word2Vec',
    'TempRefWord2Vec',
    'DynamicWord2Vec',
    'BalancedSentenceIterator',
    'SingleCorpusTemporalIterator',
    'TemporalSentence',
    'TemporalSentenceIterator',
    'CYTHON_AVAILABLE',
    'word2vec_c',
]
