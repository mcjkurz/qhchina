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

from .word2vec_base import Word2Vec
from .word2vec_tempref import TempRefWord2Vec
from .word2vec_dynamic import DynamicWord2Vec
from .word2vec_utils import (
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
