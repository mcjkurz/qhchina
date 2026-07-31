"""
GloVe models under the embeddings namespace.
"""

from .base import GloVe, CYTHON_AVAILABLE, glove_c

__all__ = ["GloVe", "CYTHON_AVAILABLE", "glove_c"]
