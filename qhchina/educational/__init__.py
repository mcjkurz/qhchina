"""Educational tools for visualization and learning.

This module provides:
- Vector visualization tools
- Educational plotting utilities
- Language model utilities
"""

from .visuals import show_vectors
from .llms import predict_next_token

__all__ = [
    'show_vectors',
    'predict_next_token',
]