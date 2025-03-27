"""Educational tools for visualization and learning.

This module provides:
- Vector visualization tools
- Educational plotting utilities
"""

# qhchina/educational/__init__.py
from .visuals import show_vectors
from .llms import predict_next_token

# Make all functions available at module level
__all__ = [
    # Visuals
    'show_vectors',
    # LLMs
    'predict_next_token',
]