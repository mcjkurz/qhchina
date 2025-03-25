"""Preprocessing module for text manipulation.

This module provides tools for:
- Text chunking and segmentation
"""

from .chunking import split_into_chunks

# Make all functions available at module level
__all__ = [
    # Chunking
    'split_into_chunks',
]
