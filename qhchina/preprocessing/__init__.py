"""Preprocessing module for text manipulation.

This module provides tools for:
- Text chunking and segmentation
"""

from .segmentation import create_segmenter, SegmentationWrapper, SpacySegmenter

# Make all functions available at module level
__all__ = [
    # Segmentation
    'create_segmenter',
    'SegmentationWrapper',
    'SpacySegmenter',
]
