"""Preprocessing module for text manipulation.

This module provides:
- Chinese text segmentation with various backends and strategies
- Text normalization (script conversion, punctuation, whitespace, quotes)

Import from submodules:
    from qhchina.preprocessing.segmentation import create_segmenter
    from qhchina.preprocessing.normalization import normalize
"""

from .segmentation import (
    create_segmenter,
    SegmentationWrapper,
    SpacySegmenter,
    JiebaSegmenter,
    BertSegmenter,
    LLMSegmenter,
)

from .normalization import (
    normalize,
    NormalizeOptions,
)

__all__ = [
    # Segmentation
    'create_segmenter',
    'SegmentationWrapper',
    'SpacySegmenter',
    'JiebaSegmenter',
    'BertSegmenter',
    'LLMSegmenter',
    # Normalization
    'normalize',
    'NormalizeOptions',
]
