"""Preprocessing module for text manipulation.

This module provides Chinese text segmentation with various backends and strategies.

Import from the segmentation submodule:
    from qhchina.preprocessing.segmentation import create_segmenter, SegmentationWrapper
"""

from .segmentation import (
    create_segmenter,
    SegmentationWrapper,
    SpacySegmenter,
    JiebaSegmenter,
    BertSegmenter,
    LLMSegmenter,
)

__all__ = [
    'create_segmenter',
    'SegmentationWrapper',
    'SpacySegmenter',
    'JiebaSegmenter',
    'BertSegmenter',
    'LLMSegmenter',
]
