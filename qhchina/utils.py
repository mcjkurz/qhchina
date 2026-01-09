"""
Utility functions for the qhchina package.

This module provides common utilities used across multiple modules.
"""

import logging
from typing import Dict, Set, Any, Optional

logger = logging.getLogger("qhchina.utils")


def validate_filters(
    filters: Optional[Dict[str, Any]],
    valid_keys: Set[str],
    context: str = "function"
) -> None:
    """
    Validate that all filter keys are recognized.
    
    Parameters:
    -----------
    filters: Dictionary of filter parameters to validate
    valid_keys: Set of valid/recognized filter keys
    context: String describing the calling context for error messages
    
    Raises:
    -------
    ValueError: If filters contains unrecognized keys
    
    Example:
    --------
    >>> validate_filters(
    ...     {'min_count': 5, 'max_p': 0.05, 'invalid_key': 'value'},
    ...     {'min_count', 'max_p', 'stopwords'},
    ...     context='compare_corpora'
    ... )
    ValueError: Unknown filter keys in compare_corpora: {'invalid_key'}. 
                Valid keys are: {'max_p', 'min_count', 'stopwords'}
    """
    if filters is None:
        return
    
    if not isinstance(filters, dict):
        raise TypeError(f"filters must be a dictionary, got {type(filters).__name__}")
    
    unknown_keys = set(filters.keys()) - valid_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown filter keys in {context}: {unknown_keys}. "
            f"Valid keys are: {valid_keys}"
        )
