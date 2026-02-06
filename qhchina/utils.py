"""
Utility functions for the qhchina package.

This module provides common utilities used across multiple modules.
"""

import logging
from typing import Dict, List, Set, Any, Optional

import numpy as np

logger = logging.getLogger("qhchina.utils")


__all__ = [
    'validate_filters',
    'apply_p_value_correction',
]


VALID_CORRECTIONS = ('bonferroni', 'fdr_bh')


def apply_p_value_correction(
    p_values: List[float],
    method: str,
) -> np.ndarray:
    """
    Apply multiple testing correction to a list of p-values.
    
    Args:
        p_values: List or array of raw p-values.
        method: Correction method. Options:
            - 'bonferroni': Bonferroni correction (controls family-wise error rate).
              Adjusted p = p * n_tests, capped at 1.0.
            - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate).
              Generally less conservative than Bonferroni.
    
    Returns:
        numpy array of adjusted p-values (same length as input).
    
    Raises:
        ValueError: If method is not recognized or p_values is empty.
    """
    if method not in VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction method '{method}'. "
            f"Valid methods are: {VALID_CORRECTIONS}"
        )
    
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    
    if n == 0:
        return p_arr
    
    if method == 'bonferroni':
        adjusted = np.minimum(p_arr * n, 1.0)
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(p_arr)
        sorted_p = p_arr[sorted_indices]
        
        # Compute adjusted p-values: p_adjusted[i] = p[i] * n / rank
        ranks = np.arange(1, n + 1, dtype=np.float64)
        adjusted_sorted = sorted_p * n / ranks
        
        # Enforce monotonicity (working backwards from largest rank)
        # Each adjusted p cannot be larger than the one at the next higher rank
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        
        # Cap at 1.0
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        
        # Restore original order
        adjusted = np.empty(n, dtype=np.float64)
        adjusted[sorted_indices] = adjusted_sorted
    
    return adjusted


def validate_filters(
    filters: Optional[Dict[str, Any]],
    valid_keys: Set[str],
    context: str = "function"
) -> None:
    """
    Validate that all filter keys are recognized.
    
    Args:
        filters: Dictionary of filter parameters to validate.
        valid_keys: Set of valid/recognized filter keys.
        context: String describing the calling context for error messages.
    
    Raises:
        ValueError: If filters contains unrecognized keys.
    
    Example:
        validate_filters(
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
