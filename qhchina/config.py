"""
Global configuration for the qhchina package.

This module provides centralized configuration for:
- Random seed management with isolated RNG instances (avoids cross-module interference)
- Default settings for various analytics modules

Usage:
    >>> import qhchina
    >>> qhchina.set_random_seed(42)  # Set global seed
    >>> qhchina.get_random_seed()    # Get current seed
    42

For module-specific RNG that doesn't affect other modules:
    >>> from qhchina.config import get_rng
    >>> rng = get_rng()  # Returns numpy RandomState seeded with global seed
    >>> rng.random()     # Use isolated RNG
"""

import numpy as np
import random
from typing import Optional, Union
import threading


# Thread-safe global configuration
_config_lock = threading.Lock()

# Global settings storage
_global_settings = {
    'random_seed': None,  # None means unseeded (random behavior)
}


def set_random_seed(seed: Optional[int]) -> None:
    """
    Set the global random seed for reproducibility across all qhchina modules.
    
    This sets a global seed that affects all modules using `get_rng()` to obtain
    their random number generators. Unlike calling `np.random.seed()` directly,
    this approach isolates RNG state per-module to avoid cross-module interference.
    
    Args:
        seed: Random seed (integer). Use None to reset to unseeded (random) behavior.
    
    Example:
        >>> import qhchina
        >>> qhchina.set_random_seed(42)
        >>> # All subsequent qhchina operations will be reproducible
        
        >>> qhchina.set_random_seed(None)
        >>> # Reset to random behavior
    
    Note:
        For backwards compatibility with existing code, individual modules 
        may still accept `random_state` or `seed` parameters. When specified,
        these override the global seed for that specific operation.
    """
    with _config_lock:
        _global_settings['random_seed'] = seed


def get_random_seed() -> Optional[int]:
    """
    Get the current global random seed.
    
    Returns:
        The current global seed, or None if unseeded.
    
    Example:
        >>> import qhchina
        >>> qhchina.set_random_seed(42)
        >>> qhchina.get_random_seed()
        42
    """
    with _config_lock:
        return _global_settings['random_seed']


def get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Get an isolated numpy RandomState instance.
    
    This creates a new RandomState that doesn't affect the global numpy.random state,
    preventing one module's random operations from affecting another's.
    
    Args:
        seed: Optional seed for this specific RNG. If None, uses the global seed.
              If the global seed is also None, returns an unseeded RNG.
    
    Returns:
        A numpy RandomState instance.
    
    Example:
        >>> from qhchina.config import get_rng
        >>> rng = get_rng(42)  # Specific seed
        >>> rng.random()
        
        >>> rng = get_rng()  # Uses global seed (if set) or random
        >>> rng.randint(0, 100)
    """
    effective_seed = seed if seed is not None else get_random_seed()
    if effective_seed is not None:
        return np.random.RandomState(effective_seed)
    else:
        return np.random.RandomState()


def get_python_rng(seed: Optional[int] = None) -> random.Random:
    """
    Get an isolated Python random.Random instance.
    
    This creates a new Random instance that doesn't affect the global random state,
    preventing one module's random operations from affecting another's.
    
    Args:
        seed: Optional seed for this specific RNG. If None, uses the global seed.
              If the global seed is also None, returns an unseeded RNG.
    
    Returns:
        A Python random.Random instance.
    
    Example:
        >>> from qhchina.config import get_python_rng
        >>> rng = get_python_rng(42)
        >>> rng.random()
    """
    effective_seed = seed if seed is not None else get_random_seed()
    rng = random.Random()
    if effective_seed is not None:
        rng.seed(effective_seed)
    return rng


def resolve_seed(local_seed: Optional[int], default_seed: Optional[int] = None) -> Optional[int]:
    """
    Resolve which seed to use, with priority: local > default > global.
    
    This is a helper for modules that accept both a `random_state`/`seed` parameter
    and want to fall back to the global seed if none is provided.
    
    Args:
        local_seed: The seed passed directly to a function/class (highest priority)
        default_seed: A default seed value (medium priority)
    
    Returns:
        The resolved seed value, or None if all are None.
    
    Example:
        >>> from qhchina.config import resolve_seed, set_random_seed
        >>> set_random_seed(42)  # Global seed
        >>> resolve_seed(None)   # Returns 42 (global)
        42
        >>> resolve_seed(123)    # Returns 123 (local overrides global)
        123
    """
    if local_seed is not None:
        return local_seed
    if default_seed is not None:
        return default_seed
    return get_random_seed()
