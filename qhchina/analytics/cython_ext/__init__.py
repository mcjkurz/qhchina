"""
Cython extensions for accelerating LDA topic modeling.

This package contains optimized Cython implementations of 
computationally intensive functions used in the LDA Gibbs sampling algorithm.

If the compiled extensions are not available, the LDA implementation 
will automatically fall back to a pure Python implementation.
"""

# This file makes the cython_ext directory a proper Python package
__all__ = []

# Try to import the compiled extensions 
try:
    from . import lda_sampler
    __all__.append('lda_sampler')
except ImportError:
    # Extensions not compiled, will fall back to Python implementation
    pass 