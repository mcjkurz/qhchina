"""Helper utilities for package functionality.

This module provides:
- Package installation utilities
- Text loading functions
- Font management tools
"""

from .fonts import load_fonts, current_font, set_font
from .texts import load_texts, sample_sentences_to_token_count, add_corpus_tags, load_stopwords, split_into_chunks
from .installers import install_package

# Make all functions available at module level
__all__ = ['load_fonts', 'current_font', 'set_font', 
           'load_texts', 'sample_sentences_to_token_count', 
           'add_corpus_tags', 'load_stopwords', 'split_into_chunks',
           'install_package']