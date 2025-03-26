"""Helper utilities for package functionality.

This module provides:
- Package installation utilities
- Text loading functions
- Font management tools
"""

# Package installation
from .installers import install_package

# Text loading
from .texts import load_texts, sample_sentences_to_token_count, add_corpus_tags, load_stopwords

# Font management
from .fonts import load_fonts, set_font, current_font

# Make all functions available at module level
__all__ = [
    # Installers
    'install_package',
    # Texts
    'load_texts', 'sample_sentences_to_token_count', 'add_corpus_tags', 'load_stopwords',
    # Fonts
    'load_fonts', 'set_font', 'current_font',
]