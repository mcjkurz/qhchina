"""Helper utilities for package functionality.

This module provides:
- Text loading functions (with automatic encoding detection)
- Font management tools
"""

from .fonts import load_fonts, current_font, set_font, list_available_fonts, list_font_aliases, get_font_path
from .texts import load_text, load_texts, load_stopwords, split_into_chunks, get_stopword_languages, detect_encoding

# Make all functions available at module level
__all__ = ['load_fonts', 'current_font', 'set_font', 'list_available_fonts', 'list_font_aliases', 'get_font_path',
           'load_text', 'load_texts', 'load_stopwords', 'split_into_chunks', 'get_stopword_languages',
           'detect_encoding']