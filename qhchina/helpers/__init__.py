"""Helper utilities for package functionality.

This module provides:
- Text loading functions (with automatic encoding detection)
- Font management tools (with automatic download from GitHub)
"""

from .fonts import (
    load_font,
    load_fonts,
    get_current_font_name,
    get_current_font_path,
    download_fonts,
    list_remote_fonts,
    list_cached_fonts,
    clear_cache,
    get_cache_dir,
)
from .texts import load_text, load_texts, load_stopwords, split_into_chunks, get_stopword_languages, detect_encoding, download_corpus, download_file, list_remote_corpora
from ..utils import LineSentenceFile

__all__ = [
    # Font management
    'load_font',
    'load_fonts',
    'get_current_font_name',
    'get_current_font_path',
    'download_fonts',
    'list_remote_fonts',
    'list_cached_fonts',
    'clear_cache',
    'get_cache_dir',
    # Text loading
    'load_text',
    'load_texts',
    'load_stopwords',
    'split_into_chunks',
    'get_stopword_languages',
    'detect_encoding',
    'download_corpus',
    'download_file',
    'list_remote_corpora',
    # Corpus streaming
    'LineSentenceFile',
]
