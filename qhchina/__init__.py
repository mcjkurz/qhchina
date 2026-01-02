"""qhchina: A package for Chinese text analytics and educational tools

Core analytics functionality is available directly.
For more specialized functions, import from specific modules:
- qhchina.analytics: Text analytics and modeling
- qhchina.preprocessing: Text preprocessing utilities
- qhchina.helpers: Utility functions
- qhchina.educational: Educational visualization tools
"""

from importlib.metadata import version as _get_version
__version__ = _get_version("qhchina")

# Import helper functions directly into the package namespace
from .helpers.fonts import load_fonts, current_font, set_font, list_available_fonts, list_font_aliases, get_font_path
from .helpers.texts import load_text, load_texts, load_stopwords