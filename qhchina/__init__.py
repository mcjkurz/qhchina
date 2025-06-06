"""qhchina: A package for Chinese text analytics and educational tools

Core analytics functionality is available directly.
For more specialized functions, import from specific modules:
- qhchina.analytics: Text analytics and modeling
- qhchina.preprocessing: Text preprocessing utilities
- qhchina.helpers: Utility functions
- qhchina.educational: Educational visualization tools
"""

__version__ = "0.0.49"

# Import helper functions directly into the package namespace
from .helpers.fonts import load_fonts, current_font, set_font
from .helpers.texts import load_texts, load_stopwords