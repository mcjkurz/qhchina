"""qhchina: A package for Chinese text analytics and educational tools

Core analytics functionality is available directly.
For more specialized functions, import from specific modules:
- qhchina.analytics: Text analytics and modeling
- qhchina.preprocessing: Text preprocessing utilities
- qhchina.helpers: Utility functions
- qhchina.educational: Educational visualization tools
"""

__version__ = "0.0.38"

# Helper functions
from .helpers import (
    load_fonts,
    set_font,
    current_font,
)
