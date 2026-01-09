"""qhchina: A package for Chinese text analytics and educational tools

Core analytics functionality is available directly.
For more specialized functions, import from specific modules:
- qhchina.analytics: Text analytics and modeling
- qhchina.preprocessing: Text preprocessing utilities
- qhchina.helpers: Utility functions
- qhchina.educational: Educational visualization tools
"""

import logging

from importlib.metadata import version as _get_version
__version__ = _get_version("qhchina")

# Configure package-level logger
# Users can customize this by getting the logger and adding handlers/formatters
logger = logging.getLogger("qhchina")

# Prevent propagation to root logger to avoid duplicate output in notebook 
# environments (Colab, Jupyter) where the root logger has its own handlers
logger.propagate = False

# Set up a default handler if none exists
if not logger.handlers:
    # Add a StreamHandler that outputs to console by default
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def set_log_level(level: str) -> None:
    """
    Set the logging level for the qhchina package.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               or 'SILENT' to suppress all messages.
    
    Example:
        >>> import qhchina
        >>> qhchina.set_log_level('WARNING')  # Only show warnings and errors
        >>> qhchina.set_log_level('DEBUG')    # Show all messages including debug
        >>> qhchina.set_log_level('SILENT')   # Suppress all messages
    """
    if level.upper() == 'SILENT':
        logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL to suppress everything
    else:
        logger.setLevel(getattr(logging, level.upper()))


# Import global configuration functions
from .config import set_random_seed, get_random_seed, get_rng, get_python_rng

# Import helper functions directly into the package namespace
from .helpers.fonts import load_fonts, current_font, set_font, list_available_fonts, list_font_aliases, get_font_path
from .helpers.texts import load_text, load_texts, load_stopwords