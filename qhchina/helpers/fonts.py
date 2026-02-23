"""
Font management for Chinese text rendering in matplotlib.

Fonts are downloaded from the qhchina-data GitHub repository on first use
and cached locally in ~/.cache/qhchina/fonts/.
"""

import logging
import os
import threading
from pathlib import Path

try:
    import matplotlib
    import matplotlib.font_manager as fm
except ImportError as e:
    raise ImportError(f"matplotlib is required for font management: {e}") from e

from qhchina.helpers.github import (
    ensure_cache_dir as _ensure_github_cache_dir,
    download_file as _download_file,
    query_github_api as _query_github_api_raw,
    CACHE_BASE as _CACHE_BASE,
)

logger = logging.getLogger("qhchina.helpers.fonts")


__all__ = [
    'load_fonts',
    'set_font',
    'get_font_path',
    'current_font',
    'download_fonts',
    'list_remote_fonts',
    'list_cached_fonts',
    'clear_cache',
    'get_cache_dir',
]


# Configuration
_DEFAULT_FONT_FILE = 'NotoSansTCSC-Regular.otf'
_CACHE_DIR = _CACHE_BASE / 'fonts'

# Thread safety
_lock = threading.Lock()
_loaded_fonts: set[str] = set()  # Track which fonts have been added to font manager


def get_cache_dir() -> Path:
    """
    Get the font cache directory path.
    
    Returns:
        Path to ~/.cache/qhchina/fonts/
    """
    return _CACHE_DIR


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    return _ensure_github_cache_dir('fonts')


def _query_github_api() -> list[dict]:
    """Query GitHub API for available fonts."""
    files = _query_github_api_raw('fonts')
    fonts = []
    for item in files:
        if item['type'] == 'file' and item['name'].endswith(('.otf', '.ttf', '.OTF', '.TTF')):
            fonts.append({
                'file': item['name'],
                'download_url': item['download_url'],
                'size': item['size'],
            })
    return fonts


def _get_font_name(font_path: Path) -> str:
    """Extract font name from a font file."""
    props = fm.FontProperties(fname=str(font_path))
    return props.get_name()


def _add_to_font_manager(font_path: Path) -> str:
    """
    Add font to matplotlib's font manager and return the font name.
    Thread-safe, skips if already loaded.
    """
    path_str = str(font_path)
    
    with _lock:
        if path_str not in _loaded_fonts:
            fm.fontManager.addfont(path_str)
            _loaded_fonts.add(path_str)
    
    return _get_font_name(font_path)


def _set_rcparams(font_name: str) -> None:
    """Set matplotlib rcParams for the given font."""
    is_serif = 'serif' in font_name.lower()
    
    if is_serif:
        matplotlib.rcParams['font.serif'] = [font_name, 'serif']
        matplotlib.rcParams['font.family'] = 'serif'
    else:
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'sans-serif']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    
    matplotlib.rcParams['axes.unicode_minus'] = False


def _ensure_cached(font_file: str, show_progress: bool = True) -> Path:
    """
    Ensure a font file is in the cache, downloading if necessary.
    
    Args:
        font_file: Font file name (e.g., 'NotoSerifTC-Regular.otf')
        show_progress: Show download progress
    
    Returns:
        Path to the cached font file
    
    Raises:
        ValueError: If font file not found in remote repository
        requests.RequestException: If download fails
    """
    _ensure_cache_dir()
    cached_path = _CACHE_DIR / font_file
    
    if cached_path.exists():
        return cached_path
    
    # Query GitHub for download URL
    remote_fonts = _query_github_api()
    font_info = next((f for f in remote_fonts if f['file'] == font_file), None)
    
    if font_info is None:
        available = [f['file'] for f in remote_fonts]
        raise ValueError(
            f"Font file '{font_file}' not found in qhchina-data repository.\n"
            f"Available fonts: {available}"
        )
    
    if show_progress:
        print(f"Downloading font '{font_file}'...")
    
    _download_file(font_info['download_url'], cached_path)
    
    if show_progress:
        size_mb = cached_path.stat().st_size / 1024 / 1024
        print(f"Font downloaded ({size_mb:.1f} MB)")
    
    return cached_path


# =============================================================================
# Public API
# =============================================================================

def load_fonts() -> str:
    """
    Load the default CJK font for matplotlib.
    
    Downloads the font from GitHub if not already cached.
    This is the simplest way to get started with Chinese text in plots.
    
    Returns:
        The font name that was set (e.g., 'Noto Sans CJK TC')
    
    Example:
        >>> import qhchina
        >>> qhchina.load_fonts()
        'Noto Sans CJK TC'
        >>> plt.title('中文標題')  # Now works!
    """
    cached_path = _ensure_cached(_DEFAULT_FONT_FILE)
    font_name = _add_to_font_manager(cached_path)
    _set_rcparams(font_name)
    return font_name


def set_font(font: str) -> str:
    """
    Set matplotlib to use a specific font.
    
    Args:
        font: One of:
              - Font file name: 'NotoSerifTC-Regular.otf' (downloads from GitHub if needed)
              - Local file path: '/path/to/font.otf' (must exist)
              - Font name: 'Noto Serif TC', 'SimHei', etc. (sets rcParams directly)
    
    Returns:
        The font name that was set
    
    Examples:
        >>> # Use a font from qhchina-data (downloads if needed)
        >>> qhchina.set_font('NotoSerifTC-Regular.otf')
        'Noto Serif TC'
        
        >>> # Use a local font file
        >>> qhchina.set_font('/path/to/MyFont.otf')
        'My Font'
        
        >>> # Use an already-loaded or system font by name
        >>> qhchina.set_font('Noto Serif TC')
        'Noto Serif TC'
    """
    font_str = str(font)
    
    if font_str.endswith(('.otf', '.ttf', '.OTF', '.TTF')):
        # It's a font file
        path = Path(font_str)
        
        if path.exists():
            # Local file path
            font_name = _add_to_font_manager(path)
        elif '/' in font_str or '\\' in font_str:
            # Looks like a path but doesn't exist
            raise FileNotFoundError(f"Font file not found: {font_str}")
        else:
            # Just a filename - download from GitHub
            cached_path = _ensure_cached(font_str)
            font_name = _add_to_font_manager(cached_path)
    else:
        # It's a font name - just use it directly
        font_name = font_str
    
    _set_rcparams(font_name)
    return font_name


def get_font_path(font: str | None = None) -> str:
    """
    Get the file path for a font (for use with WordCloud, etc.).
    
    Args:
        font: Font file name (e.g., 'NotoSerifTC-Regular.otf') or None for default.
              Can also be a local file path.
    
    Returns:
        Absolute path to the font file
    
    Example:
        >>> path = qhchina.get_font_path()  # default font
        >>> wc = WordCloud(font_path=path, ...)
        
        >>> path = qhchina.get_font_path('NotoSerifTC-Regular.otf')
    """
    if font is None:
        font = _DEFAULT_FONT_FILE
    
    font_str = str(font)
    
    # Check if it's an existing path
    if Path(font_str).exists():
        return str(Path(font_str).resolve())
    
    # Check if it ends with font extension
    if font_str.endswith(('.otf', '.ttf', '.OTF', '.TTF')):
        cached_path = _ensure_cached(font_str)
        return str(cached_path)
    
    raise ValueError(
        f"Cannot get path for '{font}'. "
        f"Provide a font file name (e.g., 'NotoSerifTC-Regular.otf') or a local file path."
    )


def download_fonts(fonts: str | list[str] | None = None) -> dict[str, str]:
    """
    Download font files from the qhchina-data repository.
    
    Args:
        fonts: Font file name(s) to download. If None, downloads ALL available fonts.
               Examples:
                 - None: download all fonts
                 - 'NotoSerifTC-Regular.otf': download single font
                 - ['NotoSerifTC-Regular.otf', 'NotoSerifSC-Regular.otf']: download multiple
    
    Returns:
        Dict mapping file names to font names:
        {'NotoSerifTC-Regular.otf': 'Noto Serif TC', ...}
    
    Example:
        >>> # Download all fonts
        >>> qhchina.download_fonts()
        {'NotoSansTCSC-Regular.otf': 'Noto Sans CJK TC', ...}
        
        >>> # Download specific font
        >>> qhchina.download_fonts('NotoSerifTC-Regular.otf')
        {'NotoSerifTC-Regular.otf': 'Noto Serif TC'}
    """
    if fonts is None:
        # Download all available fonts
        remote = _query_github_api()
        font_files = [f['file'] for f in remote]
    elif isinstance(fonts, str):
        font_files = [fonts]
    else:
        font_files = list(fonts)
    
    result = {}
    for font_file in font_files:
        cached_path = _ensure_cached(font_file)
        font_name = _add_to_font_manager(cached_path)
        result[font_file] = font_name
    
    return result


def list_remote_fonts() -> list[dict]:
    """
    Query GitHub for available fonts in the qhchina-data repository.
    
    Returns:
        List of dicts with font information:
        [{'file': 'NotoSansTCSC-Regular.otf', 'size': 17279824, 'size_mb': 16.5}, ...]
    
    Example:
        >>> qhchina.list_remote_fonts()
        [{'file': 'NotoSansTCSC-Regular.otf', 'size': 17279824, 'size_mb': 16.5}, ...]
    """
    remote = _query_github_api()
    return [
        {
            'file': f['file'],
            'size': f['size'],
            'size_mb': round(f['size'] / 1024 / 1024, 1),
        }
        for f in remote
    ]


def list_cached_fonts() -> list[dict]:
    """
    List fonts currently in the local cache.
    
    Returns:
        List of dicts with font information:
        [{'file': 'NotoSansTCSC-Regular.otf', 'font_name': 'Noto Sans CJK TC', 
          'path': '/Users/.../.cache/qhchina/fonts/NotoSansTCSC-Regular.otf', 
          'size_mb': 16.5}, ...]
    """
    if not _CACHE_DIR.exists():
        return []
    
    result = []
    for ext in ('*.otf', '*.ttf', '*.OTF', '*.TTF'):
        for font_file in _CACHE_DIR.glob(ext):
            if font_file.name.startswith('.'):
                continue
            try:
                font_name = _get_font_name(font_file)
                result.append({
                    'file': font_file.name,
                    'font_name': font_name,
                    'path': str(font_file),
                    'size_mb': round(font_file.stat().st_size / 1024 / 1024, 1),
                })
            except Exception as e:
                logger.warning(f"Could not read font {font_file.name}: {e}")
    
    return result


def current_font() -> str | None:
    """
    Get the currently configured matplotlib font name.
    
    Returns:
        The current font name, or None if using matplotlib defaults.
    """
    try:
        font_family = matplotlib.rcParams.get('font.family', [])
        
        if font_family == 'serif' or font_family == ['serif']:
            fonts = matplotlib.rcParams.get('font.serif', [])
        else:
            fonts = matplotlib.rcParams.get('font.sans-serif', [])
        
        return fonts[0] if fonts else None
    except (KeyError, IndexError):
        return None


def clear_cache() -> None:
    """
    Remove all cached fonts.
    
    Example:
        >>> qhchina.clear_cache()
        >>> qhchina.list_cached_fonts()
        []
    """
    global _loaded_fonts
    
    if _CACHE_DIR.exists():
        for font_file in _CACHE_DIR.glob('*'):
            if font_file.is_file() and not font_file.name.startswith('.'):
                font_file.unlink()
    
    with _lock:
        _loaded_fonts.clear()
