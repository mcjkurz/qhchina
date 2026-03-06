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
    'load_font',
    'load_fonts',
    'get_current_font_name',
    'get_current_font_path',
    'download_fonts',
    'list_remote_fonts',
    'list_cached_fonts',
    'clear_cache',
    'get_cache_dir',
]


# Configuration
_DEFAULT_FONT_FILE = 'NotoSansTCSC-Regular.otf'
_CACHE_DIR = _CACHE_BASE / 'fonts'
_VALID_FONT_EXTENSIONS = ('.otf', '.ttf', '.OTF', '.TTF')

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
    is_sans = 'sans' in font_name.lower()
    
    if is_sans:
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'sans-serif']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    else:
        matplotlib.rcParams['font.serif'] = [font_name, 'serif']
        matplotlib.rcParams['font.family'] = 'serif'
    
    matplotlib.rcParams['axes.unicode_minus'] = False


def _validate_font_extension(font_str: str, param_name: str) -> None:
    """Validate that a font string has a valid font extension."""
    if not font_str.endswith(_VALID_FONT_EXTENSIONS):
        raise ValueError(
            f"Invalid font file for {param_name}='{font_str}'. "
            f"Must end with one of: {', '.join(_VALID_FONT_EXTENSIONS)}"
        )


def _ensure_cached(
    font_file: str,
    show_progress: bool = True,
    force_download: bool = False,
) -> Path:
    """
    Ensure a font file is in the cache, downloading if necessary.
    
    Args:
        font_file: Font file name (e.g., 'NotoSerifTC-Regular.otf')
        show_progress: Show download progress
        force_download: Re-download even if cached
    
    Returns:
        Path to the cached font file
    
    Raises:
        ValueError: If font file not found in remote repository
        requests.RequestException: If download fails
    """
    _ensure_cache_dir()
    cached_path = _CACHE_DIR / font_file
    
    if cached_path.exists() and not force_download:
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
        action = "Re-downloading" if force_download else "Downloading"
        print(f"{action} font '{font_file}'...")
    
    _download_file(font_info['download_url'], cached_path)
    
    if show_progress:
        size_mb = cached_path.stat().st_size / 1024 / 1024
        print(f"Font downloaded ({size_mb:.1f} MB)")
    
    return cached_path


# =============================================================================
# Public API
# =============================================================================

def load_font(
    *,
    remote: str | None = None,
    path: str | Path | None = None,
    force_download: bool = False,
) -> str:
    """
    Load a font and set it as the active font for matplotlib.
    
    This function:
    1. Downloads the font file if using `remote=` and not already cached
    2. Registers the font with matplotlib's font manager
    3. Sets matplotlib's rcParams to use this font for all subsequent plots
       (font.family, font.sans-serif or font.serif, axes.unicode_minus)
    
    After calling this function, all matplotlib plots will use the loaded font.
    
    Args:
        remote: Font filename from qhchina-data repository (e.g., 'NotoSerifTC-Regular.otf').
                Uses cache if available, downloads otherwise.
        path: Path to a local font file.
        force_download: If True, re-download from repository even if cached.
                        Only applies when using `remote=`. Ignored for `path=`.
    
    Returns:
        The font name that was set (e.g., 'Noto Serif TC').
    
    If called with no arguments, loads the default font.
    
    Examples:
        >>> from qhchina.helpers import load_font
        >>> import matplotlib.pyplot as plt
        
        >>> # Load default font - plots will now render Chinese correctly
        >>> load_font()
        'Noto Sans CJK TC'
        >>> plt.title('中文標題')  # This now works!
        
        >>> # Load specific font from repository
        >>> load_font(remote='NotoSerifTC-Regular.otf')
        'Noto Serif TC'
        
        >>> # Force re-download
        >>> load_font(remote='NotoSerifTC-Regular.otf', force_download=True)
        'Noto Serif TC'
        
        >>> # Load local font file
        >>> load_font(path='/path/to/MyFont.otf')
        'My Font'
    
    Raises:
        ValueError: If both `remote` and `path` are provided, or if file extension is invalid.
        FileNotFoundError: If local `path` does not exist.
    """
    if remote is not None and path is not None:
        raise ValueError("Cannot specify both 'remote' and 'path'. Use one or the other.")
    
    if remote is not None:
        _validate_font_extension(remote, 'remote')
        cached_path = _ensure_cached(remote, force_download=force_download)
        font_name = _add_to_font_manager(cached_path)
    elif path is not None:
        path_str = str(path)
        _validate_font_extension(path_str, 'path')
        font_path = Path(path)
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found: {path}")
        font_name = _add_to_font_manager(font_path)
    else:
        # Default font
        cached_path = _ensure_cached(_DEFAULT_FONT_FILE, force_download=force_download)
        font_name = _add_to_font_manager(cached_path)
    
    _set_rcparams(font_name)
    return font_name


def load_fonts() -> str:
    """
    Load the default CJK font for matplotlib.
    
    This is a convenience function for backward compatibility.
    Equivalent to calling `load_font()` with no arguments.
    
    Returns:
        The font name that was set (e.g., 'Noto Sans CJK TC')
    
    Example:
        >>> from qhchina.helpers import load_fonts
        >>> load_fonts()
        'Noto Sans CJK TC'
        >>> plt.title('中文標題')  # Now works!
    """
    return load_font()


def get_current_font_path() -> str:
    """
    Get the file path of the currently loaded font.
    
    This is useful for tools like WordCloud that require a font file path
    rather than a font name.
    
    Returns:
        Absolute path to the currently loaded font file.
    
    Examples:
        >>> from qhchina.helpers import load_font, get_current_font_path
        
        >>> load_font()  # Load default font
        'Noto Sans CJK TC'
        >>> path = get_current_font_path()
        >>> wc = WordCloud(font_path=path, ...)
        
        >>> load_font(remote='NotoSerifTC-Regular.otf')
        'Noto Serif TC'
        >>> get_current_font_path()  # Returns path to NotoSerifTC-Regular.otf
    
    Raises:
        RuntimeError: If no font has been loaded via load_font() yet.
    """
    font_name = get_current_font_name()
    
    if font_name is None:
        raise RuntimeError(
            "No font has been loaded yet. Call load_font() first.\n"
            "Example: load_font() or load_font(remote='NotoSerifTC-Regular.otf')"
        )
    
    # Use matplotlib's font manager to find the font file path
    props = fm.FontProperties(family=font_name)
    font_path = fm.findfont(props, fallback_to_default=False)
    
    return font_path


def download_fonts(fonts: list[str] | None = None) -> dict[str, str]:
    """
    Pre-download font files from the qhchina-data repository.
    
    Use this to download fonts for offline use. The fonts are downloaded
    but not set as the active matplotlib font. To download and activate
    a font, use `load_font()` instead.
    
    Args:
        fonts: List of font file names to download. If None, downloads ALL available fonts.
    
    Returns:
        Dict mapping file names to font names:
        {'NotoSerifTC-Regular.otf': 'Noto Serif TC', ...}
    
    Examples:
        >>> from qhchina.helpers import download_fonts
        
        >>> # Download all fonts
        >>> download_fonts()
        {'NotoSansTCSC-Regular.otf': 'Noto Sans CJK TC', ...}
        
        >>> # Download specific fonts
        >>> download_fonts(['NotoSerifTC-Regular.otf', 'NotoSerifSC-Regular.otf'])
        {'NotoSerifTC-Regular.otf': 'Noto Serif TC', ...}
    
    Raises:
        TypeError: If fonts is not a list or None.
        ValueError: If any font file has an invalid extension.
    """
    if fonts is None:
        # Download all available fonts
        remote = _query_github_api()
        font_files = [f['file'] for f in remote]
    elif isinstance(fonts, list):
        font_files = fonts
    else:
        raise TypeError(
            f"fonts must be a list of font file names or None, got {type(fonts).__name__}. "
            f"For a single font, use: download_fonts(['{fonts}'])"
        )
    
    for font_file in font_files:
        _validate_font_extension(font_file, 'fonts')
    
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


def get_current_font_name() -> str | None:
    """
    Get the currently loaded matplotlib font name.
    
    Returns:
        The current font name, or None if using matplotlib defaults.
    
    Example:
        >>> from qhchina.helpers import load_font, get_current_font_name
        >>> load_font(remote='NotoSerifTC-Regular.otf')
        'Noto Serif TC'
        >>> get_current_font_name()
        'Noto Serif TC'
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
