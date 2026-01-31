import logging
import shutil
import threading
from typing import Optional
try:
    import matplotlib
    import matplotlib.font_manager
except Exception as e:
    raise ImportError(f"Failed to import matplotlib: {e}") from e
from pathlib import Path
import os

logger = logging.getLogger("qhchina.helpers.fonts")


__all__ = [
    'set_font',
    'load_fonts',
    'get_font_path',
    'current_font',
    'list_available_fonts',
    'list_font_aliases',
]


PACKAGE_PATH = Path(__file__).parents[1].resolve() # qhchina
CJK_FONT_PATH = Path(f'{PACKAGE_PATH}/data/fonts').resolve()
MPL_FONT_PATH = Path(f'{matplotlib.get_data_path()}/fonts/ttf').resolve()

# Font aliases for convenient access
FONT_ALIASES = {
    'sans': 'Noto Sans CJK TC',
    'sans-tc': 'Noto Sans CJK TC',
    'sans-sc': 'Noto Sans CJK TC',  # Contains both TC and SC characters
    'serif-tc': 'Noto Serif TC',
    'serif-sc': 'Noto Serif SC',
}

# Mapping from font names to font file names
FONT_FILES = {
    'Noto Sans CJK TC': 'NotoSansTCSC-Regular.otf',
    'Noto Serif TC': 'NotoSerifTC-Regular.otf',
    'Noto Serif SC': 'NotoSerifSC-Regular.otf',
}

# Thread-safe global state for font loading
_fonts_lock = threading.Lock()
_fonts_loaded = False

def set_font(font='Noto Sans CJK TC') -> None:
    """
    Set the matplotlib font for Chinese text rendering.
    
    Configures matplotlib to use a Chinese-compatible font for plots
    and visualizations. This is required for proper display of Chinese
    characters in matplotlib figures.
    
    Args:
        font (str): Font specification. Can be:
            - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
            - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
            - Path to a font file: '/path/to/font.otf' or '/path/to/font.ttf'
    
    Raises:
        FileNotFoundError: If a font file path is provided but doesn't exist.
        ValueError: If the font cannot be loaded or set.
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> from qhchina.helpers import set_font
        >>> 
        >>> set_font('sans')  # Use sans-serif font for Chinese
        >>> plt.title("中文标题")
        >>> plt.show()
        >>> 
        >>> # Or use serif font
        >>> set_font('serif-sc')
    """
    global _fonts_loaded
    
    # Check if font is a file path
    is_file_path = False
    font_path = None
    resolved_font = font
    
    # Detect if input is a font file path (must end with .otf, .ttf, .OTF, or .TTF)
    if isinstance(font, (str, Path)):
        font_str = str(font)
        if font_str.endswith(('.otf', '.ttf', '.OTF', '.TTF')):
            font_path = Path(font_str)
            if font_path.exists() and font_path.is_file():
                is_file_path = True
            elif not font_path.exists():
                raise FileNotFoundError(f"Font file not found: {font_path}")
    
    if is_file_path:
        # Load custom font file
        try:
            matplotlib.font_manager.fontManager.addfont(str(font_path))
            # Extract font name from the file
            font_props = matplotlib.font_manager.FontProperties(fname=str(font_path))
            resolved_font = font_props.get_name()
        except Exception as e:
            raise ValueError(f"Error loading custom font from: {font_path}") from e
    else:
        # Auto-load bundled fonts if not already loaded (thread-safe check)
        with _fonts_lock:
            if not _fonts_loaded:
                load_fonts(target_font=None, verbose=False)
        
        # Resolve alias to actual font name
        resolved_font = FONT_ALIASES.get(font, font)
    
    try:
        # Determine if this is a serif or sans-serif font (case-insensitive)
        is_serif = 'serif' in resolved_font.lower()
        
        if is_serif:
            # Set serif font list and family
            matplotlib.rcParams['font.serif'] = [resolved_font, 'serif']
            matplotlib.rcParams['font.family'] = 'serif'
        else:
            # Set sans-serif font list and family
            matplotlib.rcParams['font.sans-serif'] = [resolved_font, 'sans-serif']
            matplotlib.rcParams['font.family'] = 'sans-serif'
        
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        raise ValueError(f"Error setting font '{resolved_font}' (from input: '{font}')") from e

def load_fonts(target_font: str = 'Noto Sans CJK TC', verbose: bool = False) -> list[dict] | None:
    """
    Load bundled CJK fonts into matplotlib.
    
    Copies the package's CJK fonts to matplotlib's font directory and registers
    them for use. This function is thread-safe and idempotent.
    
    Args:
        target_font (str): Font to set as default after loading. Options:
            - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
            - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
            - None: Load fonts but don't set a default
        verbose (bool): If True, print loading details and return font info.
    
    Returns:
        list[dict] | None: When verbose=True, returns list of font info dicts:
            - 'font_name': Full font name (e.g., 'Noto Sans CJK TC')
            - 'aliases': List of aliases (e.g., ['sans', 'sans-tc'])
            - 'path': Absolute path to the font file
            When verbose=False, returns None.
    
    Raises:
        OSError: If fonts cannot be copied to matplotlib directory.
    
    Example:
        >>> from qhchina.helpers import load_fonts
        >>> # Load fonts and set sans-serif as default
        >>> load_fonts('sans')
        >>> 
        >>> # Get info about available fonts
        >>> fonts = load_fonts(target_font=None, verbose=True)
        >>> for f in fonts:
        ...     print(f"{f['font_name']}: {f['aliases']}")
    """
    global _fonts_loaded
    
    if verbose:
        logger.info(f"{PACKAGE_PATH=}")
        logger.info(f"{CJK_FONT_PATH=}")
        logger.info(f"{MPL_FONT_PATH=}")
    cjk_fonts = [file.name for file in Path(f'{CJK_FONT_PATH}').glob('**/*') if not file.name.startswith(".")]
    
    errors = []
    for font in cjk_fonts:
        try:
            source = Path(f'{CJK_FONT_PATH}/{font}').resolve()
            target = Path(f'{MPL_FONT_PATH}/{font}').resolve()
            # Only copy if target doesn't exist or is older than source
            if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
                shutil.copy(source, target)
                if verbose:
                    logger.info(f"Copied font: {font}")
            matplotlib.font_manager.fontManager.addfont(f'{target}')
            if verbose:
                logger.info(f"Loaded font: {font}")
        except Exception as e:
            errors.append(f"{font}: {e}")
    
    with _fonts_lock:
        if errors and not _fonts_loaded:
            # Only raise on first load attempt if all fonts failed
            if len(errors) == len(cjk_fonts):
                raise OSError(f"Failed to load any fonts. Errors: {errors}")
        
        # Mark fonts as loaded
        _fonts_loaded = True
    
    if target_font:
        # Resolve alias before setting
        resolved_font = FONT_ALIASES.get(target_font, target_font)
        if verbose:
            if target_font != resolved_font:
                logger.info(f"Resolving alias '{target_font}' to '{resolved_font}'")
            logger.info(f"Setting font to: {resolved_font}")
        set_font(target_font)
    
    # Build list of font info dictionaries
    # Create reverse mapping: font_name -> list of aliases
    font_to_aliases = {}
    for alias, font_name in FONT_ALIASES.items():
        if font_name not in font_to_aliases:
            font_to_aliases[font_name] = []
        font_to_aliases[font_name].append(alias)
    
    if verbose:
        font_info_list = []
        for font_name, font_file in FONT_FILES.items():
            font_info_list.append({
                'font_name': font_name,
                'aliases': font_to_aliases.get(font_name, []),
                'path': str(MPL_FONT_PATH / font_file)
            })
        return font_info_list

def get_font_path(font: str = 'Noto Sans CJK TC') -> str:
    """
    Get the file path to a CJK font for external libraries.
    
    Returns the path to a font file that can be used with libraries
    that require explicit font paths (e.g., WordCloud, PIL).
    
    Args:
        font (str): Font name or alias. Options:
            - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
            - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
    
    Returns:
        str: Absolute path to the font file.
    
    Raises:
        ValueError: If the font name is not recognized.
    
    Example:
        >>> from qhchina.helpers import get_font_path
        >>> from wordcloud import WordCloud
        >>> 
        >>> font_path = get_font_path('sans')
        >>> wc = WordCloud(font_path=font_path, width=800, height=400)
        >>> wc.generate("这是一个词云示例")
    """
    # Resolve alias to font name
    resolved_font = FONT_ALIASES.get(font, font)
    
    # Get font file name
    font_file = FONT_FILES.get(resolved_font)
    if font_file is None:
        raise ValueError(f"Unknown font: '{font}'. Available fonts: {list(FONT_FILES.keys())}")
    
    return str(MPL_FONT_PATH / font_file)

def current_font() -> Optional[str]:
    """
    Get the name of the currently configured matplotlib font.
    
    Returns:
        str | None: The current font name, or None if no font is configured.
    
    Raises:
        RuntimeError: If there's an error accessing font configuration.
    
    Example:
        >>> from qhchina.helpers import set_font, current_font
        >>> set_font('serif-tc')
        >>> print(current_font())
        'Noto Serif TC'
    """
    try:
        # Check serif first if family is serif
        if matplotlib.rcParams.get('font.family') == ['serif']:
            fonts = matplotlib.rcParams.get('font.serif', [])
        else:
            fonts = matplotlib.rcParams.get('font.sans-serif', [])
        return fonts[0] if fonts else None
    except (KeyError, IndexError):
        return None
    except Exception as e:
        raise RuntimeError(f"Error getting current font: {e}") from e

def list_available_fonts() -> dict:
    """
    List all CJK fonts bundled with the package.
    
    Returns:
        dict: Mapping of font file names to their internal font names.
            Example: {'NotoSansTCSC-Regular.otf': 'Noto Sans CJK TC'}
    
    Example:
        >>> from qhchina.helpers import list_available_fonts
        >>> fonts = list_available_fonts()
        >>> for filename, fontname in fonts.items():
        ...     print(f"{filename} -> {fontname}")
    """
    font_info = {}
    cjk_fonts = [file for file in Path(f'{CJK_FONT_PATH}').glob('*.otf') if not file.name.startswith(".")]
    
    for font_file in cjk_fonts:
        try:
            font_props = matplotlib.font_manager.FontProperties(fname=str(font_file))
            font_name = font_props.get_name()
            font_info[font_file.name] = font_name
        except Exception as e:
            logger.error(f"Error reading font: {font_file.name}")
            logger.error(f"Error: {e}")
    
    return font_info

def list_font_aliases() -> dict:
    """
    List all available font aliases.
    
    Returns:
        dict: Mapping of short aliases to full font names.
            Example: {'sans': 'Noto Sans CJK TC', 'serif-tc': 'Noto Serif TC'}
    
    Example:
        >>> from qhchina.helpers import list_font_aliases
        >>> aliases = list_font_aliases()
        >>> print(aliases)
        {'sans': 'Noto Sans CJK TC', 'sans-tc': 'Noto Sans CJK TC', ...}
    """
    return FONT_ALIASES.copy()
