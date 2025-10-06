import shutil
try:
    import matplotlib
    import matplotlib.font_manager
except Exception as e:
    print(f"Error importing matplotlib: {e}")
    raise e
from pathlib import Path

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

def set_font(font='Noto Sans CJK TC') -> None:
    """
    Set the matplotlib font for Chinese text rendering.
    
    Args:
        font: Font name or alias. Can be:
              - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
              - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
    """
    # Resolve alias to actual font name
    resolved_font = FONT_ALIASES.get(font, font)
    
    try:
        # Determine if this is a serif or sans-serif font
        is_serif = 'Serif' in resolved_font
        
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
        print(f"Error setting font: {resolved_font} (from input: {font})")
        print(f"Error: {e}")

def load_fonts(target_font : str = 'Noto Sans CJK TC', verbose=False) -> None:
    """
    Load CJK fonts into matplotlib and optionally set a default font.
    
    Args:
        target_font: Font name or alias to set as default. Can be:
                     - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
                     - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
                     - None: Load fonts but don't set a default
        verbose: If True, print detailed loading information
    """
    if verbose:
        print(f"{PACKAGE_PATH=}")
        print(f"{CJK_FONT_PATH=}")
        print(f"{MPL_FONT_PATH=}")
    cjk_fonts = [file.name for file in Path(f'{CJK_FONT_PATH}').glob('**/*') if not file.name.startswith(".")]
    
    for font in cjk_fonts:
        try:
            source = Path(f'{CJK_FONT_PATH}/{font}').resolve()
            target = Path(f'{MPL_FONT_PATH}/{font}').resolve()
            shutil.copy(source, target)
            matplotlib.font_manager.fontManager.addfont(f'{target}')
            if verbose:
                print(f"Loaded font: {font}")
        except Exception as e:
            print(f"Error loading font: {font}")
            print(f"Matplotlib font directory path: {MPL_FONT_PATH}")
            print(f"Error: {e}")
    if target_font:
        # Resolve alias before setting
        resolved_font = FONT_ALIASES.get(target_font, target_font)
        if verbose:
            if target_font != resolved_font:
                print(f"Resolving alias '{target_font}' to '{resolved_font}'")
            print(f"Setting font to: {resolved_font}")
        set_font(target_font)

def current_font() -> str:
    try:
        return matplotlib.rcParams['font.sans-serif'][0]
    except Exception as e:
        print(f"Error getting current font")
        print(f"Error: {e}")
        return None

def list_available_fonts() -> dict:
    """
    List all available CJK fonts bundled with the package.
    Returns a dictionary mapping font file names to their internal font names.
    """
    font_info = {}
    cjk_fonts = [file for file in Path(f'{CJK_FONT_PATH}').glob('*.otf') if not file.name.startswith(".")]
    
    for font_file in cjk_fonts:
        try:
            font_props = matplotlib.font_manager.FontProperties(fname=str(font_file))
            font_name = font_props.get_name()
            font_info[font_file.name] = font_name
        except Exception as e:
            print(f"Error reading font: {font_file.name}")
            print(f"Error: {e}")
    
    return font_info

def list_font_aliases() -> dict:
    """
    List all available font aliases for convenient access.
    Returns a dictionary mapping aliases to their full font names.
    """
    return FONT_ALIASES.copy()
