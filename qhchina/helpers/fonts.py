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

def set_font(font='Noto Sans CJK TC') -> None:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font, 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Error setting font: {font}")
        print(f"Error: {e}")

def load_fonts(target_font : str = 'Noto Sans CJK TC', verbose=False) -> None:
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
        if verbose:
            print(f"Setting font to: {target_font}")
        set_font(target_font)

def current_font() -> str:
    try:
        return matplotlib.rcParams['font.sans-serif'][0]
    except Exception as e:
        print(f"Error getting current font")
        print(f"Error: {e}")
        return None
