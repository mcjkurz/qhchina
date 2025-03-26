import shutil
import matplotlib
import matplotlib.font_manager
from pathlib import Path

PACKAGE_PATH = Path(__file__).parents[1].resolve() # qhchina
CJK_FONT_PATH = Path(f'{PACKAGE_PATH}/data/fonts').resolve()
MPL_FONT_PATH = Path(f'{matplotlib.get_data_path()}/fonts/ttf').resolve()

def set_font(font='Noto Sans CJK TC') -> None:
    matplotlib.rcParams['font.sans-serif'] = [font, 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

def load_fonts(target_font : str = 'Noto Sans CJK TC', verbose=False) -> None:
    if verbose:
        print(f"{PACKAGE_PATH=}")
        print(f"{CJK_FONT_PATH=}")
        print(f"{MPL_FONT_PATH=}")
    cjk_fonts = [file.name for file in Path(f'{CJK_FONT_PATH}').glob('**/*') if not file.name.startswith(".")]
    
    for font in cjk_fonts:
        source = Path(f'{CJK_FONT_PATH}/{font}').resolve()
        target = Path(f'{MPL_FONT_PATH}/{font}').resolve()
        shutil.copy(source, target)
        matplotlib.font_manager.fontManager.addfont(f'{target}')
        if verbose:
            print(f"Loaded font: {font}")
    if target_font:
        if verbose:
            print(f"Setting font to: {target_font}")
        set_font(target_font)

def current_font() -> str:
    return matplotlib.rcParams['font.sans-serif'][0]
