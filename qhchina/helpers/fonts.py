import shutil
import matplotlib
import matplotlib.font_manager
from pathlib import Path

HELPERS_PATH = Path(__file__).parents[0].resolve()
CJK_FONT_PATH = Path(f'{HELPERS_PATH}/fonts').resolve()
MPL_FONT_PATH = Path(f'{matplotlib.get_data_path()}/fonts/ttf').resolve()

def set_font(font='Noto Sans CJK TC') -> None:
    matplotlib.rcParams['font.sans-serif'] = [font, 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

def load_fonts(target_font : str = 'Noto Sans CJK TC') -> None:
    print(HELPERS_PATH)
    print(CJK_FONT_PATH)
    print(MPL_FONT_PATH)
    cjk_fonts = [file.name for file in Path(f'{CJK_FONT_PATH}').glob('**/*') if not file.name.startswith(".")]
    for font in cjk_fonts:
        source = Path(f'{CJK_FONT_PATH}/{font}').resolve()
        target = Path(f'{MPL_FONT_PATH}/{font}').resolve()
        shutil.copy(source, target)
        matplotlib.font_manager.fontManager.addfont(f'{target}')
        print(font)
    if target_font:
        set_font(target_font)
