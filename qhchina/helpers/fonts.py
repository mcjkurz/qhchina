import shutil
import matplotlib
import requests
from pathlib import Path

PACKAGE_PATH = Path(__file__).parents[0].resolve()
CJK_FONT_URL = f'https://github.com/mcjkurz/qhchina/raw/refs/heads/main/qhchina/data/fonts/'
CJK_FONT_PATH = Path(f'{PACKAGE_PATH}/data/fonts').resolve()
MPL_FONT_PATH = Path(f'{matplotlib.get_data_path()}/fonts/ttf').resolve()

def set_font(font='Noto Sans CJK TC') -> None:
    matplotlib.rcParams['font.sans-serif'] = [font, 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

def download_font(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

def load_font(font_filenames : list[str] = ["NotoSansCJKTC-Regular.otf"]) -> None:
    for filename in font_filenames:
        font_path = CJK_FONT_PATH / filename
        font_path.parent.mkdir(parents=True, exist_ok=True)
        download_font(CJK_FONT_URL, font_path)

    cjk_fonts = [file.name for file in Path(f'{CJK_FONT_PATH}').glob('**/*')]
    for font in cjk_fonts:
        print(font)
        source = Path(f'{CJK_FONT_PATH}/{font}').resolve()
        target = Path(f'{MPL_FONT_PATH}/{font}').resolve()
        shutil.copy(source, target)
        matplotlib.font_manager.fontManager.addfont(f'{target}')
    set_font('Noto Sans CJK TC')