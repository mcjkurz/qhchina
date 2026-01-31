---
layout: docs_with_sidebar
title: Helper Utilities
permalink: /pkg_docs/helpers/
functions:
  - name: load_fonts()
    anchor: load_fonts
  - name: current_font()
    anchor: current_font
  - name: set_font()
    anchor: set_font
  - name: list_available_fonts()
    anchor: list_available_fonts
  - name: list_font_aliases()
    anchor: list_font_aliases
  - name: get_font_path()
    anchor: get_font_path
  - name: load_text()
    anchor: load_text
  - name: load_texts()
    anchor: load_texts
  - name: load_stopwords()
    anchor: load_stopwords
  - name: split_into_chunks()
    anchor: split_into_chunks
  - name: get_stopword_languages()
    anchor: get_stopword_languages
  - name: detect_encoding()
    anchor: detect_encoding
import_from: qhchina.helpers
include_imported: True
---

# Helper Utilities

The `qhchina.helpers` module provides utilities for font management and text loading when working with Chinese texts.

---

## API Reference

<!-- API-START -->

<h3 id="load_fonts">load_fonts()</h3>

```python
load_fonts(target_font: str = 'Noto Sans CJK TC', verbose: bool = False)
```

Load bundled CJK fonts into matplotlib.

Copies the package's CJK fonts to matplotlib's font directory and registers
them for use. This function is thread-safe and idempotent.

**Parameters:**
- `target_font` (str): Font to set as default after loading. Options:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
  - None: Load fonts but don't set a default
- `verbose` (bool): If True, print loading details and return font info.

**Returns:**
list[dict] | None: When verbose=True, returns list of font info dicts:
- 'font_name': Full font name (e.g., 'Noto Sans CJK TC')
- 'aliases': List of aliases (e.g., ['sans', 'sans-tc'])
- 'path': Absolute path to the font file
When verbose=False, returns None.

**Raises:**
- `OSError`: If fonts cannot be copied to matplotlib directory.

**Example:**
```python
>>> from qhchina.helpers import load_fonts
>>> # Load fonts and set sans-serif as default
>>> load_fonts('sans')
>>> 
>>> # Get info about available fonts
>>> fonts = load_fonts(target_font=None, verbose=True)
>>> for f in fonts:
...     print(f"{f['font_name']}: {f['aliases']}")
```

<br>

<h3 id="current_font">current_font()</h3>

```python
current_font()
```

Get the name of the currently configured matplotlib font.

**Returns:**
str | None: The current font name, or None if no font is configured.

**Raises:**
- `RuntimeError`: If there's an error accessing font configuration.

**Example:**
```python
>>> from qhchina.helpers import set_font, current_font
>>> set_font('serif-tc')
>>> print(current_font())
'Noto Serif TC'
```

<br>

<h3 id="set_font">set_font()</h3>

```python
set_font(font='Noto Sans CJK TC')
```

Set the matplotlib font for Chinese text rendering.

Configures matplotlib to use a Chinese-compatible font for plots
and visualizations. This is required for proper display of Chinese
characters in matplotlib figures.

**Parameters:**
- `font` (str): Font specification. Can be:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
  - Path to a font file: '/path/to/font.otf' or '/path/to/font.ttf'

**Raises:**
- `FileNotFoundError`: If a font file path is provided but doesn't exist.
- `ValueError`: If the font cannot be loaded or set.

**Example:**
```python
>>> import matplotlib.pyplot as plt
>>> from qhchina.helpers import set_font
>>> 
>>> set_font('sans')  # Use sans-serif font for Chinese
>>> plt.title("中文标题")
>>> plt.show()
>>> 
>>> # Or use serif font
>>> set_font('serif-sc')
```

<br>

<h3 id="list_available_fonts">list_available_fonts()</h3>

```python
list_available_fonts()
```

List all CJK fonts bundled with the package.

**Returns:**
(dict) Mapping of font file names to their internal font names.
Example: {'NotoSansTCSC-Regular.otf': 'Noto Sans CJK TC'}

**Example:**
```python
>>> from qhchina.helpers import list_available_fonts
>>> fonts = list_available_fonts()
>>> for filename, fontname in fonts.items():
...     print(f"{filename} -> {fontname}")
```

<br>

<h3 id="list_font_aliases">list_font_aliases()</h3>

```python
list_font_aliases()
```

List all available font aliases.

**Returns:**
(dict) Mapping of short aliases to full font names.
Example: {'sans': 'Noto Sans CJK TC', 'serif-tc': 'Noto Serif TC'}

**Example:**
```python
>>> from qhchina.helpers import list_font_aliases
>>> aliases = list_font_aliases()
>>> print(aliases)
{'sans': 'Noto Sans CJK TC', 'sans-tc': 'Noto Sans CJK TC', ...}
```

<br>

<h3 id="get_font_path">get_font_path()</h3>

```python
get_font_path(font: str = 'Noto Sans CJK TC')
```

Get the file path to a CJK font for external libraries.

Returns the path to a font file that can be used with libraries
that require explicit font paths (e.g., WordCloud, PIL).

**Parameters:**
- `font` (str): Font name or alias. Options:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'

**Returns:**
(str) Absolute path to the font file.

**Raises:**
- `ValueError`: If the font name is not recognized.

**Example:**
```python
>>> from qhchina.helpers import get_font_path
>>> from wordcloud import WordCloud
>>> 
>>> font_path = get_font_path('sans')
>>> wc = WordCloud(font_path=font_path, width=800, height=400)
>>> wc.generate("这是一个词云示例")
```

<br>

<h3 id="load_text">load_text()</h3>

```python
load_text(filename, encoding='utf-8')
```

Load text content from a single file.

**Parameters:**
- `filename` (str): Path to the text file.
- `encoding` (str): File encoding (default: "utf-8").
  Use "auto" to automatically detect the encoding using chardet.

**Returns:**
(str) The complete text content of the file.

**Raises:**
- `ValueError`: If filename is not a string.
- `FileNotFoundError`: If the file does not exist.

**Example:**
```python
>>> from qhchina.helpers import load_text
>>> text = load_text("novel.txt", encoding="utf-8")
>>> # Or with auto-detection for unknown encodings
>>> text = load_text("old_text.txt", encoding="auto")
```

<br>

<h3 id="load_texts">load_texts()</h3>

```python
load_texts(filenames, encoding='utf-8')
```

Load text content from multiple files.

**Parameters:**
- `filenames` (list): List of file paths to load.
  Can also pass a single string for one file.
- `encoding` (str): File encoding (default: "utf-8").
  Use "auto" to detect encoding for each file individually.

**Returns:**
(list) List of text strings, one per file, in the same order as filenames.

**Example:**
```python
>>> from qhchina.helpers import load_texts
>>> import glob
>>> files = glob.glob("corpus/*.txt")
>>> texts = load_texts(files)
>>> print(f"Loaded {len(texts)} documents")
```

<br>

<h3 id="load_stopwords">load_stopwords()</h3>

```python
load_stopwords(language: str = 'zh_sim')
```

Load a stopword list for Chinese text processing.

Provides pre-built stopword lists for different Chinese variants.
These can be used with segmenters and other text processing tools.

**Parameters:**
- `language` (str): Stopword list identifier. Available options:
  - 'zh_sim': Modern simplified Chinese (default)
  - 'zh_tr': Modern traditional Chinese
  - 'zh_cl_sim': Classical Chinese in simplified characters
  - 'zh_cl_tr': Classical Chinese in traditional characters
  Use get_stopword_languages() to see all available options.

**Returns:**
(set) A set of stopword strings.

**Raises:**
- `ValueError`: If the specified language is not available.

**Example:**
```python
>>> from qhchina.helpers import load_stopwords
>>> from qhchina.preprocessing import create_segmenter
>>> 
>>> stopwords = load_stopwords("zh_sim")
>>> segmenter = create_segmenter(
...     backend="jieba",
...     filters={"stopwords": stopwords}
... )
```

<br>

<h3 id="split_into_chunks">split_into_chunks()</h3>

```python
split_into_chunks(sequence, chunk_size, overlap=0.0)
```

Split a sequence into fixed-size chunks with optional overlap.

Works with both strings (splits by character) and lists (splits by item).
Useful for processing long texts that exceed model limits.

**Parameters:**
- `sequence` (str or list): The text or token list to split.
- `chunk_size` (int): Maximum size of each chunk.
  For strings: number of characters.
  For lists: number of items.
- `overlap` (float): Fraction of overlap between consecutive chunks (0.0 to 1.0).
  Default is 0.0 (no overlap). Use overlap for context preservation
  when processing with models that need surrounding context.

**Returns:**
(list) List of chunks (strings if input was string, lists if input was list).
The last chunk may be smaller than chunk_size.

**Raises:**
- `ValueError`: If overlap is not between 0.0 and 1.0.

**Example:**
```python
>>> from qhchina.helpers import split_into_chunks
>>> # Split text into 100-character chunks with 10% overlap
>>> text = "这是一段很长的中文文本..." * 50
>>> chunks = split_into_chunks(text, chunk_size=100, overlap=0.1)
>>> 
>>> # Split a token list
>>> tokens = ["word1", "word2", "word3", "word4", "word5"]
>>> chunks = split_into_chunks(tokens, chunk_size=2)
>>> print(chunks)
[['word1', 'word2'], ['word3', 'word4'], ['word5']]
```

<br>

<h3 id="get_stopword_languages">get_stopword_languages()</h3>

```python
get_stopword_languages()
```

List all available stopword language codes.

**Returns:**
(list) Sorted list of available language codes.
Typical values include: 'zh_sim', 'zh_tr', 'zh_cl_sim', 'zh_cl_tr'

**Example:**
```python
>>> from qhchina.helpers import get_stopword_languages
>>> print(get_stopword_languages())
['zh_cl_sim', 'zh_cl_tr', 'zh_sim', 'zh_tr']
```

<br>

<h3 id="detect_encoding">detect_encoding()</h3>

```python
detect_encoding(filename, num_bytes=10000)
```

Detect the encoding of a text file automatically.

Uses the chardet library to detect the character encoding of a file
by analyzing a sample of bytes from the beginning.

**Parameters:**
- `filename` (str): Path to the file to analyze.
- `num_bytes` (int): Number of bytes to read for detection (default: 10000).
  Larger values improve accuracy but slow down detection.

**Returns:**
(str) The detected encoding (e.g., 'utf-8', 'gb18030', 'big5').
Returns 'gb18030' for GB2312/GBK files as it's a superset.

**Raises:**
- `ImportError`: If chardet is not installed.

**Example:**
```python
>>> from qhchina.helpers import detect_encoding
>>> encoding = detect_encoding("chinese_text.txt")
>>> print(encoding)
'utf-8'
```

<br>

<!-- API-END -->

---

## Examples

**Basic Font Setup**

```python
from qhchina.helpers import load_fonts, set_font
import matplotlib.pyplot as plt

# Load fonts and set Traditional Chinese serif as default
load_fonts('serif-tc')

# Create a plot with Chinese text
plt.figure(figsize=(8, 6))
plt.title('中國古典詩歌分析')
plt.xlabel('時間')
plt.ylabel('頻率')
plt.show()
```

**Using Custom Fonts**

```python
from qhchina.helpers import set_font
import matplotlib.pyplot as plt

# Use your own font file
set_font('/path/to/your/custom-font.otf')

# Or set it when loading fonts
from qhchina.helpers import load_fonts
load_fonts(target_font='/path/to/your/custom-font.ttf')

# Now your plots will use the custom font
plt.figure(figsize=(8, 6))
plt.title('使用自定義字體')
plt.show()
```

**Loading Texts and Stopwords**

```python
from qhchina.helpers import load_text, load_texts, load_stopwords, split_into_chunks
from qhchina.helpers.texts import detect_encoding, get_stopword_languages

# Load a single text file
text = load_text('document.txt')

# Load with automatic encoding detection (requires chardet)
text = load_text('古文.txt', encoding='auto')

# Detect encoding manually
encoding = detect_encoding('古文.txt')
print(f"Detected encoding: {encoding}")

# Load multiple files
texts = load_texts(['file1.txt', 'file2.txt', 'file3.txt'])

# See available stopword languages
languages = get_stopword_languages()
print(f"Available: {languages}")  # ['zh_cl_sim', 'zh_cl_tr', 'zh_sim', 'zh_tr']

# Load stopwords
stopwords = load_stopwords('zh_sim')

# Split long text into chunks
chunks = split_into_chunks(text, chunk_size=1000, overlap=0.1)
```
