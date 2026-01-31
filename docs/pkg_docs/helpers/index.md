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
has_examples: True
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

Load CJK fonts into matplotlib and optionally set a default font.

This function is thread-safe and can be called from multiple threads simultaneously.

**Parameters:**
- `target_font`: Font name or alias to set as default. Can be:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
  - None: Load fonts but don't set a default
- `verbose`: If True, print detailed loading information and return font info

**Returns:**
list[dict] | None: Only when verbose=True, returns a list of dictionaries,
each containing:
- 'font_name': Full font name (e.g., 'Noto Sans CJK TC')
- 'aliases': List of aliases for the font (e.g., ['sans', 'sans-tc'])
- 'path': Absolute path to the font file
When verbose=False, returns None.

**Raises:**
- `OSError`: If fonts cannot be copied to matplotlib directory.

<br>

<h3 id="current_font">current_font()</h3>

```python
current_font()
```

Get the currently configured font name.

**Returns:**
The current font name, or None if no font is configured.

**Raises:**
- `RuntimeError`: If there's an error accessing font configuration.

<br>

<h3 id="set_font">set_font()</h3>

```python
set_font(font='Noto Sans CJK TC')
```

Set the matplotlib font for Chinese text rendering.

This function is thread-safe.

**Parameters:**
- `font`: Font name, alias, or path to font file. Can be:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'
  - Path to font file: '/path/to/font.otf' or '/path/to/font.ttf'

**Raises:**
- `FileNotFoundError`: If a font file path is provided but doesn't exist.
- `ValueError`: If the font cannot be loaded or set.

<br>

<h3 id="list_available_fonts">list_available_fonts()</h3>

```python
list_available_fonts()
```

List all available CJK fonts bundled with the package.

Returns a dictionary mapping font file names to their internal font names.

<br>

<h3 id="list_font_aliases">list_font_aliases()</h3>

```python
list_font_aliases()
```

List all available font aliases for convenient access.

Returns a dictionary mapping aliases to their full font names.

<br>

<h3 id="get_font_path">get_font_path()</h3>

```python
get_font_path(font: str = 'Noto Sans CJK TC')
```

Get the file path for a CJK font (for use with WordCloud, etc.).

**Parameters:**
- `font`: Font name or alias. Can be:
  - Full font name: 'Noto Sans CJK TC', 'Noto Serif TC', 'Noto Serif SC'
  - Alias: 'sans', 'sans-tc', 'sans-sc', 'serif-tc', 'serif-sc'

**Returns:**
(str) Absolute path to the font file

**Example:**
```python
font_path = qhchina.get_font_path()
wc = WordCloud(font_path=font_path, ...)
```

<br>

<h3 id="load_text">load_text()</h3>

```python
load_text(filename, encoding='utf-8')
```

Loads text from a file.

**Parameters:**
- `filename` (str): The filename to load text from.
- `encoding` (str): The encoding of the file. Default is "utf-8".
  Use "auto" to automatically detect the encoding.

**Returns:**
(str) The text content of the file.

<br>

<h3 id="load_texts">load_texts()</h3>

```python
load_texts(filenames, encoding='utf-8')
```

Loads text from multiple files.

**Parameters:**
- `filenames` (list): A list of filenames to load text from.
- `encoding` (str): The encoding of the files. Default is "utf-8".
  Use "auto" to automatically detect encoding for each file.

**Returns:**
(list) A list of text contents from the files.

<br>

<h3 id="load_stopwords">load_stopwords()</h3>

```python
load_stopwords(language: str = 'zh_sim')
```

Load stopwords from a file for the specified language.

**Parameters:**
- `language`: Language code (default: "zh_sim" for simplified Chinese).
  Use get_stopword_languages() to see available options.

**Returns:**
Set of stopwords

**Raises:**
- `ValueError`: If the specified language is not available.

<br>

<h3 id="split_into_chunks">split_into_chunks()</h3>

```python
split_into_chunks(sequence, chunk_size, overlap=0.0)
```

Splits text or a list of tokens into chunks with optional overlap between consecutive chunks.

**Parameters:**
- `sequence` (str or list): The text string or list of tokens to be split.
- `chunk_size` (int): The size of each chunk (characters for text, items for lists).
- `overlap` (float): The fraction of overlap between consecutive chunks (0.0 to 1.0).
  Default is 0.0 (no overlap).

**Returns:**
(list) A list of chunks. If input is a string, each chunk is a string.
If input is a list, each chunk is a list of tokens.
Note: The last chunk may be smaller than chunk_size if the sequence
doesn't divide evenly.

**Raises:**
- `ValueError`: If overlap is not between 0 and 1, or if chunk_size is not positive.

<br>

<h3 id="get_stopword_languages">get_stopword_languages()</h3>

```python
get_stopword_languages()
```

Get all available stopword language codes.

**Returns:**
List of available language codes (e.g., ['zh_sim', 'zh_cl_sim', 'zh_cl_tr'])

<br>

<h3 id="detect_encoding">detect_encoding()</h3>

```python
detect_encoding(filename, num_bytes=10000)
```

Detects the encoding of a file.

**Parameters:**
- `filename` (str): The path to the file.
- `num_bytes` (int): Number of bytes to read for detection. Default is 10000.
  Larger values may be more accurate but slower.

**Returns:**
(str) The detected encoding (e.g., 'utf-8', 'gb2312', 'gbk', 'big5').

**Raises:**
- `ImportError`: If chardet is not installed.

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
