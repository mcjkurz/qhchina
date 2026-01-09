---
layout: docs_with_sidebar
title: Helper Utilities
permalink: /pkg_docs/helpers/
functions:
  - name: load_fonts()
    anchor: font-management
  - name: set_font()
    anchor: font-management
  - name: get_font_path()
    anchor: font-management
  - name: load_text()
    anchor: text-loading
  - name: load_stopwords()
    anchor: text-loading
  - name: split_into_chunks()
    anchor: text-loading
---

# Helper Utilities

The `qhchina.helpers` module provides utilities for font management and text loading when working with Chinese texts.

## Font Management

### Functions

```python
load_fonts(target_font='Noto Sans CJK TC', verbose=False)
```

Load CJK fonts into matplotlib and set a default font.

**Parameters:**
- `target_font` (str): Font name or alias to set as default. Options:
  - Bundled fonts: `'sans'`, `'sans-tc'`, `'sans-sc'`, `'serif-tc'`, `'serif-sc'`, or full names: `'Noto Sans CJK TC'`, `'Noto Serif TC'`, `'Noto Serif SC'`
  - Custom font: Path to your own font file (`.otf` or `.ttf`)
  - `None`: Load bundled fonts only without setting a default
- `verbose` (bool): Print detailed loading information

**Returns:** (list[dict]) List of dictionaries, each containing:
- `'font_name'`: Full font name (e.g., `'Noto Sans CJK TC'`)
- `'aliases'`: List of aliases for the font (e.g., `['sans', 'sans-tc', 'sans-sc']`)
- `'path'`: Absolute path to the font file

**Example:**

```python
from qhchina.helpers import load_fonts

fonts = load_fonts()
for font in fonts:
    print(f"{font['font_name']}: {font['aliases']}")
    # Noto Sans CJK TC: ['sans', 'sans-tc', 'sans-sc']
    # Noto Serif TC: ['serif-tc']
    # Noto Serif SC: ['serif-sc']
```

<br>

```python
set_font(font='Noto Sans CJK TC')
```

Set the matplotlib font for Chinese text rendering.

**Parameters:**
- `font` (str): Font name, alias, or path to font file. Options:
  - Bundled fonts: `'sans'`, `'sans-tc'`, `'sans-sc'`, `'serif-tc'`, `'serif-sc'`, or full names: `'Noto Sans CJK TC'`, `'Noto Serif TC'`, `'Noto Serif SC'`
  - Custom font: Path to your own font file (`.otf` or `.ttf`)

<br>

```python
current_font()
```

Get the currently active font name.

**Returns:** (str) Current font name

<br>

```python
list_available_fonts()
```

Get dictionary of bundled font files and their internal names.

**Returns:** (dict) Mapping of font file names to font names

<br>

```python
list_font_aliases()
```

Get dictionary of font aliases and their corresponding names.

**Returns:** (dict) Mapping of aliases to full font names

---

```python
get_font_path(font='Noto Sans CJK TC')
```

Get the file path for a CJK font. Useful when you need to pass a font path to external libraries like WordCloud.

**Parameters:**
- `font` (str): Font name or alias. Options:
  - Bundled fonts: `'sans'`, `'sans-tc'`, `'sans-sc'`, `'serif-tc'`, `'serif-sc'`, or full names: `'Noto Sans CJK TC'`, `'Noto Serif TC'`, `'Noto Serif SC'`

**Returns:** (str) Absolute path to the font file

**Example:**

```python
from qhchina.helpers import load_fonts, get_font_path
from wordcloud import WordCloud

# Load fonts first - returns info about all loaded fonts
fonts = load_fonts()

# Get font path for WordCloud using get_font_path
font_path = get_font_path('sans')
wc = WordCloud(font_path=font_path, width=800, height=400)

# Or use the path directly from load_fonts return value
sans_font = next(f for f in fonts if f['font_name'] == 'Noto Sans CJK TC')
wc = WordCloud(font_path=sans_font['path'], width=800, height=400)
```

Note: `get_font_path` can also be imported directly from `qhchina`.

---

### Available Fonts

<style>
.bordered-table {
  border-collapse: collapse;
  width: 100%;
}
.bordered-table th,
.bordered-table td {
  border: 1px solid black;
  padding: 8px;
  text-align: left;
}
.bordered-table th {
  background-color: #f2f2f2;
}
</style>

<div markdown="0">
<table class="bordered-table">
  <thead>
    <tr>
      <th>Font File</th>
      <th>Font Name</th>
      <th>Aliases</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>NotoSansTCSC-Regular.otf</code></td>
      <td>Noto Sans CJK TC</td>
      <td><code>'sans'</code>, <code>'sans-tc'</code>, <code>'sans-sc'</code></td>
      <td>Sans-serif font with Traditional and Simplified Chinese</td>
    </tr>
    <tr>
      <td><code>NotoSerifTC-Regular.otf</code></td>
      <td>Noto Serif TC</td>
      <td><code>'serif-tc'</code></td>
      <td>Serif font for Traditional Chinese</td>
    </tr>
    <tr>
      <td><code>NotoSerifSC-Regular.otf</code></td>
      <td>Noto Serif SC</td>
      <td><code>'serif-sc'</code></td>
      <td>Serif font for Simplified Chinese</td>
    </tr>
  </tbody>
</table>
</div>

## Text Loading

### Functions

```python
load_text(filepath, encoding='utf-8')
```

Load text from a single file.

**Parameters:**
- `filepath` (str): Path to the text file
- `encoding` (str): File encoding. Use `'auto'` to automatically detect encoding.

**Returns:** (str) Text content of the file

<br>

```python
load_texts(filepaths, encoding='utf-8')
```

Load text from multiple files.

**Parameters:**
- `filepaths` (list): List of file paths
- `encoding` (str): File encoding. Use `'auto'` to automatically detect encoding for each file.

**Returns:** (list) List of text contents

<br>

```python
detect_encoding(filename, num_bytes=10000)
```

Detect the encoding of a file.

**Parameters:**
- `filename` (str): Path to the file
- `num_bytes` (int): Number of bytes to read for detection (default: 10000)

**Returns:** (str) Detected encoding (e.g., 'utf-8', 'gb18030')

**Note:** Requires the `chardet` package (`pip install chardet`)

<br>

```python
load_stopwords(language='zh_sim')
```

Load stopwords for filtering.

**Parameters:**
- `language` (str): Language code. Available options:
  - `'zh_sim'` - Simplified Chinese
  - `'zh_tr'` - Traditional Chinese
  - `'zh_cl_sim'` - Classical Chinese (Simplified)
  - `'zh_cl_tr'` - Classical Chinese (Traditional)

**Returns:** (set) Set of stopwords

<br>

```python
get_stopword_languages()
```

Get all available stopword language codes.

**Returns:** (list) List of available language codes

<br>

```python
split_into_chunks(sequence, chunk_size, overlap=0.0)
```

Split text or a list of tokens into chunks with optional overlap.

**Parameters:**
- `sequence` (str or list): Text string or list of tokens to split
- `chunk_size` (int): Size of each chunk (characters for text, items for lists)
- `overlap` (float): Fraction of overlap between consecutive chunks (0.0 to 1.0)

**Returns:** (list) List of chunks

<br>

## Examples

### Basic Font Setup

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

### Using Custom Fonts

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

### Loading Texts and Stopwords

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
