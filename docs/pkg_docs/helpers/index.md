---
layout: docs_with_sidebar
title: Helper Utilities
permalink: /pkg_docs/helpers/
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
- `target_font` (str): Font name or alias to set as default. Options: `'sans'`, `'sans-tc'`, `'sans-sc'`, `'serif-tc'`, `'serif-sc'`, or full names: `'Noto Sans CJK TC'`, `'Noto Serif TC'`, `'Noto Serif SC'`
- `verbose` (bool): Print detailed loading information

```python
set_font(font='Noto Sans CJK TC')
```

Set the matplotlib font for Chinese text rendering.

**Parameters:**
- `font` (str): Font name or alias (same options as `load_fonts`)

```python
current_font()
```

Get the currently active font name.

**Returns:** (str) Current font name

```python
list_available_fonts()
```

Get dictionary of bundled font files and their internal names.

**Returns:** (dict) Mapping of font file names to font names

```python
list_font_aliases()
```

Get dictionary of font aliases and their corresponding names.

**Returns:** (dict) Mapping of aliases to full font names

### Available Fonts

| Font File | Font Name | Aliases | Description |
|-----------|-----------|---------|-------------|
| `NotoSansTCSC-Regular.otf` | Noto Sans CJK TC | `'sans'`, `'sans-tc'`, `'sans-sc'` | Sans-serif font with Traditional and Simplified Chinese |
| `NotoSerifTC-Regular.otf` | Noto Serif TC | `'serif-tc'` | Serif font for Traditional Chinese |
| `NotoSerifSC-Regular.otf` | Noto Serif SC | `'serif-sc'` | Serif font for Simplified Chinese |

## Text Loading

### Functions

```python
load_text(filepath, encoding='utf-8')
```

Load text from a single file.

**Parameters:**
- `filepath` (str): Path to the text file
- `encoding` (str): File encoding

**Returns:** (str) Text content of the file

```python
load_texts(filepaths, encoding='utf-8')
```

Load text from multiple files.

**Parameters:**
- `filepaths` (list): List of file paths
- `encoding` (str): File encoding

**Returns:** (list) List of text contents

```python
load_stopwords(language='zh_sim')
```

Load stopwords for filtering.

**Parameters:**
- `language` (str): Language code (`'zh_sim'` for Simplified Chinese, `'zh_tr'` for Traditional Chinese)

**Returns:** (set) Set of stopwords

```python
get_stopword_languages()
```

Get all available stopword language codes.

**Returns:** (list) List of available language codes

```python
split_into_chunks(sequence, chunk_size, overlap=0.0)
```

Split text or a list of tokens into chunks with optional overlap.

**Parameters:**
- `sequence` (str or list): Text string or list of tokens to split
- `chunk_size` (int): Size of each chunk (characters for text, items for lists)
- `overlap` (float): Fraction of overlap between consecutive chunks (0.0 to 1.0)

**Returns:** (list) List of chunks

## Examples

### Basic Font Setup

```python
from qhchina.helpers import load_fonts
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

### Loading Texts and Stopwords

```python
from qhchina.helpers import load_text, load_texts, load_stopwords, split_into_chunks

# Load a single text file
text = load_text('document.txt')

# Load multiple files
texts = load_texts(['file1.txt', 'file2.txt', 'file3.txt'])

# Load stopwords
stopwords = load_stopwords('zh_sim')

# Split long text into chunks
chunks = split_into_chunks(text, chunk_size=1000, overlap=0.1)
```
