---
layout: docs_with_sidebar
title: Helper Utilities
permalink: /pkg_docs/helpers/
---

# Helper Utilities in qhChina

The qhChina package provides helper utilities for common tasks when working with Chinese texts, including font management for matplotlib visualizations and text loading utilities.

## Font Management

When creating visualizations with Chinese text in matplotlib, you need to configure fonts that support Chinese characters. The `qhchina.helpers` module provides convenient functions for loading and managing CJK (Chinese, Japanese, Korean) fonts.

### Quick Start

```python
from qhchina.helpers import load_fonts

# Load all bundled CJK fonts and set default to sans-serif
load_fonts()

# Or use a convenient alias to load and set a specific font
load_fonts('serif-tc')  # Traditional Chinese serif
load_fonts('serif-sc')  # Simplified Chinese serif
load_fonts('sans')      # Sans-serif (default)
```

### Available Fonts

The package includes three Noto fonts for Chinese text:

| Font File | Font Name | Alias | Description |
|-----------|-----------|-------|-------------|
| `NotoSansTCSC-Regular.otf` | Noto Sans CJK TC | `'sans'`, `'sans-tc'`, `'sans-sc'` | Sans-serif font with both Traditional and Simplified Chinese |
| `NotoSerifTC-Regular.otf` | Noto Serif TC | `'serif-tc'` | Serif font for Traditional Chinese |
| `NotoSerifSC-Regular.otf` | Noto Serif SC | `'serif-sc'` | Serif font for Simplified Chinese |

### Font Styles: Sans-Serif vs. Serif

**Sans-serif** fonts (like Noto Sans) are clean and modern without decorative strokes:
- Best for: UI elements, presentations, digital displays
- Characteristics: Clean lines, contemporary appearance

**Serif** fonts (like Noto Serif) have small decorative strokes at the ends of characters:
- Best for: Books, formal documents, traditional texts
- Characteristics: Classical appearance, traditional feel

### Loading Fonts

The `load_fonts()` function copies the bundled fonts into matplotlib's font directory and registers them:

```python
from qhchina.helpers import load_fonts

# Basic usage - load fonts and set sans-serif as default
load_fonts()

# Load fonts and set serif font for Traditional Chinese
load_fonts('serif-tc')

# Use full font name instead of alias
load_fonts('Noto Serif SC')

# Verbose mode to see what's happening
load_fonts('serif-tc', verbose=True)
```

### Switching Fonts

You can change fonts at any time using `set_font()`:

```python
from qhchina.helpers import set_font
import matplotlib.pyplot as plt

# Set to Traditional Chinese serif
set_font('serif-tc')

# Create a plot with Chinese text
plt.figure(figsize=(8, 6))
plt.title('中國古典詩歌分析')
plt.xlabel('時間')
plt.ylabel('頻率')
plt.show()

# Switch to Simplified Chinese sans-serif
set_font('sans-sc')

# Create another plot
plt.figure(figsize=(8, 6))
plt.title('中国古典诗歌分析')
plt.show()
```

### Checking Current Font

```python
from qhchina.helpers import current_font

# Get the currently active font
font = current_font()
print(f"Current font: {font}")  # e.g., "Noto Serif TC"
```

### Discovering Available Fonts

Use `list_available_fonts()` to see all bundled fonts and their names:

```python
from qhchina.helpers import list_available_fonts

# Get dictionary of font files and their internal names
fonts = list_available_fonts()

for filename, fontname in fonts.items():
    print(f"{filename} → {fontname}")

# Output:
# NotoSansTCSC-Regular.otf → Noto Sans CJK TC
# NotoSerifTC-Regular.otf → Noto Serif TC
# NotoSerifSC-Regular.otf → Noto Serif SC
```

Use `list_font_aliases()` to see all available aliases:

```python
from qhchina.helpers import list_font_aliases

# Get dictionary of aliases and their corresponding font names
aliases = list_font_aliases()

for alias, fontname in aliases.items():
    print(f"{alias} → {fontname}")

# Output:
# sans → Noto Sans CJK TC
# sans-tc → Noto Sans CJK TC
# sans-sc → Noto Sans CJK TC
# serif-tc → Noto Serif TC
# serif-sc → Noto Serif SC
```

### Complete Example

```python
import matplotlib.pyplot as plt
from qhchina.helpers import load_fonts, set_font, current_font

# Load fonts and set Traditional Chinese serif as default
load_fonts('serif-tc')
print(f"Loaded font: {current_font()}")

# Create a figure with Traditional Chinese
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot with serif font
ax1.text(0.5, 0.5, '古典詩詞研究', 
         fontsize=24, ha='center', va='center')
ax1.set_title(f'襯線字體 (Serif)\n{current_font()}')
ax1.axis('off')

# Switch to sans-serif for right plot
set_font('sans')
ax2.text(0.5, 0.5, '古典詩詞研究', 
         fontsize=24, ha='center', va='center')
ax2.set_title(f'無襯線字體 (Sans-Serif)\n{current_font()}')
ax2.axis('off')

plt.tight_layout()
plt.show()
```

## Text Loading Utilities

The helpers module also provides utilities for loading Chinese texts from files:

```python
from qhchina.helpers import load_text, load_texts, load_stopwords

# Load a single text file
text = load_text('path/to/document.txt')

# Load multiple text files
texts = load_texts(['file1.txt', 'file2.txt', 'file3.txt'])

# Load stopwords for filtering
stopwords_simplified = load_stopwords('zh_sim')  # Simplified Chinese
stopwords_traditional = load_stopwords('zh_tr')  # Traditional Chinese
```

### Splitting Text into Chunks

For processing large texts, you can split them into manageable chunks:

```python
from qhchina.helpers import split_into_chunks

long_text = "很长的文本内容..."

# Split into chunks of approximately 1000 characters
chunks = split_into_chunks(long_text, chunk_size=1000)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} characters")
```

## API Reference

### Font Management Functions

- **`load_fonts(target_font='Noto Sans CJK TC', verbose=False)`**: Load CJK fonts into matplotlib
  - `target_font`: Font name or alias to set as default
  - `verbose`: Print detailed loading information

- **`set_font(font='Noto Sans CJK TC')`**: Set the matplotlib font
  - `font`: Font name or alias (e.g., 'serif-tc', 'sans', 'Noto Serif SC')

- **`current_font()`**: Get the currently active font name

- **`list_available_fonts()`**: Get dictionary of bundled font files and their names

- **`list_font_aliases()`**: Get dictionary of font aliases and their corresponding names

### Text Loading Functions

- **`load_text(filepath)`**: Load text from a single file
  - Returns: String containing the file contents

- **`load_texts(filepaths)`**: Load text from multiple files
  - `filepaths`: List of file paths
  - Returns: List of strings

- **`load_stopwords(language)`**: Load stopwords for filtering
  - `language`: 'zh_sim' (Simplified) or 'zh_tr' (Traditional)
  - Returns: Set of stopwords

- **`split_into_chunks(text, chunk_size=1000)`**: Split text into chunks
  - `text`: Text to split
  - `chunk_size`: Approximate size of each chunk in characters
  - Returns: List of text chunks

