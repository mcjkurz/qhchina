---
layout: docs_with_sidebar
title: Collocation Analysis
permalink: /pkg_docs/collocations/
---

# Collocation Analysis

The `qhchina.analytics.collocations` module provides tools for identifying words that frequently co-occur together in text.

## Functions

```python
find_collocates(sentences, target_words, method='window', horizon=5, filters=None, 
                as_dataframe=True, max_sentence_length=256, alternative='greater')
```

Statistical significance is computed using Fisher's exact test with the "greater" alternative by default, testing whether observed co-occurrence exceeds expected frequency.

**Parameters:**
- `sentences` (list): List of tokenized sentences (each sentence is a list of tokens)
- `target_words` (str or list): Target word(s) to find collocates for
- `method` (str): Collocation method
  - `'window'`: Use sliding window of specified horizon (default)
  - `'sentence'`: Use whole sentences as context
- `horizon` (int or tuple): Context window size (only used if `method='window'`)
  - `int`: Symmetric window (e.g., `5` means 5 words on each side)
  - `tuple`: Asymmetric window `(left, right)` (e.g., `(0, 5)` means only 5 words to the right)
- `max_sentence_length` (int): Maximum sentence length for preprocessing. Sentences longer than this value are truncated to avoid excessive memory usage from outliers. Default is 256 tokens. Set to `None` for no limit. Used by both `'window'` and `'sentence'` methods.
- `alternative` (str): Alternative hypothesis for Fisher's exact test. Options are `'greater'` (default), `'less'`, or `'two-sided'`.
- `filters` (dict): Optional filters to apply *after* the statistics are computed on the full corpus:
  - `'max_p'`: Maximum p-value threshold for statistical significance
  - `'stopwords'`: List of words to exclude
  - `'min_word_length'`: Minimum character length for collocates
  - `'min_exp_local'`: Minimum expected local frequency
  - `'max_exp_local'`: Maximum expected local frequency
  - `'min_obs_local'`: Minimum observed local frequency
  - `'max_obs_local'`: Maximum observed local frequency
  - `'min_ratio_local'`: Minimum local frequency ratio (obs/exp)
  - `'max_ratio_local'`: Maximum local frequency ratio (obs/exp)
  - `'min_obs_global'`: Minimum global frequency
  - `'max_obs_global'`: Maximum global frequency
- `as_dataframe` (bool): Return results as pandas DataFrame

**Returns:** (DataFrame or list) Collocation statistics containing:
- `target`: The target word
- `collocate`: A co-occurring word
- `exp_local`: Expected frequency of co-occurrence
- `obs_local`: Observed frequency of co-occurrence
- `ratio_local`: Ratio of observed to expected frequency
- `obs_global`: Total frequency of the collocate
- `p_value`: Statistical significance of the association

<br>

```python
plot_collocates(collocates, x_col='ratio_local', y_col='p_value', 
                x_scale='log', y_scale='log', color=None, colormap='viridis', 
                color_by=None, title=None, figsize=(10, 8), fontsize=10, 
                show_labels=False, label_top_n=None, alpha=0.6, marker_size=50, 
                show_diagonal=False, diagonal_color='red', filename=None, 
                xlabel=None, ylabel=None)
```

Visualize collocation results as a flexible 2D scatter plot.

**Parameters:**
- `collocates` (DataFrame or list): Output from `find_collocates`
- `x_col` (str): Column for x-axis (default: `'ratio_local'`)
- `y_col` (str): Column for y-axis (default: `'p_value'`)
- `x_scale`, `y_scale` (str): Axis scales - `'log'` (default), `'linear'`, `'symlog'`, or `'logit'`
- `color` (str or list): Color(s) for points
- `colormap` (str): Matplotlib colormap when using `color_by` (default: `'viridis'`)
- `color_by` (str): Column to color points by (e.g., `'ratio_local'`, `'obs_local'`)
- `show_labels` (bool): Show text labels for collocates
- `label_top_n` (int): Label only top N points. For `p_value`, labels smallest (most significant) values; for other columns, labels largest values.
- `show_diagonal` (bool): Draw y=x diagonal reference line (useful for obs vs exp plots)
- `diagonal_color` (str): Color for diagonal line (default: `'red'`)
- `alpha` (float): Point transparency (0.0 to 1.0)
- `marker_size` (int): Size of scatter points
- `figsize` (tuple): Figure size (width, height) in inches
- `fontsize` (int): Base font size
- `filename` (str): Save plot to file (optional)
- `xlabel`, `ylabel` (str): Custom axis labels (auto-generated if `None`)

**Returns:** None (displays the plot)

<br>

```python
cooc_matrix(documents, method='window', horizon=5, min_abs_count=1, min_doc_count=1, 
            vocab_size=None, binary=False, as_dataframe=True, vocab=None, 
            use_sparse=False)
```

Create a co-occurrence matrix from a collection of documents.

**Parameters:**
- `documents` (list): List of tokenized documents (each document is a list of tokens)
- `method` (str): Co-occurrence method
  - `'window'`: Use sliding window (default)
  - `'document'`: Use whole documents as context
- `horizon` (int or tuple): Context window size (only used if `method='window'`)
  - `int`: Symmetric window (e.g., `5` means 5 words on each side)
  - `tuple`: Asymmetric window `(left, right)` (e.g., `(0, 5)` means only 5 words to the right)
- `min_abs_count` (int): Minimum word frequency to include
- `min_doc_count` (int): Minimum number of documents a word must appear in
- `vocab_size` (int): Maximum vocabulary size (optional)
- `binary` (bool): Count co-occurrences as binary (0/1) rather than frequencies
- `as_dataframe` (bool): Return matrix as pandas DataFrame
- `vocab` (list or set): Predefined vocabulary to use (optional)
- `use_sparse` (bool): Use sparse matrix for better memory efficiency

**Returns:** 
- If `as_dataframe=True`: pandas DataFrame with rows and columns labeled by vocabulary
- If `as_dataframe=False` and `use_sparse=False`: tuple of (numpy array, word_to_index dictionary)
- If `as_dataframe=False` and `use_sparse=True`: tuple of (scipy sparse matrix, word_to_index dictionary)

<br>

## Examples

### Finding Collocates

```python
from qhchina.analytics.collocations import find_collocates

# Example tokenized sentences
sentences = [
    ["中国", "经济", "发展", "改革"],
    ["美国", "经济", "市场", "金融"],
    ["中国", "市场", "贸易", "改革"],
    # More sentences...
]

# Find collocates of "经济" using window method
collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    method="window",
    horizon=3,  # 3 words on each side
    filters={
        'max_p': 0.05,          # Only statistically significant
        'stopwords': ["的", "了"],
        'min_word_length': 2
    },
    as_dataframe=True
)

# Find words that appear to the RIGHT of "经济"
# Use horizon=(3, 0) to look left from candidate positions
right_collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    horizon=(3, 0),  # Look 3 positions left from candidates → finds words to the right of target
    filters={'max_p': 0.05}
)

# Find words that appear to the LEFT of "经济"
# Use horizon=(0, 3) to look right from candidate positions
left_collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    horizon=(0, 3),  # Look 3 positions right from candidates → finds words to the left of target
    filters={'max_p': 0.05}
)

# Display top collocates
top_collocates = collocates.sort_values("p_value").head(10)
for _, row in top_collocates.iterrows():
    print(f"{row['collocate']}: obs={row['obs_local']}, ratio={row['ratio_local']:.2f}, p={row['p_value']:.4f}")
```

### Visualizing Collocates

```python
from qhchina.analytics.collocations import find_collocates, plot_collocates

# Find collocates
collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    alternative='two-sided',
    filters={'max_p': 0.05, 'min_obs_local': 2}
)

# Default: ratio vs p-value (log scales)
plot_collocates(collocates, title="Collocates of 经济")

# Observed vs expected with diagonal reference line
plot_collocates(
    collocates,
    x_col='exp_local',
    y_col='obs_local',
    x_scale='log',
    y_scale='log',
    show_diagonal=True,
    title='Observed vs Expected'
)

# Color by ratio, label top 15 most strongly associated
plot_collocates(
    collocates,
    x_col='obs_global',
    y_col='p_value',
    color_by='ratio_local',
    colormap='RdYlBu_r',
    show_labels=True,
    label_top_n=15,
    title='Corpus Frequency vs Significance'
)
```

### Creating Co-occurrence Matrix

```python
from qhchina.analytics.collocations import cooc_matrix

# Create co-occurrence matrix
cooc = cooc_matrix(
    documents=sentences,
    method="window",
    horizon=2,
    min_abs_count=2,
    as_dataframe=True
)

# Find words that co-occur with a target word
target_word = "经济"
if target_word in cooc.index:
    cooc_with_target = cooc[target_word].sort_values(ascending=False)
    print(f"Words co-occurring with '{target_word}':")
    print(cooc_with_target.head(10))
```
