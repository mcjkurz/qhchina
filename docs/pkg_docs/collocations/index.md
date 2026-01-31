---
layout: docs_with_sidebar
title: Collocation Analysis
permalink: /pkg_docs/collocations/
functions:
  - name: FilterOptions
    anchor: filteroptions
  - name: find_collocates()
    anchor: find_collocates
  - name: cooc_matrix()
    anchor: cooc_matrix
  - name: plot_collocates()
    anchor: plot_collocates
import_from: qhchina.analytics.collocations
---

# Collocation Analysis

The `qhchina.analytics.collocations` module provides tools for identifying words that frequently co-occur together in text.

```python
from qhchina.analytics.collocations import find_collocates

collocates = find_collocates(sentences, target_words=["经济"], horizon=5)
top_collocates = collocates.sort_values("p_value").head(10)  # Most significant collocates
```

---

## API Reference

<!-- API-START -->

<h3 id="filteroptions">FilterOptions</h3>

```python
FilterOptions(*args, **kwargs)
```

Type definition for filter options in collocation analysis.

<br>

<h3 id="find_collocates">find_collocates()</h3>

```python
find_collocates(
    sentences: List[List[str]],
    target_words: Union[str, List[str]],
    method: str = 'window',
    horizon: Union[int, tuple, NoneType] = None,
    filters: Optional[qhchina.analytics.collocations.FilterOptions] = None,
    as_dataframe: bool = True,
    max_sentence_length: Optional[int] = 256,
    alternative: str = 'greater'
)
```

Find collocates for target words within a corpus of sentences.

Parameters:
-----------
sentences : List[List[str]]
    List of tokenized sentences, where each sentence is a list of tokens.
target_words : Union[str, List[str]]
    Target word(s) to find collocates for.
method : str, default='window'
    Method to use for calculating collocations. Either 'window' or 'sentence'.
    - 'window': Uses a sliding window of specified horizon around each token
    - 'sentence': Considers whole sentences as context units (horizon not applicable)
horizon : Optional[Union[int, tuple]], default=None
    Context window size relative to the target word. Only applicable when method='window'. 
    Must be None when method='sentence'.
    - int: Symmetric window (e.g., 5 means 5 words on each side of target)
    - tuple: Asymmetric window (left, right) specifying how many words to look
             on each side of the target word:
             - (0, 5) finds collocates up to 5 words to the RIGHT of target
             - (5, 0) finds collocates up to 5 words to the LEFT of target
             - (2, 3) finds collocates 2 words left and 3 words right of target
    - None: Uses default of 5 for 'window' method
filters : Optional[FilterOptions], optional
    Dictionary of filters to apply to results, AFTER computation is done:
    - 'max_p': float - Maximum p-value threshold for statistical significance
    - 'stopwords': List[str] - Words to exclude from results
    - 'min_word_length': int - Minimum character length for collocates
    - 'min_exp_local': float - Minimum expected local frequency
    - 'max_exp_local': float - Maximum expected local frequency
    - 'min_obs_local': int - Minimum observed local frequency
    - 'max_obs_local': int - Maximum observed local frequency
    - 'min_ratio_local': float - Minimum local frequency ratio (obs/exp)
    - 'max_ratio_local': float - Maximum local frequency ratio (obs/exp)
    - 'min_obs_global': int - Minimum global frequency
    - 'max_obs_global': int - Maximum global frequency
as_dataframe : bool, default=True
    If True, return results as a pandas DataFrame.
max_sentence_length : Optional[int], default=256
    Maximum sentence length for preprocessing. Used by both 'window' and 'sentence' methods.
    Longer sentences will be truncated to avoid memory bloat from outliers. 
    Set to None for no limit (may use a lot of memory with very long sentences).
alternative : str, default='greater'
    Alternative hypothesis for Fisher's exact test. Options are:
    - 'greater': Test if observed co-occurrence is greater than expected (default)
    - 'less': Test if observed co-occurrence is less than expected
    - 'two-sided': Test if observed co-occurrence differs from expected

Returns:
--------
Union[List[Dict], pd.DataFrame]
    List of dictionaries or DataFrame containing collocation statistics.

<br>

<h3 id="cooc_matrix">cooc_matrix()</h3>

```python
cooc_matrix(
    documents: List[List[str]],
    method: str = 'window',
    horizon: Union[int, Tuple[int, int], NoneType] = None,
    min_abs_count: int = 1,
    min_doc_count: int = 1,
    vocab_size: Optional[int] = None,
    binary: bool = False,
    as_dataframe: bool = True,
    vocab: Union[List[str], set, NoneType] = None,
    use_sparse: bool = False
)
```

Calculate a co-occurrence matrix from a list of documents.

Parameters:
-----------
documents : list
    List of tokenized documents, where each document is a list of tokens.
method : str, default='window'
    Method to use for calculating co-occurrences. Either 'window' or 'document'.
horizon : Optional[Union[int, tuple]], default=None
    Context window size relative to each word. Only applicable when method='window'.
    Must be None when method='document'.
    - int: Symmetric window (e.g., 5 means 5 words on each side)
    - tuple: Asymmetric window (left, right) specifying words on each side:
             - (0, 5) counts co-occurrences with words up to 5 positions to the RIGHT
             - (5, 0) counts co-occurrences with words up to 5 positions to the LEFT
    - None: Uses default of 5 for 'window' method
min_abs_count : int, default=1
    Minimum absolute count for a word to be included in the vocabulary.
min_doc_count : int, default=1
    Minimum number of documents a word must appear in to be included.
vocab_size : int, optional
    Maximum size of the vocabulary. Words are sorted by frequency.
binary : bool, default=False
    If True, count co-occurrences as binary (0/1) rather than frequencies.
as_dataframe : bool, default=True
    If True, return the co-occurrence matrix as a pandas DataFrame.
vocab : list or set, optional
    Predefined vocabulary to use. Words will still be filtered by min_abs_count and min_doc_count.
    If vocab_size is also provided, only the top vocab_size words will be kept.
use_sparse : bool, default=False
    If True, use a sparse matrix representation for better memory efficiency with large vocabularies.
    
Returns:
--------
If as_dataframe=True:
    pandas DataFrame with rows and columns labeled by vocabulary
If as_dataframe=False and use_sparse=False:
    tuple of (numpy array, word_to_index dictionary)
If as_dataframe=False and use_sparse=True:
    tuple of (scipy sparse matrix, word_to_index dictionary)

<br>

<h3 id="plot_collocates">plot_collocates()</h3>

```python
plot_collocates(
    collocates: Union[List[Dict], pandas.core.frame.DataFrame],
    x_col: str = 'ratio_local',
    y_col: str = 'p_value',
    x_scale: str = 'log',
    y_scale: str = 'log',
    color: Union[str, List[str], NoneType] = None,
    colormap: str = 'viridis',
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    fontsize: int = 10,
    show_labels: bool = False,
    label_top_n: Optional[int] = None,
    alpha: float = 0.6,
    marker_size: int = 50,
    show_diagonal: bool = False,
    diagonal_color: str = 'red',
    filename: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
)
```

Visualize collocation results as a 2D scatter plot.

Creates a customizable scatter plot from collocation data. By default, plots
ratio_local (x-axis) vs p_value (y-axis) with logarithmic scales, but allows
full flexibility to plot any columns with any scale type.

Parameters:
-----------
collocates : Union[List[Dict], pd.DataFrame]
    Output from find_collocates, either as a list of dictionaries or DataFrame.
x_col : str, default='ratio_local'
    Column name to plot on x-axis. Common choices: 'ratio_local', 'obs_local',
    'exp_local', 'obs_global'.
y_col : str, default='p_value'
    Column name to plot on y-axis. Common choices: 'p_value', 'obs_local',
    'ratio_local', 'obs_global'.
x_scale : str, default='log'
    Scale for x-axis. Options: 'log', 'linear', 'symlog', 'logit'.
    For ratio_local, 'log' makes the scale symmetric around 1.
y_scale : str, default='log'
    Scale for y-axis. Options: 'log', 'linear', 'symlog', 'logit'.
    For p_value, 'log' is recommended to visualize small values.
color : Optional[Union[str, List[str]]], default=None
    Color(s) for the points. Can be a single color string, list of colors,
    or None to use default.
colormap : str, default='viridis'
    Matplotlib colormap to use when color_by is specified.
color_by : Optional[str], default=None
    Column name to use for coloring points (e.g., 'obs_local', 'obs_global').
title : Optional[str], default=None
    Title for the plot.
figsize : tuple, default=(10, 8)
    Figure size as (width, height) in inches.
fontsize : int, default=10
    Base font size for labels.
show_labels : bool, default=False
    Whether to show collocate text labels next to points.
label_top_n : Optional[int], default=None
    If specified, only label the top N points. When color_by is set, ranks by that
    column; otherwise ranks by y-axis values. For p_value, labels smallest (most 
    significant) values; for other metrics, labels largest values.
alpha : float, default=0.6
    Transparency of points (0.0 to 1.0).
marker_size : int, default=50
    Size of markers.
show_diagonal : bool, default=False
    Whether to draw a diagonal reference line (y=x). Useful for observed vs
    expected plots to show where values match perfectly.
diagonal_color : str, default='red'
    Color of the diagonal reference line.
filename : Optional[str], default=None
    If provided, saves the figure to the specified file path.
xlabel : Optional[str], default=None
    Label for x-axis. If None, auto-generated from x_col and x_scale.
ylabel : Optional[str], default=None
    Label for y-axis. If None, auto-generated from y_col and y_scale.

Returns:
--------
None
    Displays the plot using matplotlib. To further customize, use plt.gca() 
    to get the current axes object after calling this function.

Examples:
---------
>>> # Basic usage: ratio vs p-value with log scales (default)
>>> collocates = find_collocates(sentences, ['天'])
>>> plot_collocates(collocates)

>>> # Plot observed vs expected frequency
>>> plot_collocates(collocates, x_col='exp_local', y_col='obs_local',
...                 x_scale='linear', y_scale='linear')

>>> # Plot global frequency vs ratio with custom scales
>>> plot_collocates(collocates, x_col='obs_global', y_col='ratio_local',
...                 x_scale='log', y_scale='log')

>>> # With labels and custom styling
>>> plot_collocates(collocates, show_labels=True, label_top_n=20,
...                 color='red', title='Collocates of 天')

>>> # Color by a column
>>> plot_collocates(collocates, color_by='obs_local', colormap='plasma')

<br>

<!-- API-END -->

---

## Examples

**Finding Collocates**

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
right_collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    horizon=(0, 3),  # 0 words left, 3 words right of target
    filters={'max_p': 0.05}
)

# Find words that appear to the LEFT of "经济"
left_collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    horizon=(3, 0),  # 3 words left, 0 words right of target
    filters={'max_p': 0.05}
)

# Display top collocates
top_collocates = collocates.sort_values("p_value").head(10)
for _, row in top_collocates.iterrows():
    print(f"{row['collocate']}: obs={row['obs_local']}, ratio={row['ratio_local']:.2f}, p={row['p_value']:.4f}")
```

**Visualizing Collocates**

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

**Creating Co-occurrence Matrix**

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
