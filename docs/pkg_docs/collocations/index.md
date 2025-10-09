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
                as_dataframe=True, max_sentence_length=256, batch_size=10000, 
                alternative='greater')
```

Statistical significance is computed using Fisher's exact test with the "greater" alternative by default, testing whether observed co-occurrence exceeds expected frequency.

**Parameters:**
- `sentences` (list): List of tokenized sentences (each sentence is a list of tokens)
- `target_words` (str or list): Target word(s) to find collocates for
- `method` (str): Collocation method
  - `'window'`: Use sliding window of specified horizon (default)
  - `'sentence'`: Use whole sentences as context
- `horizon` (int): Context window size (only used if `method='window'`)
- `max_sentence_length` (int): Maximum sentence length for preprocessing. Longer sentences are truncated to avoid memory bloat. Default is 256. Set to `None` for no limit.
- `batch_size` (int): Number of sentences to process per batch. Default is 10000. Controls memory usage - smaller batches use less RAM. For typical use cases, the default works well. Adjust only if memory-constrained (use smaller values) or have abundant RAM (use larger values for marginal speed gains).
- `alternative` (str): Alternative hypothesis for Fisher's exact test. Options are `'greater'` (default), `'less'`, or `'two-sided'`.
- `filters` (dict): Optional filters to apply *after* the statistics are computed on the full corpus:
  - `'max_p'`: Maximum p-value threshold for statistical significance
  - `'stopwords'`: List of words to exclude
  - `'min_length'`: Minimum character length for collocates
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
- `horizon` (int): Context window size (only used if `method='window'`)
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
    horizon=3,
    filters={
        'max_p': 0.05,          # Only statistically significant
        'stopwords': ["的", "了"],
        'min_length': 2
    },
    as_dataframe=True
)

# Display top collocates
top_collocates = collocates.sort_values("p_value").head(10)
for _, row in top_collocates.iterrows():
    print(f"{row['collocate']}: obs={row['obs_local']}, ratio={row['ratio_local']:.2f}, p={row['p_value']:.4f}")
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
