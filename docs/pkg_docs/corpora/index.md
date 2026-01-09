---
layout: docs_with_sidebar
title: Corpus Analysis
permalink: /pkg_docs/corpora/
functions:
  - name: compare_corpora()
    anchor: compare_corpora
---

# Corpus Analysis

The `qhchina.analytics.corpora` module provides tools for comparing corpora and identifying linguistic patterns.

```python
from qhchina.analytics.corpora import compare_corpora

results = compare_corpora(corpus_a, corpus_b, method="fisher")
words_in_a = results[results["rel_ratio"] > 1]  # Words more common in corpus A
```

---

<h3 id="compare_corpora">compare_corpora()</h3>

```python
compare_corpora(corpusA, corpusB, method='fisher', filters=None, 
                as_dataframe=True)
```

Identify statistically significant differences in word usage between two corpora.

**Parameters:**
- `corpusA` (list): First corpus - either a flat list of tokens or a list of sentences (each sentence being a list of tokens)
- `corpusB` (list): Second corpus - same format as corpusA
- `method` (str): Statistical test to use (all tests use two-sided alternatives)
  - `'fisher'`: Fisher's exact test (default)
  - `'chi2'`: Chi-square test without correction
  - `'chi2_corrected'`: Chi-square test with Yates' correction
- `filters` (dict): Optional filters to apply *after* the statistics are computed on the full corpora:
  - `'min_count'`: Minimum count threshold for a word to be included. Can be:
    - Single int (applies to both corpora)
    - Tuple of (min_countA, min_countB)
  - `'max_p'`: Maximum p-value threshold for statistical significance
  - `'stopwords'`: List of words to exclude
  - `'min_word_length'`: Minimum character length for words
- `as_dataframe` (bool): Return results as pandas DataFrame

**Returns:** (DataFrame or list) Comparison statistics containing:
- `word`: The word being compared
- `abs_freqA`: Absolute frequency in corpus A
- `abs_freqB`: Absolute frequency in corpus B
- `rel_freqA`: Relative frequency in corpus A
- `rel_freqB`: Relative frequency in corpus B
- `rel_ratio`: Ratio of relative frequencies (A:B)
- `p_value`: Statistical significance of the difference

**Note:** A small p-value indicates that the difference in word frequency between corpora is statistically significant. A `rel_ratio` > 1 indicates the word is more common in corpus A; < 1 indicates more common in corpus B. Two-sided tests are used because we want to detect whether words are overrepresented in either corpus.

---

## Examples

**Basic Corpus Comparison**

```python
from qhchina.analytics.corpora import compare_corpora

# Example corpora (tokenized)
corpus_a = ["中国", "经济", "发展", "改革", "经济", "政策", "中国", "市场"]
corpus_b = ["美国", "经济", "市场", "金融", "美国", "贸易", "进口", "出口"]

# Compare the corpora
results = compare_corpora(
    corpusA=corpus_a,
    corpusB=corpus_b,
    method="fisher",
    filters={
        "min_count": 3,      # Minimum count in both corpora
        "max_p": 0.05,       # Only statistically significant differences
        "stopwords": ["的", "了"],
        "min_word_length": 2
    },
    as_dataframe=True
)

# Sort by statistical significance
results = results.sort_values("p_value")

# Identify words overrepresented in corpus A
words_in_A = results[
    (results["p_value"] < 0.05) & 
    (results["rel_ratio"] > 1)
]

# Identify words overrepresented in corpus B
words_in_B = results[
    (results["p_value"] < 0.05) & 
    (results["rel_ratio"] < 1)
]

print("Words more common in Corpus A:")
print(words_in_A[["word", "rel_ratio", "p_value"]].head(10))

print("\nWords more common in Corpus B:")
print(words_in_B[["word", "rel_ratio", "p_value"]].head(10))
```

**Visualizing Corpus Differences**

```python
import matplotlib.pyplot as plt
import numpy as np

# Get top significant differences
top_words = results.sort_values("p_value").head(10)

# Plot as bar chart
plt.figure(figsize=(10, 6))
plt.barh(
    top_words["word"],
    np.log2(top_words["rel_ratio"]),
    color=[("blue" if ratio > 1 else "red") for ratio in top_words["rel_ratio"]]
)
plt.axvline(x=0, color="black", linestyle="-")
plt.xlabel("Log2 Ratio (Corpus A / Corpus B)")
plt.title("Most Significant Word Frequency Differences")
plt.tight_layout()
plt.show()
```
