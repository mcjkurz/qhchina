---
layout: docs_with_sidebar
title: Stylometry
permalink: /pkg_docs/stylometry/
functions:
  - name: Stylometry()
    anchor: stylometry
  - name: fit_transform()
    anchor: fit_transform
  - name: transform()
    anchor: transform
  - name: predict()
    anchor: predict
  - name: predict_author()
    anchor: predict_author
  - name: most_similar()
    anchor: most_similar
  - name: similarity()
    anchor: similarity
  - name: distance()
    anchor: distance
  - name: get_author_profile()
    anchor: get_author_profile
  - name: distance_matrix()
    anchor: distance_matrix
  - name: hierarchical_clustering()
    anchor: hierarchical_clustering
  - name: plot()
    anchor: plot
  - name: dendrogram()
    anchor: dendrogram
  - name: bootstrap_predict()
    anchor: bootstrap_predict
  - name: rolling_delta()
    anchor: rolling_delta
  - name: extract_mfw()
    anchor: extract_mfw
  - name: burrows_delta()
    anchor: burrows_delta
  - name: compute_yule_k()
    anchor: compute_yule_k
  - name: compare_corpora()
    anchor: compare_corpora
---

# Stylometry

The `qhchina.analytics.stylometry` module provides tools for authorship attribution and document clustering using statistical analysis of writing style. By default, the module uses z-score normalization to transform word frequencies, which standardizes feature values across documents and makes them comparable regardless of document length.

> **Note:** This module is inspired by the R package [stylo](https://github.com/computationalstylistics/stylo), a much more comprehensive implementation for computational stylistics.

```python
from qhchina.analytics.stylometry import Stylometry

# Corpus: dict mapping author names to lists of tokenized documents
corpus = {
    '鲁迅': [
        ['照', '我', '自己', '想', '虽然', '不', '是', '恶人', ...],
        ['当初', '他', '还', '只是', '冷笑', '随后', '眼光', '便', '凶狠', '起来', ...],
    ],
    '沈从文': [
        ['小溪', '流', '下去', '绕', '山岨', '流', ...],
        ['那', '条', '河水', '便是', '历史', '上', '知名', '的', '酉水', ...],
    ],
}

stylo = Stylometry(n_features=100, distance='cosine')
stylo.fit_transform(corpus)

# Analyze the transformed data
predicted = stylo.predict_author(unknown_text)  # Predict authorship
similar = stylo.most_similar('鲁迅_1')          # Find similar documents (returns similarity)
sim = stylo.similarity('鲁迅_1', '沈从文_1')    # Compare two documents (higher = more similar)
dist = stylo.distance('鲁迅_1', '沈从文_1')     # Compare two documents (lower = more similar)
```

---

<h3 id="stylometry">Stylometry()</h3>

The main class for stylometric analysis supports both supervised (with labeled training data) and unsupervised (clustering) approaches.

**Workflow:**
1. Create a `Stylometry` instance with desired parameters
2. Call `fit_transform()` with your corpus (dict or list of tokenized documents)
3. Analyze with: `plot()`, `dendrogram()`, `most_similar()`, `similarity()`, `distance()`, `predict()`

```python
Stylometry(n_features=100, ngram_range=(1, 1), transform='zscore',
           distance='cosine', classifier='delta', cull=None, chunk_size=None, mode='centroid')
```

**Parameters:**
- `n_features` (int): Number of most frequent n-grams to use as features (default: 100)
- `ngram_range` (tuple): Range of n-gram sizes as (min_n, max_n). Default (1, 1) = unigrams only. Use (1, 2) for unigrams + bigrams, (2, 3) for bigrams + trigrams, etc.
- `transform` (str): Feature transformation method
  - `'zscore'`: Z-score normalization (default)
  - `'tfidf'`: TF-IDF weighting
- `distance` (str): Distance metric for comparison
  - `'cosine'`: Cosine distance (default)
  - `'burrows_delta'`: Burrows' Delta (classic stylometric measure)
  - `'manhattan'`: Manhattan (L1) distance
  - `'euclidean'`: Euclidean (L2) distance
  - `'eder_delta'`: Eder's Delta (weighted variant of Burrows' Delta)
- `classifier` (str): Classification method
  - `'delta'`: Delta-based attribution (default)
  - `'svm'`: Support Vector Machine classification
- `cull` (float): Minimum document frequency ratio (0.0-1.0). N-grams appearing in fewer than cull×100% of documents are removed before feature selection. Default: None (no culling)
- `chunk_size` (int): If set, split documents into chunks of this many tokens before analysis. Useful for analyzing long documents. Default: None
- `mode` (str): Attribution mode for delta classifier
  - `'centroid'`: Aggregate all author texts into one profile per author (default)
  - `'instance'`: Keep individual texts separate (k-NN style)

<br>

<h3 id="fit_transform">fit_transform()</h3>

```python
fit_transform(corpus, labels=None)
```

Fit the model on a corpus and transform documents to feature vectors (z-score or TF-IDF, depending on `transform` parameter).

**Parameters:**
- `corpus`: Either:
  - **Dict** (supervised): `{'author_a': [[tok1, tok2, ...], [tok1, ...]], 'author_b': [...]}`
  - **List** (unsupervised): `[[tok1, tok2, ...], [tok1, ...], ...]`
- `labels` (list): Optional list of labels, **one per document** (must match corpus length). Documents sharing the same label are grouped together as belonging to the same author. If not provided, all documents are assigned the label `'unk'`. Ignored for dict input.

  | Labels | Result |
  |--------|--------|
  | `['A', 'A', 'B', 'B']` | Groups into `{'A': [doc1, doc2], 'B': [doc3, doc4]}` |
  | `['ch1', 'ch2', 'ch3']` | Each doc is its own group (useful for clustering) |
  | `None` | All docs grouped as `{'unk': [doc1, doc2, ...]}` |

Document IDs are generated based on grouping: when a label has only one document, the label is used directly as the ID (e.g., `chapter1`). When multiple documents share a label, IDs are suffixed with numbers (e.g., `author_a_1`, `author_a_2`).

**Returns:** self (for method chaining)

<br>

<h3 id="transform">transform()</h3>

```python
transform(tokens)
```

Transform a tokenized text to a feature vector using fitted features. This allows you to transform new documents without modifying the model's internal state.

**Parameters:**
- `tokens` (list): List of tokens (a tokenized document)

**Returns:** (numpy.ndarray) Feature vector of shape (n_features,)

<br>

<h3 id="predict">predict()</h3>

```python
predict(text, k=1)
```

Predict the most likely author for a tokenized text.

**Parameters:**
- `text` (list): List of tokens (the disputed text)
- `k` (int): Number of top results to return (must be ≥ 1). If `k` exceeds the number of available items, all items are returned.

**Returns:** (list) List of (author, distance) tuples sorted by distance ascending (most similar first)
- In `'centroid'` mode: returns top k author centroids by distance
- In `'instance'` mode: returns the k nearest documents with their author labels

<br>

<h3 id="predict_author">predict_author()</h3>

```python
predict_author(text, k=1)
```

Convenience method to get just the predicted author name.

**Parameters:**
- `text` (list): List of tokens
- `k` (int): For `'instance'` mode only: number of nearest neighbors for majority voting. In `'centroid'` mode, this parameter is ignored.

**Returns:** (str) Predicted author name

<br>

<h3 id="most_similar">most_similar()</h3>

```python
most_similar(query, k=None, return_distance=False)
```

Find the most similar documents to a query.

**Parameters:**
- `query`: Document ID (str) or list of tokens.
- `k` (int): Number of results to return. If None, returns all.
- `return_distance` (bool): If False (default), returns similarity (higher = more similar). If True, returns distance.

**Returns:** (list) List of (doc_id, value) tuples sorted by similarity.

<br>

<h3 id="similarity">similarity()</h3>

```python
similarity(a, b)
```

Compute the similarity between two documents. Higher = more similar.

**Parameters:**
- `a`, `b`: Document ID (str) or list of tokens.

**Returns:** (float) Similarity value. For cosine: -1 to 1. For others: 0 to 1.

<br>

<h3 id="distance">distance()</h3>

```python
distance(a, b)
```

Compute the distance between two documents. Lower = more similar.

**Parameters:**
- `a`, `b`: Document ID (str) or list of tokens.

**Returns:** (float) Distance value.

<br>

<h3 id="get_author_profile">get_author_profile()</h3>

```python
get_author_profile(author)
```

Get the feature values for a specific author.

**Parameters:**
- `author` (str): Author name

**Returns:** (DataFrame) DataFrame with 'feature' and 'value' columns, sorted by value descending

<br>

```python
get_feature_comparison()
```

Get a comparison table of feature values across all fitted authors.

**Returns:** (DataFrame) DataFrame with one column per author plus a 'variance' column

<br>

<h3 id="distance_matrix">distance_matrix()</h3>

```python
distance_matrix(level='document')
```

Compute pairwise distance matrix from fitted data.

**Parameters:**
- `level` (str): `'document'` for individual documents, `'author'` for author profiles

**Returns:** (tuple) (distance_matrix, labels)

<br>

<h3 id="hierarchical_clustering">hierarchical_clustering()</h3>

```python
hierarchical_clustering(method='average', level='document')
```

Perform hierarchical clustering on fitted data.

**Parameters:**
- `method` (str): Linkage method (`'single'`, `'complete'`, `'average'`, `'weighted'`, `'ward'`)
- `level` (str): `'document'` or `'author'`

**Returns:** (tuple) (linkage_matrix, labels) for scipy dendrogram

<br>

<h3 id="plot">plot()</h3>

```python
plot(method='pca', level='document', figsize=(10, 8), show_labels=True,
     labels=None, title=None, colors=None, marker_size=100, fontsize=12, 
     filename=None, random_state=42, show=True)
```

Create a 2D scatter plot of documents or authors.

**Parameters:**
- `method` (str): Dimensionality reduction method (`'pca'`, `'tsne'`, or `'mds'`)
- `level` (str): `'document'` for individual documents, `'author'` for author profiles
- `figsize` (tuple): Figure size as (width, height)
- `show_labels` (bool): Show text labels for points (default: True)
- `labels` (list): Custom labels for points (overrides default doc_ids/author names)
- `title` (str): Plot title (no title if None)
- `colors` (dict): Custom colors for authors `{author: color}` (auto-assigned if None)
- `marker_size` (int): Size of scatter points (default: 100)
- `fontsize` (int): Base font size (default: 12)
- `filename` (str): Save plot to file (displays only if None)
- `random_state` (int): Random seed for t-SNE/MDS reproducibility (default: 42)
- `show` (bool): If True, display plot. If False, return (fig, ax) for editing.

**Returns:** None if show=True, otherwise (fig, ax) tuple.

<br>

<h3 id="dendrogram">dendrogram()</h3>

```python
dendrogram(method='average', level='document', orientation='top', 
           figsize=(12, 8), labels=None, title=None, fontsize=10, 
           color_threshold=None, filename=None, show=True)
```

Visualize hierarchical clustering as a dendrogram.

**Parameters:**
- `method` (str): Linkage method
- `level` (str): `'document'` or `'author'`
- `orientation` (str): Dendrogram orientation (`'top'`, `'bottom'`, `'left'`, `'right'`)
- `figsize` (tuple): Figure size
- `labels` (list): Custom labels for leaves (overrides default doc_ids/author names)
- `title` (str): Plot title (no title if None)
- `fontsize` (int): Font size for labels
- `color_threshold` (float): Distance threshold for coloring. Links below get cluster colors, links above get uniform color. Default uses 0.7 * max distance. Set high to color more from bottom, low to color from top.
- `filename` (str): Save plot to file
- `show` (bool): If True, display plot. If False, return result dict.

**Returns:** None if show=True, otherwise dict with `'fig'`, `'ax'`, and scipy dendrogram data (`'ivl'`, `'leaves'`, `'color_list'`, etc.).

<br>

<h3 id="bootstrap_predict">bootstrap_predict()</h3>

```python
bootstrap_predict(text, n_iter=100, sample_ratio=0.8, distance=None)
```

Bootstrap analysis for prediction robustness. Resamples features n_iter times and computes prediction statistics to assess how robust the attribution is.

**Parameters:**
- `text` (list): List of tokens (the disputed text)
- `n_iter` (int): Number of bootstrap iterations (default: 100)
- `sample_ratio` (float): Fraction of features to use per iteration, 0.0-1.0 (default: 0.8)
- `distance` (str): Distance metric override (optional)

**Returns:** (dict) Dictionary containing:
- `'prediction'`: Most frequent prediction across iterations
- `'confidence'`: Proportion of iterations agreeing with top prediction
- `'distribution'`: Dict of author -> proportion of iterations
- `'distances'`: Dict of author -> (mean_distance, std_distance)
- `'n_iterations'`: Number of iterations performed

<br>

<h3 id="rolling_delta">rolling_delta()</h3>

```python
rolling_delta(text, reference=None, window=5000, step=1000, distance=None, 
              show=True, figsize=(12, 6), title=None, filename=None)
```

Rolling window analysis across a long text. Computes distance to a reference at each window position, useful for detecting authorship changes or style variation within a text.

**Parameters:**
- `text` (list): List of tokens (the long text to analyze)
- `reference` (str): Author name to compare against. If None, compares each window to the average representation of the entire text (self-comparison mode for detecting internal variation)
- `window` (int): Window size in tokens (default: 5000)
- `step` (int): Step size for sliding window (default: 1000)
- `distance` (str): Distance metric override (optional)
- `show` (bool): If True, display plot (default: True)
- `figsize` (tuple): Figure size (default: (12, 6))
- `title` (str): Plot title (optional)
- `filename` (str): Save plot to file (optional)

**Returns:** (DataFrame) Results with columns:
- `'position'`: Token position of window start
- `'distance'`: Distance to reference at that position

<br>

<h3 id="extract_mfw">extract_mfw()</h3>

```python
extract_mfw(ngram_counts, n=100)
```

Extract the n Most Frequent n-grams from a Counter.

**Parameters:**
- `ngram_counts` (Counter): A Counter object containing n-gram counts
- `n` (int): Number of most frequent n-grams to extract

**Returns:** (list) List of the n most common n-grams

<br>

<h3 id="burrows_delta">burrows_delta()</h3>

```python
burrows_delta(vec_a, vec_b)
```

Burrows' Delta distance: the mean absolute difference between z-score vectors. A classic stylometric measure. Note: This function is designed for z-score normalized vectors; for best results, use `transform='zscore'` when creating the Stylometry instance.

**Parameters:**
- `vec_a` (numpy.ndarray): First feature vector (ideally z-score normalized)
- `vec_b` (numpy.ndarray): Second feature vector (ideally z-score normalized)

**Returns:** (float) Distance value (lower = more similar)

<br>

```python
cosine_distance(vec_a, vec_b)
```

Cosine distance: 1 - cosine_similarity.

**Returns:** (float) Distance value (0 = identical, 2 = opposite)

<br>

```python
manhattan_distance(vec_a, vec_b)
```

Manhattan (L1) distance: sum of absolute differences.

**Returns:** (float) Distance value (lower = more similar)

<br>

```python
euclidean_distance(vec_a, vec_b)
```

Euclidean (L2) distance: square root of sum of squared differences.

**Returns:** (float) Distance value (lower = more similar)

<br>

```python
eder_delta(vec_a, vec_b)
```

Eder's Delta distance: a variation of Burrows' Delta with different weighting. Squares the differences and takes the square root of the mean, giving more weight to larger differences.

**Returns:** (float) Distance value (lower = more similar)

<br>

```python
get_relative_frequencies(tokens)
```

Compute relative frequencies for a list of tokens.

**Parameters:**
- `tokens` (list): List of tokens

**Returns:** (dict) Mapping of tokens to relative frequencies (count / total)

<br>

<h3 id="compute_yule_k">compute_yule_k()</h3>

```python
compute_yule_k(tokens)
```

Compute Yule's K characteristic for vocabulary richness. Higher values indicate less diverse vocabulary. Relatively independent of text length.

**Parameters:**
- `tokens` (list): List of tokens

**Returns:** (float) Yule's K value (typically between 50-200 for normal texts)

---

## Examples

**Authorship Attribution**

```python
from qhchina.analytics.stylometry import Stylometry

# Prepare corpus: dict mapping author names to their documents
# Each document is a list of tokens
corpus = {
    'author_a': [
        ['这', '是', '作者', 'A', '的', '第一篇', '文章', '...'],
        ['作者', 'A', '的', '另一篇', '文章', '...'],
    ],
    'author_b': [
        ['这', '是', '作者', 'B', '写', '的', '内容', '...'],
        ['作者', 'B', '的', '其他', '作品', '...'],
    ],
}

# Create and fit the model
stylo = Stylometry(n_features=100, distance='cosine', mode='centroid')
stylo.fit_transform(corpus)

# Predict authorship for an unknown text
unknown_text = ['他', '终于', '在', '无物', '之', '阵', '中', '老衰', '，', ...]
results = stylo.predict(unknown_text)  # Returns top 1 author
for author, distance in results:
    print(f"{author}: {distance:.4f}")

# Get top 3 most similar authors
results = stylo.predict(unknown_text, k=3)
for author, distance in results:
    print(f"{author}: {distance:.4f}")

# Get predicted author directly
predicted = stylo.predict_author(unknown_text)
print(f"Predicted author: {predicted}")
```

**Finding Similar Documents**

```python
# Find documents most similar to a specific document (returns similarity by default)
similar = stylo.most_similar('author_a_1', k=5)
for doc_id, sim in similar:
    print(f"{doc_id}: {sim:.4f}")  # higher = more similar

# Find documents similar to new text (without adding it to the corpus)
similar = stylo.most_similar(['这', '是', '新', '文本', '...'], k=3)
for doc_id, sim in similar:
    print(f"{doc_id}: {sim:.4f}")

# Compare two specific documents
sim = stylo.similarity('author_a_1', 'author_b_1')  # higher = more similar
dist = stylo.distance('author_a_1', 'author_b_1')   # lower = more similar
print(f"Similarity: {sim:.4f}, Distance: {dist:.4f}")
```

**Analyzing Author Profiles**

```python
# Get feature profile for an author
profile = stylo.get_author_profile('author_a')
print("Top distinctive features for author_a:")
print(profile.head(10))

# Compare features across all authors
comparison = stylo.get_feature_comparison()
print("Most variable features across authors:")
print(comparison.head(10))
```

**Document Clustering (Unsupervised)**

```python
from qhchina.analytics.stylometry import Stylometry

# Documents without author labels
documents = [
    ['文档', '一', '的', '内容', '...'],
    ['文档', '二', '的', '内容', '...'],
    ['文档', '三', '的', '内容', '...'],
]

# Create model and fit on documents (unsupervised mode)
stylo = Stylometry(n_features=50, distance='burrows_delta')
stylo.fit_transform(documents)  # All docs get 'unk' author, IDs: unk_1, unk_2, unk_3

# Visualize documents in 2D space
stylo.plot(
    method='pca',
    title='Document Clustering',
    filename='clustering.png'
)

# Create dendrogram
stylo.dendrogram(
    method='average',
    filename='dendrogram.png'
)

# Find similar documents
similar = stylo.most_similar('unk_1')
```

**Supervised Visualization**

```python
# After fitting on labeled corpus
stylo.fit_transform(corpus)

# Plot at document level (colored by author)
stylo.plot(
    method='pca',
    level='document',
    title='Documents by Author',
    filename='docs_pca.png'
)

# Plot at author level (one point per author)
stylo.plot(
    method='mds',
    level='author',
    title='Author Similarity',
    filename='authors_mds.png'
)

# Hierarchical clustering of documents
stylo.dendrogram(
    method='ward',
    level='document',
    filename='doc_dendrogram.png'
)
```

**Instance Mode with k-NN**

```python
# Use instance mode for k-nearest neighbor attribution
stylo = Stylometry(n_features=100, distance='cosine', mode='instance')
stylo.fit_transform(corpus)

# Find the 5 nearest training documents to a disputed text
results = stylo.predict(unknown_text, k=5)
for author, distance in results:
    print(f"{author}: {distance:.4f}")

# Get predicted author via majority vote among 5 nearest neighbors
predicted = stylo.predict_author(unknown_text, k=5)
print(f"Predicted author (majority vote): {predicted}")
```

**Corpus Balance Warning**

The module automatically warns when author corpus sizes are highly imbalanced (3x difference or more), as this can skew the Most Frequent Words calculation toward the larger corpus:

```python
# This will trigger a warning if corpus sizes are imbalanced
corpus = {
    'prolific_author': [doc1, doc2, doc3, ..., doc100],  # Many documents
    'rare_author': [doc1, doc2],  # Few documents
}
stylo.fit_transform(corpus)
# UserWarning: Imbalanced corpus: 'prolific_author' has X tokens while 
# 'rare_author' has only Y tokens (Z.Zx difference)...
```

Consider balancing text sizes across authors for more reliable results.

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

**Example:**

```python
from qhchina.analytics.stylometry import compare_corpora

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
```
