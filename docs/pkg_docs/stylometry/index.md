---
layout: docs_with_sidebar
title: Stylometry
permalink: /pkg_docs/stylometry/
functions:
  - name: Stylometry
    anchor: stylometry
  - name: Stylometry.bootstrap_predict()
    anchor: stylometry-bootstrap_predict
  - name: Stylometry.dendrogram()
    anchor: stylometry-dendrogram
  - name: Stylometry.distance()
    anchor: stylometry-distance
  - name: Stylometry.distance_matrix()
    anchor: stylometry-distance_matrix
  - name: Stylometry.fit_transform()
    anchor: stylometry-fit_transform
  - name: Stylometry.get_author_profile()
    anchor: stylometry-get_author_profile
  - name: Stylometry.get_feature_comparison()
    anchor: stylometry-get_feature_comparison
  - name: Stylometry.hierarchical_clustering()
    anchor: stylometry-hierarchical_clustering
  - name: Stylometry.most_similar()
    anchor: stylometry-most_similar
  - name: Stylometry.plot()
    anchor: stylometry-plot
  - name: Stylometry.predict()
    anchor: stylometry-predict
  - name: Stylometry.predict_author()
    anchor: stylometry-predict_author
  - name: Stylometry.predict_confidence()
    anchor: stylometry-predict_confidence
  - name: Stylometry.rolling_delta()
    anchor: stylometry-rolling_delta
  - name: Stylometry.similarity()
    anchor: stylometry-similarity
  - name: Stylometry.transform()
    anchor: stylometry-transform
  - name: Stylometry.vocabulary_stats()
    anchor: stylometry-vocabulary_stats
  - name: compare_corpora()
    anchor: compare_corpora
  - name: extract_mfw()
    anchor: extract_mfw
  - name: burrows_delta()
    anchor: burrows_delta
  - name: manhattan_distance()
    anchor: manhattan_distance
  - name: euclidean_distance()
    anchor: euclidean_distance
  - name: eder_delta()
    anchor: eder_delta
  - name: get_relative_frequencies()
    anchor: get_relative_frequencies
  - name: compute_yule_k()
    anchor: compute_yule_k
has_examples: True
import_from: qhchina.analytics.stylometry
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

## API Reference

<!-- API-START -->

<h3 id="stylometry">Stylometry</h3>

```python
Stylometry(
    n_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 1),
    transform: str = 'zscore',
    distance: str = 'cosine',
    classifier: str = 'delta',
    cull: Optional[float] = None,
    chunk_size: Optional[int] = None,
    mode: str = 'centroid'
)
```

Stylometry for authorship attribution and document clustering.

Implements classic and modern stylometric methods for analyzing writing style,
comparing authors, and attributing disputed texts. Inspired by the R package
'stylo' but designed for Chinese text analysis.

**Parameters:**
- `n_features` (int): Number of most frequent n-grams to use as features (default: 100).
  Higher values capture more stylistic variation but may include noise.
- `ngram_range` (tuple): Range of n-gram sizes as (min_n, max_n). Default (1, 1) = unigrams.
  Use (1, 2) for unigrams + bigrams, (2, 2) for bigrams only.
- `transform` (str): Feature transformation method:
  - 'zscore': Z-score normalization (default, recommended for Delta methods)
  - 'tfidf': TF-IDF weighting
- `distance` (str): Distance metric for comparing documents:
  - 'cosine': Cosine distance (default)
  - 'burrows_delta': Classic Burrows' Delta
  - 'manhattan': Manhattan/L1 distance
  - 'euclidean': Euclidean/L2 distance
  - 'eder_delta': Eder's Delta variant
- `classifier` (str): Classification method for authorship attribution:
  - 'delta': Delta-based nearest neighbor (default)
  - 'svm': Support Vector Machine
- `cull` (float): Minimum document frequency ratio (0.0-1.0). N-grams appearing in
  fewer than cull*100% of documents are removed. Helps filter rare words.
  Default: None (no culling).
- `chunk_size` (int): If set, split documents into chunks of this many tokens.
  Useful for comparing texts of similar length.
- `mode` (str): Attribution mode for delta classifier:
  - 'centroid': Compare to author centroids (averaged profiles)
  - 'instance': Compare to individual text instances

**Example:**
```python
from qhchina.analytics.stylometry import Stylometry

# Prepare corpus: dict mapping author names to lists of tokenized documents
corpus = {
...     '鲁迅': [tokens_luxun_1, tokens_luxun_2],
...     '茅盾': [tokens_maodun_1, tokens_maodun_2]
... }

# Create and fit stylometry model
stylo = Stylometry(n_features=100, ngram_range=(1, 2), cull=0.2)
stylo.fit_transform(corpus)

# Visualize results
stylo.plot()  # PCA/MDS scatter plot
stylo.dendrogram()  # Hierarchical clustering

# Attribute disputed text
author, confidence = stylo.predict(disputed_tokens)
```

<h4 id="stylometry-bootstrap_predict">Stylometry.bootstrap_predict()</h4>

```python
bootstrap_predict(text: List[str], n_iter: int = 100, sample_ratio: float = 0.8, distance: Optional[str] = None, seed: Optional[int] = None)
```

Bootstrap analysis for prediction robustness.

Resamples features n_iter times and computes prediction statistics
to assess how robust the attribution is.

**Parameters:**
- `text`: List of tokens (the disputed text)
- `n_iter`: Number of bootstrap iterations
- `sample_ratio`: Fraction of features to use per iteration (0.0-1.0)
- `distance`: Distance metric override
- `seed`: Random seed for reproducibility. If None, results will vary
  between calls.

**Returns:**
Dict with:
- 'prediction': Most frequent prediction across iterations
- 'confidence': Proportion of iterations agreeing with top prediction
- 'distribution': Dict of author -> proportion of iterations
- 'distances': Dict of author -> (mean_distance, std_distance)
- 'n_iterations': Number of iterations performed

<h4 id="stylometry-dendrogram">Stylometry.dendrogram()</h4>

```python
dendrogram(method: str = 'average', level: str = 'document', orientation: str = 'top', figsize: Tuple[int, int] = (12, 8), labels: Optional[List[str]] = None, title: Optional[str] = None, fontsize: int = 10, color_threshold: Optional[float] = None, filename: Optional[str] = None, show: bool = True, distance: Optional[str] = None)
```

Visualize hierarchical clustering as a dendrogram.

**Parameters:**
- `method`: Linkage method
- `level`: 'document' or 'author'
- `orientation`: 'top', 'bottom', 'left', or 'right'
- `figsize`: Figure size
- `labels`: Custom labels for leaves
- `title`: Plot title
- `fontsize`: Font size for labels
- `color_threshold`: Distance threshold for coloring
- `filename`: If provided, save figure to this path
- `show`: If True, display plot. If False, return result dict.
- `distance`: Distance metric override.

**Returns:**
None if show=True, otherwise dict with 'fig', 'ax', and dendrogram data.

<h4 id="stylometry-distance">Stylometry.distance()</h4>

```python
distance(a: Union[str, List[str]], b: Union[str, List[str]], distance: Optional[str] = None)
```

Compute the distance between two documents. Lower = more similar.

<h4 id="stylometry-distance_matrix">Stylometry.distance_matrix()</h4>

```python
distance_matrix(level: str = 'document', distance: Optional[str] = None)
```

Compute pairwise distance matrix from fitted data.

**Parameters:**
- `level`: 'document' for individual documents, 'author' for author profiles
- `distance`: Distance metric override.

**Returns:**
(distance_matrix, labels)

<h4 id="stylometry-fit_transform">Stylometry.fit_transform()</h4>

```python
fit_transform(corpus: Union[Dict[str, List[List[str]]], List[List[str]]], labels: Optional[List[str]] = None)
```

Fit the model on a corpus and transform documents to feature vectors.

**Parameters:**
- `corpus`: Either:
  - Dict mapping author names to their documents (supervised):
    {'AuthorA': [[tok1, tok2, ...], [tok1, ...]], 'AuthorB': [...]}
  - List of tokenized documents (unsupervised):
    [[tok1, tok2, ...], [tok1, ...], ...]
- `labels`: Optional list of labels for list input. Documents sharing
  the same label are grouped together.

<h4 id="stylometry-get_author_profile">Stylometry.get_author_profile()</h4>

```python
get_author_profile(author: str)
```

Get the feature values for a specific author.

Returns a DataFrame with 'feature' and 'value' columns, sorted by value descending.

<h4 id="stylometry-get_feature_comparison">Stylometry.get_feature_comparison()</h4>

```python
get_feature_comparison()
```

Get a comparison table of feature values across all fitted authors.

Returns a DataFrame with one column per author plus a 'variance' column.

<h4 id="stylometry-hierarchical_clustering">Stylometry.hierarchical_clustering()</h4>

```python
hierarchical_clustering(method: str = 'average', level: str = 'document', distance: Optional[str] = None)
```

Perform hierarchical clustering on fitted data.

**Parameters:**
- `method`: Linkage method - 'single', 'complete', 'average', 'weighted', or 'ward'
- `level`: 'document' or 'author'
- `distance`: Distance metric override.

**Returns:**
(linkage_matrix, labels)

<h4 id="stylometry-most_similar">Stylometry.most_similar()</h4>

```python
most_similar(query: Union[str, List[str]], k: Optional[int] = None, return_distance: bool = False, distance: Optional[str] = None)
```

Find the most similar documents to a query.

**Parameters:**
- `query`: Document ID (str) or list of tokens.
- `k`: Number of results to return. If None, returns all.
- `return_distance`: If False, returns similarity. If True, returns distance.
- `distance`: Distance metric override.

**Returns:**
List of (doc_id, value) tuples sorted by similarity (most similar first).

<h4 id="stylometry-plot">Stylometry.plot()</h4>

```python
plot(method: str = 'pca', level: str = 'document', figsize: Tuple[int, int] = (10, 8), show_labels: bool = True, labels: Optional[List[str]] = None, title: Optional[str] = None, colors: Optional[Dict[str, str]] = None, marker_size: int = 100, fontsize: int = 12, filename: Optional[str] = None, random_state: int = 42, show: bool = True)
```

Create a 2D scatter plot of documents or authors.

**Parameters:**
- `method`: Dimensionality reduction - 'pca', 'tsne', or 'mds'
- `level`: 'document' for individual documents, 'author' for author profiles
- `figsize`: Figure size as (width, height)
- `show_labels`: Whether to show text labels on points
- `labels`: Custom labels for points
- `title`: Custom title
- `colors`: Dict mapping author names to colors
- `marker_size`: Size of scatter points
- `fontsize`: Base font size
- `filename`: If provided, save figure to this path
- `random_state`: Random seed for t-SNE/MDS
- `show`: If True, display plot. If False, return (fig, ax).

**Returns:**
None if show=True, otherwise (fig, ax) tuple.

<h4 id="stylometry-predict">Stylometry.predict()</h4>

```python
predict(text: List[str], k: int = 1, distance: Optional[str] = None, classifier: Optional[str] = None)
```

Predict the most likely author for a tokenized text.

**Parameters:**
- `text`: List of tokens (the disputed text)
- `k`: Number of top results to return.
- `distance`: Distance metric override (for delta classifier).
- `classifier`: Classifier override ('delta' or 'svm').

**Returns:**
List of (author, score) tuples.
- For 'delta': score is distance (lower = more similar)
- For 'svm': score is probability (higher = more likely)

<h4 id="stylometry-predict_author">Stylometry.predict_author()</h4>

```python
predict_author(text: List[str], k: int = 1, distance: Optional[str] = None, classifier: Optional[str] = None)
```

Convenience method to get just the predicted author name.

**Parameters:**
- `text`: List of tokens (the disputed text)
- `k`: For 'instance' mode only: number of nearest neighbors for majority voting.
  In 'centroid' mode, this parameter is ignored.
- `distance`: Distance metric override (for delta classifier).
- `classifier`: Classifier override ('delta' or 'svm').

**Returns:**
Predicted author name (str).

<h4 id="stylometry-predict_confidence">Stylometry.predict_confidence()</h4>

```python
predict_confidence(text: List[str], k: int = 1, classifier: Optional[str] = None)
```

Predict with unified confidence scores (higher = more likely).

Abstracts away the difference between delta (distance) and SVM (probability).

**Returns:**
List of (author, confidence) tuples where confidence is 0-1, higher = more likely.

<h4 id="stylometry-rolling_delta">Stylometry.rolling_delta()</h4>

```python
rolling_delta(text: List[str], reference: Optional[str] = None, window: int = 5000, step: int = 1000, distance: Optional[str] = None, show: bool = True, figsize: Tuple[int, int] = (12, 6), title: Optional[str] = None, filename: Optional[str] = None)
```

Rolling window analysis across a long text.

Computes distance to a reference at each window position,
useful for detecting authorship changes or style variation within a text.

**Parameters:**
- `text`: List of tokens (the long text to analyze)
- `reference`: Author name to compare against. If None, compares each
  window to the average representation of the entire text
  (self-comparison mode for detecting internal variation).
- `window`: Window size in tokens
- `step`: Step size for sliding window
- `distance`: Distance metric override
- `show`: If True, display plot
- `figsize`: Figure size for plot
- `title`: Plot title
- `filename`: If provided, save figure to this path

**Returns:**
DataFrame with columns:
- 'position': Starting token position of window
- 'distance': Distance to reference
- 'end_position': Ending token position of window

<h4 id="stylometry-similarity">Stylometry.similarity()</h4>

```python
similarity(a: Union[str, List[str]], b: Union[str, List[str]], distance: Optional[str] = None)
```

Compute the similarity between two documents. Higher = more similar.

<h4 id="stylometry-transform">Stylometry.transform()</h4>

```python
transform(tokens: List[str], warn_oov: bool = True)
```

Transform a tokenized text to a feature vector using fitted features.

**Parameters:**
- `tokens`: List of tokens (a tokenized document)
- `warn_oov`: If True (default), warn when the text has low overlap with
  the trained features (less than 50% of n-grams recognized).

**Returns:**
Feature vector (numpy array)

<h4 id="stylometry-vocabulary_stats">Stylometry.vocabulary_stats()</h4>

```python
vocabulary_stats()
```

Get vocabulary richness statistics for all fitted documents.

**Returns:**
DataFrame with columns: doc_id, author, yule_k, token_count, type_count

<br>

<h3 id="compare_corpora">compare_corpora()</h3>

```python
compare_corpora(
    corpusA: Union[List[str], List[List[str]]],
    corpusB: Union[List[str], List[List[str]]],
    method: str = 'fisher',
    filters: Optional[Dict] = None,
    correction: Optional[str] = None,
    as_dataframe: bool = True
)
```

Compare two corpora to identify statistically significant differences in word usage.

**Parameters:**
- `corpusA`: Either a flat list of tokens or a list of sentences (each sentence 
  being a list of tokens).
- `corpusB`: Either a flat list of tokens or a list of sentences (each sentence 
  being a list of tokens).
- `method` (str): 'fisher' for Fisher's exact test or 'chi2' or 'chi2_corrected' 
  for the chi-square test. All tests use two-sided alternatives.
- `filters` (dict): Dictionary of filters to apply to results.
  All filters (except ``max_adjusted_p``) are applied BEFORE multiple testing 
  correction, defining the "family" of hypotheses being tested. This maximizes 
  statistical power by not correcting for words that were never of interest.
  
  Available filters:
  
  - 'min_count': int or tuple - Minimum count threshold(s) for a word to be 
    included (can be a single int for both corpora or tuple (min_countA, 
    min_countB)). Default is 0.
  - 'stopwords': list - Words to exclude from results.
  - 'min_word_length': int - Minimum character length for words.
  - 'max_p': float - Maximum raw p-value threshold.
  - 'max_adjusted_p': float - Maximum adjusted p-value (requires correction,
    applied after correction is computed).
    
- `correction` (str): Multiple testing correction method. When set,
  an ``adjusted_p_value`` column is added to the results. The correction
  is applied AFTER all other filters, so only words that pass those
  filters count toward the number of tests.
  
  - 'bonferroni': Bonferroni correction (conservative, controls family-wise 
    error rate).
  - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate,
    recommended for corpus comparison).
  - None: No correction (default).
- `as_dataframe` (bool): Whether to return a pandas DataFrame.

**Returns:**
If as_dataframe is True: pandas.DataFrame containing information about each 
    word's frequency in both corpora, the p-value, and the ratio of relative 
    frequencies.
If as_dataframe is False: List[dict] where each dict contains information 
    about a word's frequency in both corpora, the p-value, and the ratio of 
    relative frequencies.

<br>

<h3 id="extract_mfw">extract_mfw()</h3>

```python
extract_mfw(ngram_counts: collections.Counter, n: int = 100)
```

Extract the Most Frequent Words (MFW) from a frequency counter.

**Parameters:**
- `ngram_counts` (Counter): A Counter object with n-gram/word frequencies.
- `n` (int): Number of most frequent items to return (default: 100).

**Returns:**
(list) The n most common n-grams/words, ordered by frequency.

**Example:**
```python
from collections import Counter
from qhchina.analytics.stylometry import extract_mfw
counts = Counter(['的', '是', '了', '的', '我', '的'])
mfw = extract_mfw(counts, n=2)
print(mfw)
['的', '是']
```

<br>

<h3 id="burrows_delta">burrows_delta()</h3>

```python
burrows_delta(vec_a: numpy.ndarray, vec_b: numpy.ndarray)
```

Compute Burrows' Delta distance between two feature vectors.

A classic stylometric measure for authorship attribution. Calculates the
mean absolute difference between z-score normalized frequency vectors.
Lower values indicate more similar writing styles.

**Parameters:**
- `vec_a` (np.ndarray): First z-score feature vector.
- `vec_b` (np.ndarray): Second z-score feature vector.

**Returns:**
(float) Burrows' Delta distance (lower = more similar).

<br>

<h3 id="manhattan_distance">manhattan_distance()</h3>

```python
manhattan_distance(vec_a: numpy.ndarray, vec_b: numpy.ndarray)
```

Compute Manhattan (L1) distance between two vectors.

**Parameters:**
- `vec_a` (np.ndarray): First feature vector.
- `vec_b` (np.ndarray): Second feature vector.

**Returns:**
(float) Sum of absolute differences between corresponding elements.

<br>

<h3 id="euclidean_distance">euclidean_distance()</h3>

```python
euclidean_distance(vec_a: numpy.ndarray, vec_b: numpy.ndarray)
```

Compute Euclidean (L2) distance between two vectors.

**Parameters:**
- `vec_a` (np.ndarray): First feature vector.
- `vec_b` (np.ndarray): Second feature vector.

**Returns:**
(float) Square root of sum of squared differences.

<br>

<h3 id="eder_delta">eder_delta()</h3>

```python
eder_delta(vec_a: numpy.ndarray, vec_b: numpy.ndarray)
```

Eder's Delta distance: a variation of Burrows' Delta with different weighting.

Eder's Delta squares the differences and takes the square root of the mean,
giving more weight to larger differences. It also normalizes by vector length.

Formula: $\Delta_E = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (a_i - b_i)^2}$

Reference: Eder, M. (2013). "Mind your corpus: systematic errors in authorship attribution"

<br>

<h3 id="get_relative_frequencies">get_relative_frequencies()</h3>

```python
get_relative_frequencies(items: List[str])
```

Compute relative frequencies for a list of items (tokens or n-grams).

**Returns:**
Dict mapping each unique item to its relative frequency (count / total)

<br>

<h3 id="compute_yule_k">compute_yule_k()</h3>

```python
compute_yule_k(tokens: List[str])
```

Compute Yule's K characteristic for vocabulary richness.

Yule's K is a measure of lexical diversity that is relatively independent
of text length. Higher values indicate less diverse vocabulary.

Formula: $K = 10^4 \cdot \frac{M_2 - M_1}{M_1^2}$

where $M_1$ = total tokens, $M_2 = \sum_r r^2 \cdot V_r$ (sum of frequency squared 
times count of words with that frequency)

**Parameters:**
- `tokens`: List of tokens

**Returns:**
Yule's K value (typically between 50-200 for normal texts)

<br>

<!-- API-END -->

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
