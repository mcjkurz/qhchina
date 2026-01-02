---
layout: docs_with_sidebar
title: Stylometry
permalink: /pkg_docs/stylometry/
---

# Stylometry

The `qhchina.analytics.stylometry` module provides tools for authorship attribution and document clustering using statistical analysis of writing style.

> **Note:** This module is inspired by the R package [stylo](https://github.com/computationalstylistics/stylo), a much more comprehensive implementation for computational stylistics.

```python
from qhchina.analytics.stylometry import Stylometry

stylo = Stylometry(n_features=100, distance='cosine')
stylo.fit_transform({'author_a': [tokens_a1, tokens_a2], 'author_b': [tokens_b1, tokens_b2]})

# Analyze the transformed data
predicted = stylo.predict_author(unknown_text)  # Predict authorship
similar = stylo.most_similar('author_a_1')      # Find similar documents
dist = stylo.distance('author_a_1', 'author_b_1')  # Compare two documents
```

---

## Stylometry Class

The main class for stylometric analysis supports both supervised (with labeled training data) and unsupervised (clustering) approaches.

### Workflow

1. Create a `Stylometry` instance with desired parameters
2. Call `fit_transform()` with your corpus (dict or list of tokenized documents)
3. Analyze with: `plot()`, `dendrogram()`, `most_similar()`, `distance()`, `predict()`

### Initialization

```python
Stylometry(n_features=100, distance='cosine', mode='centroid')
```

**Parameters:**
- `n_features` (int): Number of most frequent words to use as features (default: 100)
- `distance` (str): Distance metric for comparison
  - `'cosine'`: Cosine distance (default)
  - `'burrows_delta'`: Burrows' Delta (classic stylometric measure)
  - `'manhattan'`: Manhattan (L1) distance
- `mode` (str): Attribution mode for supervised analysis
  - `'centroid'`: Aggregate all author texts into one profile per author (default)
  - `'instance'`: Keep individual texts separate (k-NN style)

---

### Core Method

```python
fit_transform(corpus, labels=None)
```

Fit the model on a corpus and transform documents to z-score vectors.

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

Document IDs are auto-generated as `{label}_{n}` (e.g., `author_a_1`, `author_a_2`, `unk_1`).

**Returns:** self (for method chaining)

<br>

```python
transform(tokens)
```

Transform a tokenized text to a z-score vector using fitted features. This allows you to transform new documents without modifying the model's internal state.

**Parameters:**
- `tokens` (list): List of tokens (a tokenized document)

**Returns:** (numpy.ndarray) Z-score vector of shape (n_features,)

<br>

---

### Prediction Methods

```python
predict(text, k=1)
```

Predict the most likely author for a tokenized text.

**Parameters:**
- `text` (list): List of tokens (the disputed text)
- `k` (int): Number of nearest neighbors to consider (only used in `'instance'` mode)

> **Note:** In `'centroid'` mode, the `k` parameter is ignored and a warning is issued if `k != 1`. All author centroids are compared and returned. Use `mode='instance'` for k-NN behavior.

**Returns:** (list) List of (author, distance) tuples sorted by distance ascending
- In `'centroid'` mode: returns distances to each author centroid
- In `'instance'` mode: returns the k nearest documents with their author labels

<br>

```python
predict_author(text, k=1)
```

Convenience method to get just the predicted author name.

**Parameters:**
- `text` (list): List of tokens
- `k` (int): Number of nearest neighbors for majority vote (only in `'instance'` mode)

**Returns:** (str) Predicted author name

In `'instance'` mode with `k > 1`, returns the majority vote among the k nearest neighbors.

<br>

---

### Distance Methods

```python
most_similar(query, k=None)
```

Find the most similar documents to a query.

**Parameters:**
- `query`: Either:
  - A document ID (str): find documents similar to this document
  - A list of tokens: transform and find similar documents
- `k` (int): Number of results to return. If None, returns all documents.

**Returns:** (list) List of (doc_id, distance) tuples sorted by distance ascending (most similar first). If query is a doc_id, that document is excluded from results.

<br>

```python
distance(a, b)
```

Compute the distance between two documents or texts.

**Parameters:**
- `a`: First document - either a doc_id (str) or list of tokens
- `b`: Second document - either a doc_id (str) or list of tokens

**Returns:** (float) Distance value (lower = more similar)

<br>

---

### Analysis Methods

```python
get_author_profile(author)
```

Get the z-score normalized feature values for a specific author.

**Parameters:**
- `author` (str): Author name

**Returns:** (DataFrame) DataFrame with 'feature' and 'zscore' columns, sorted by z-score descending

<br>

```python
get_feature_comparison()
```

Get a comparison table of feature z-scores across all fitted authors.

**Returns:** (DataFrame) DataFrame with one column per author plus a 'variance' column

<br>

```python
distance_matrix(level='document')
```

Compute pairwise distance matrix from fitted data.

**Parameters:**
- `level` (str): `'document'` for individual documents, `'author'` for author profiles

**Returns:** (tuple) (distance_matrix, labels)

<br>

```python
hierarchical_clustering(method='average', level='document')
```

Perform hierarchical clustering on fitted data.

**Parameters:**
- `method` (str): Linkage method (`'single'`, `'complete'`, `'average'`, `'weighted'`, `'ward'`)
- `level` (str): `'document'` or `'author'`

**Returns:** (tuple) (linkage_matrix, labels) for scipy dendrogram

<br>

---

### Visualization Methods

```python
plot(method='pca', level='document', figsize=(10, 8), show_labels=True, 
     title=None, colors=None, marker_size=100, fontsize=12, 
     filename=None, random_state=42)
```

Create a 2D scatter plot of documents or authors.

**Parameters:**
- `method` (str): Dimensionality reduction method (`'pca'`, `'tsne'`, or `'mds'`)
- `level` (str): `'document'` for individual documents, `'author'` for author profiles
- `figsize` (tuple): Figure size as (width, height)
- `show_labels` (bool): Show text labels for points (default: True)
- `title` (str): Plot title (auto-generated if None)
- `colors` (dict): Custom colors for authors `{author: color}` (auto-assigned if None)
- `marker_size` (int): Size of scatter points (default: 100)
- `fontsize` (int): Base font size (default: 12)
- `filename` (str): Save plot to file (displays only if None)
- `random_state` (int): Random seed for t-SNE/MDS reproducibility (default: 42)

<br>

```python
dendrogram(method='average', level='document', orientation='top', 
           figsize=(12, 8), fontsize=10, filename=None)
```

Visualize hierarchical clustering as a dendrogram.

**Parameters:**
- `method` (str): Linkage method
- `level` (str): `'document'` or `'author'`
- `orientation` (str): Dendrogram orientation (`'top'`, `'bottom'`, `'left'`, `'right'`)
- `figsize` (tuple): Figure size
- `fontsize` (int): Font size for labels
- `filename` (str): Save plot to file

<br>

---

## Utility Functions

```python
extract_mfw(texts, n=100)
```

Extract the n Most Frequent Words from a collection of tokenized texts.

**Parameters:**
- `texts` (list): List of tokenized documents
- `n` (int): Number of most frequent words to extract

**Returns:** (list) List of the n most common words

<br>

```python
burrows_delta(vec_a, vec_b)
```

Burrows' Delta distance: the mean absolute difference between z-score vectors.

**Returns:** (float) Distance value (lower = more similar)

<br>

```python
get_relative_frequencies(tokens)
```

Compute relative word frequencies for a tokenized text.

**Parameters:**
- `tokens` (list): List of tokens

**Returns:** (dict) Mapping of words to relative frequencies

<br>

---

## Examples

### Authorship Attribution

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
unknown_text = ['这', '篇', '文章', '的', '作者', '是', '谁', '...']
results = stylo.predict(unknown_text)
for author, distance in results:
    print(f"{author}: {distance:.4f}")

# Get predicted author directly
predicted = stylo.predict_author(unknown_text)
print(f"Predicted author: {predicted}")
```

### Finding Similar Documents

```python
# Find documents most similar to a specific document
similar = stylo.most_similar('author_a_1', k=5)
for doc_id, distance in similar:
    print(f"{doc_id}: {distance:.4f}")

# Find documents similar to new text (without adding it to the corpus)
similar = stylo.most_similar(['这', '是', '新', '文本', '...'], k=3)
for doc_id, distance in similar:
    print(f"{doc_id}: {distance:.4f}")

# Compare two specific documents
dist = stylo.distance('author_a_1', 'author_b_1')
print(f"Distance: {dist:.4f}")

# Compare a document to new text
dist = stylo.distance('author_a_1', ['新', '文本', '...'])
print(f"Distance: {dist:.4f}")
```

### Analyzing Author Profiles

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

### Document Clustering (Unsupervised)

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

### Supervised Visualization

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

### Instance Mode with k-NN

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

### Corpus Balance Warning

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
