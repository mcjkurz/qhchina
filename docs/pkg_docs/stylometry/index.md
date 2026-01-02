---
layout: docs_with_sidebar
title: Stylometry
permalink: /pkg_docs/stylometry/
---

# Stylometry

The `qhchina.analytics.stylometry` module provides tools for authorship attribution and document clustering using statistical analysis of writing style.

## Stylometry Class

The main class for stylometric analysis supports both supervised (with labeled training data) and unsupervised (clustering) approaches.

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

### Supervised Methods

```python
fit(corpus)
```

Fit the model on a labeled corpus for supervised authorship analysis.

**Parameters:**
- `corpus` (dict): Dictionary mapping author names to their documents. Each document is a list of tokens.
  ```python
  {'author_a': [['word1', 'word2', ...], ['more', 'tokens', ...]], 
   'author_b': [['other', 'words', ...]], ...}
  ```

**Returns:** self (for method chaining)

<br>

```python
predict(text, k=1)
```

Predict the most likely author for a tokenized text.

**Parameters:**
- `text` (list): List of tokens (the disputed text)
- `k` (int): Number of nearest neighbors (only used in 'instance' mode)

**Returns:** (list) List of (author, distance) tuples sorted by distance ascending

<br>

```python
predict_author(text, k=1)
```

Convenience method to get just the predicted author name.

**Parameters:**
- `text` (list): List of tokens
- `k` (int): Number of nearest neighbors for majority vote in 'instance' mode

**Returns:** (str) Predicted author name

<br>

### Unsupervised Methods

```python
transform(documents, labels=None)
```

Transform documents to z-score vectors for clustering/visualization without prior fitting.

**Parameters:**
- `documents` (list): List of tokenized documents
- `labels` (list): Optional labels for each document (defaults to Doc_1, Doc_2, ...)

**Returns:** (tuple) (z_score_vectors, labels)

<br>

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
hierarchical_clustering(documents=None, labels=None, method='average', level='document')
```

Perform hierarchical clustering.

**Parameters:**
- `documents` (list): Optional documents for unsupervised mode
- `labels` (list): Optional labels for documents
- `method` (str): Linkage method (`'single'`, `'complete'`, `'average'`, `'weighted'`, `'ward'`)
- `level` (str): `'document'` or `'author'` (only for supervised mode)

**Returns:** (tuple) (linkage_matrix, labels) for scipy dendrogram

<br>

### Visualization Methods

```python
plot(documents=None, labels=None, method='pca', level='document', figsize=(10, 8),
     show_labels=True, title=None, colors=None, marker_size=100, fontsize=12,
     filename=None, random_state=42)
```

Create a 2D scatter plot of documents or authors.

**Parameters:**
- `documents` (list): Optional documents for unsupervised mode
- `labels` (list): Optional labels for documents
- `method` (str): Dimensionality reduction method (`'pca'`, `'tsne'`, or `'mds'`)
- `level` (str): `'document'` or `'author'` (only for supervised mode)
- `figsize` (tuple): Figure size
- `show_labels` (bool): Show text labels for points
- `title` (str): Plot title
- `colors` (dict): Custom colors for authors `{author: color}`
- `marker_size` (int): Size of scatter points
- `fontsize` (int): Base font size
- `filename` (str): Save plot to file

<br>

```python
dendrogram(documents=None, labels=None, method='average', level='document',
           orientation='top', figsize=(12, 8), fontsize=10, filename=None)
```

Visualize hierarchical clustering as a dendrogram.

**Parameters:**
- `documents` (list): Optional documents for unsupervised mode
- `labels` (list): Optional labels for documents
- `method` (str): Linkage method
- `level` (str): `'document'` or `'author'` (only for supervised mode)
- `orientation` (str): Dendrogram orientation (`'top'`, `'bottom'`, `'left'`, `'right'`)
- `figsize` (tuple): Figure size
- `fontsize` (int): Font size for labels
- `filename` (str): Save plot to file

<br>

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
stylo.fit(corpus)

# Predict authorship for an unknown text
unknown_text = ['这', '篇', '文章', '的', '作者', '是', '谁', '...']
results = stylo.predict(unknown_text)
for author, distance in results:
    print(f"{author}: {distance:.4f}")

# Get predicted author directly
predicted = stylo.predict_author(unknown_text)
print(f"Predicted author: {predicted}")
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
labels = ['Doc 1', 'Doc 2', 'Doc 3']

# Create model for unsupervised analysis
stylo = Stylometry(n_features=50, distance='burrows_delta')

# Visualize documents in 2D space
stylo.plot(
    documents=documents,
    labels=labels,
    method='pca',
    title='Document Clustering',
    filename='clustering.png'
)

# Create dendrogram
stylo.dendrogram(
    documents=documents,
    labels=labels,
    method='average',
    filename='dendrogram.png'
)
```

### Supervised Visualization

```python
# After fitting on labeled corpus
stylo.fit(corpus)

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

