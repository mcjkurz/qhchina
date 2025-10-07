# qhChina Analytics Module

This module provides utilities for text analytics and topic modeling.

## Features

- **Collocation Analysis**: Find significant word co-occurrences in text
- **Corpus Comparison**: Statistically compare different corpora
- **Vector Operations**: Project and manipulate word embeddings
- **Topic Modeling**: Fast LDA implementation with Cython acceleration

## Installation

```bash
 install qhchina
```

For maximum performance with topic modeling, also install Cython:

```bash
pip install cython
```

After installation, you might want to compile the Cython extensions:

```python
import qhchina.analytics
qhchina.analytics.compile_cython_extensions()
```

## Usage Examples

### Topic Modeling with LDA

The module includes a fast Latent Dirichlet Allocation (LDA) implementation with Gibbs sampling:

```python
from qhchina.analytics import LDAGibbsSampler

# Prepare your data - each document as a list of tokens
documents = [
    ["word1", "word2", "word3", ...],
    ["word2", "word4", "word5", ...],
    # ...
]

# Initialize and train the model
lda = LDAGibbsSampler(
    n_topics=10,         # Number of topics
    alpha=0.1,           # Document-topic prior
    beta=0.01,           # Topic-word prior
    iterations=500,      # Number of Gibbs sampling iterations
    min_word_count=2,    # Minimum word count to include in vocab
    stopwords={"a", "the", "and"}  # Words to exclude
)

lda.fit(documents)

# Get top words for each topic
for i, topic in enumerate(lda.get_topic_words(10)):
    print(f"Topic {i}:")
    for word, prob in topic:
        print(f"  {word}: {prob:.4f}")
        
# Plot the topics
lda.plot_topic_words()

# Infer topics for a new document
new_doc = ["word1", "word3", "word6"]
topic_dist = lda.inference(new_doc)

# Visualize documents in 2D space (colored by dominant topic)
lda.visualize_documents(
    method='pca',           # Options: 'pca', 'tsne', 'mds', 'umap'
    show_labels=True,       # Show document labels
    label_strategy='auto',  # Automatic label display
    figsize=(12, 10),       # Figure size
    dpi=150,                # High resolution
    filename='docs_vis.png' # Save to file
)

# Create interactive HTML visualization
lda.visualize_documents(
    method='tsne',
    doc_labels=['Doc1', 'Doc2', ...],  # Custom document names
    format='html',          # Interactive HTML output
    filename='interactive.html'
)
```

### Document Visualization

The `visualize_documents()` method provides powerful visualization of your document corpus in 2D space:

**Dimensionality Reduction Methods:**
- `pca`: Principal Component Analysis (fast, linear)
- `tsne`: t-SNE (captures non-linear structure)
- `mds`: Multidimensional Scaling
- `umap`: UMAP (requires `umap-learn` package)

**Coloring:**
- By default, colors documents by their dominant topic
- Specify `n_clusters` to use k-means clustering instead

**Features:**
- Optional document labels with automatic spacing (uses `adjusttext` if available)
- Static (matplotlib) or interactive (HTML) visualizations
- Customizable appearance (size, colors, transparency, resolution)
- Smart label display strategies (auto/all/sample/none)

**Example:**
```python
# PCA with document labels
lda.visualize_documents(
    method='pca',
    doc_labels=['Document ' + str(i) for i in range(len(docs))],
    show_labels=True,
    label_strategy='sample',
    max_labels=30,
    filename='pca_visualization.png'
)

# Interactive t-SNE with hover tooltips
lda.visualize_documents(
    method='tsne',
    doc_labels=doc_names,
    format='html',
    filename='interactive_tsne.html'
)

# K-means clustering with MDS (specify n_clusters)
lda.visualize_documents(
    method='mds',
    n_clusters=5,  # Use k-means with 5 clusters
    figsize=(14, 10),
    dpi=200
)
```

**Optional Dependencies:**
- `umap-learn`: For UMAP dimensionality reduction
- `adjusttext`: For better label placement in static plots

Install with: `pip install umap-learn adjusttext`

See the `examples.py` module for more detailed examples.

## Performance Optimization

The LDA implementation uses Cython for performance-critical parts:

- Approximately 10-50x faster than pure Python implementation
- Automatically falls back to Python if Cython is not available
- Supports resuming training and saving/loading models

## API Reference

For complete API documentation, see the docstrings in each module:

- `collocations.py` - For collocation analysis
- `corpora.py` - For corpus comparison
- `vectors.py` - For vector operations
- `modeling.py` - For BERT-based modeling
- `topicmodels.py` - For topic modeling
- `examples.py` - For usage examples 