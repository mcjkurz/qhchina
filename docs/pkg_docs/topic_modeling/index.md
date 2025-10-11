---
layout: docs_with_sidebar
title: Topic Modeling
permalink: /pkg_docs/topic_modeling/
---

# Topic Modeling

The `qhchina.analytics.topicmodels` module provides Latent Dirichlet Allocation (LDA) with Gibbs sampling for discovering thematic structure in document collections.

## LDAGibbsSampler

### Initialization

```python
LDAGibbsSampler(n_topics=10, alpha=None, beta=None, iterations=100, burnin=0, 
                random_state=None, log_interval=None, min_word_count=1, 
                max_vocab_size=None, min_word_length=1, stopwords=None, 
                use_cython=True, estimate_alpha=1)
```

### Parameters

- `n_topics` (int): Number of topics (default: 10)
- `alpha` (float or array): Document-topic prior. Can be a float (symmetric prior) or array of floats (asymmetric prior, one value per topic). If None, uses `50/n_topics` (Griffiths & Steyvers, 2004) (default: None)
- `beta` (float): Topic-word prior. If None, uses `1/n_topics` (Griffiths & Steyvers, 2004) (default: None)
- `iterations` (int): Number of Gibbs sampling iterations (default: 100)
- `burnin` (int): Initial iterations before alpha optimization (default: 0)
- `random_state` (int): Random seed for reproducibility (default: None)
- `log_interval` (int): Calculate and print perplexity every N iterations (default: None)
- `min_word_count` (int): Minimum word count to include in vocabulary (default: 1)
- `max_vocab_size` (int): Maximum vocabulary size (default: None)
- `min_word_length` (int): Minimum word length to include (default: 1)
- `stopwords` (set): Set of words to exclude (default: None)
- `use_cython` (bool): Use Cython acceleration if available (default: True)
- `estimate_alpha` (int): Estimate alpha every N iterations (0 = disable) (default: 1)

### Main Methods

```python
fit(documents)
```

Fit the LDA model to documents.

**Parameters:**
- `documents` (list): List of tokenized documents (each document is a list of tokens)

<br>

```python
get_topics(n_words=10)
```

Get top words for all topics.

**Parameters:**
- `n_words` (int): Number of words per topic

**Returns:** (list) List of lists containing (word, probability) tuples

<br>

```python
get_topic_words(topic_id, n_words=10)
```

Get top words for a specific topic.

**Parameters:**
- `topic_id` (int): Topic ID
- `n_words` (int): Number of words to return

**Returns:** (list) List of (word, probability) tuples

<br>

```python
get_document_topics(doc_id, sort_by_prob=False)
```

Get topic distribution for a document.

**Parameters:**
- `doc_id` (int): Document ID
- `sort_by_prob` (bool): Sort topics by probability

**Returns:** (list) List of (topic_id, probability) tuples

<br>

```python
get_top_documents(topic_id, n_docs=10)
```

Get top documents for a topic.

**Parameters:**
- `topic_id` (int): Topic ID
- `n_docs` (int): Number of documents to return

**Returns:** (list) List of (doc_id, probability) tuples

<br>

```python
get_topic_distribution()
```

Get overall topic distribution across the corpus.

**Returns:** (numpy.ndarray) Topic distribution

<br>

```python
inference(new_doc, inference_iterations=100)
```

Infer topic distribution for a new document.

**Parameters:**
- `new_doc` (list): Tokenized document
- `inference_iterations` (int): Number of inference iterations

**Returns:** (numpy.ndarray) Topic distribution

<br>

```python
topic_similarity(topic_i, topic_j, metric='jsd')
```

Calculate similarity between two topics.

**Parameters:**
- `topic_i` (int): First topic ID
- `topic_j` (int): Second topic ID
- `metric` (str): Similarity metric. Available options:
  - `'jsd'` - Jensen-Shannon Divergence
  - `'hellinger'` - Hellinger Distance
  - `'cosine'` - Cosine Similarity
  - `'kl'` - Kullback-Leibler Divergence

**Returns:** (float) Similarity score

<br>

```python
topic_correlation_matrix(metric='jsd')
```

Calculate pairwise similarity between all topics.

**Parameters:**
- `metric` (str): Similarity metric

**Returns:** (numpy.ndarray) Similarity matrix

<br>

```python
document_similarity(doc_i, doc_j, metric='jsd')
```

Calculate similarity between two documents.

**Parameters:**
- `doc_i` (int): First document ID
- `doc_j` (int): Second document ID
- `metric` (str): Similarity metric

**Returns:** (float) Similarity score

<br>

```python
document_similarity_matrix(doc_ids=None, metric='jsd')
```

Calculate pairwise similarity between documents.

**Parameters:**
- `doc_ids` (list): List of document IDs (None for all)
- `metric` (str): Similarity metric

**Returns:** (numpy.ndarray) Similarity matrix

<br>

```python
plot_topic_words(n_words=10, figsize=(12, 8), fontsize=10, filename=None, 
                 separate_files=False, dpi=72, orientation='horizontal')
```

Plot top words for topics as bar charts.

**Parameters:**
- `n_words` (int): Number of words per topic
- `figsize` (tuple): Figure size
- `fontsize` (int): Font size
- `filename` (str): Output filename (None for display)
- `separate_files` (bool): Create separate file for each topic
- `dpi` (int): Resolution
- `orientation` (str): Bar orientation ('horizontal' or 'vertical')

<br>

```python
visualize_documents(method='pca', n_clusters=None, doc_labels=None,
                   show_labels=False, label_strategy='auto', use_adjusttext=True,
                   max_labels=None, figsize=(12, 10), dpi=150, alpha=0.7, size=50,
                   cmap='tab10', title=None, filename=None, format='static',
                   random_state=None, highlight=None, n_topic_words=4, **kwargs)
```

Visualize documents in 2D space using dimensionality reduction.

Documents are automatically colored by dominant topic, or by k-means clusters if `n_clusters` is specified.

**Parameters:**
- `method` (str): Dimensionality reduction method. Options:
  - `'pca'` - Principal Component Analysis (fast, linear)
  - `'tsne'` - t-SNE (captures non-linear structure)
  - `'mds'` - Multidimensional Scaling
  - `'umap'` - UMAP (requires `umap-learn` package)
- `n_clusters` (int): If specified, use k-means clustering instead of topic coloring
- `doc_labels` (list): Optional document names/labels
- `show_labels` (bool): Whether to show document labels
- `label_strategy` (str): Label display strategy ('auto', 'all', 'sample', 'none')
- `use_adjusttext` (bool): Use adjustText for better label placement (if available)
- `max_labels` (int): Maximum number of labels to show per topic/cluster
- `figsize` (tuple): Figure size
- `dpi` (int): Resolution
- `alpha` (float): Point transparency (0-1)
- `size` (float): Point size
- `cmap` (str): Matplotlib colormap name
- `title` (str): Plot title
- `filename` (str): Output filename
- `format` (str): Output format ('static' or 'html')
- `random_state` (int): Random seed
- `highlight` (int or list): Topic ID(s) to highlight. Non-highlighted topics shown in gray. In HTML format, all topics appear in legend and can be toggled interactively
- `n_topic_words` (int): Number of representative words per topic in legend (default: 4). Increase figsize width if using many words
- `**kwargs`: Additional parameters for dimensionality reduction (e.g., `perplexity` for t-SNE, `n_neighbors` for UMAP)

**Returns:** 2D coordinates array (if format='static'), None (if format='html')

**Note:** Optional dependencies: `umap-learn` for UMAP, `adjusttext` for label adjustment

<br>

```python
save(filepath)
```

Save model to file.

**Parameters:**
- `filepath` (str): Path to save the model

<br>

```python
LDAGibbsSampler.load(filepath)
```

Load model from file. This is a class method.

**Parameters:**
- `filepath` (str): Path to load the model from

**Returns:**
- Loaded LDAGibbsSampler instance

<br>

## Examples

### Basic Topic Modeling

```python
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_stopwords

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Example tokenized documents
documents = [
    ["人工智能", "正在", "改变", "我们", "的", "生活", "方式"],
    ["医生", "建议", "患者", "多", "喝", "水", "每天", "运动"],
    ["中国", "传统", "文化", "源远流长", "需要", "传承"],
    # More documents...
]

# Create and fit model
lda = LDAGibbsSampler(
    n_topics=5,
    iterations=100,
    burnin=20,
    log_interval=20,
    stopwords=stopwords,
    min_word_count=2,
    estimate_alpha=1
)
lda.fit(documents)

# Get topics
topics = lda.get_topics(n_words=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}:")
    for word, prob in topic:
        print(f"  {word}: {prob:.4f}")

# Visualize topics
lda.plot_topic_words(n_words=10, figsize=(12, 20), filename="topics.png")

# Save model
lda.save("lda_model.npy")

# Load model later
loaded_lda = LDAGibbsSampler.load("lda_model.npy")
```

### Analyzing Documents and Topics

```python
# Get topic distribution for a specific document
doc_topics = lda.get_document_topics(doc_id=0, sort_by_prob=True)
print(f"Document 0 topics:")
for topic_id, prob in doc_topics:
    print(f"  Topic {topic_id}: {prob:.4f}")

# Infer topics for a new document
new_doc = ["人工智能", "技术", "医疗", "领域"]
topic_dist = lda.inference(new_doc, inference_iterations=50)
print("New document topic distribution:", topic_dist)

# Calculate topic similarity
similarity = lda.topic_similarity(topic_i=0, topic_j=1, metric='jsd')
print(f"Topic similarity: {similarity:.4f}")
```

### Visualizing Documents in 2D Space

```python
# PCA visualization colored by dominant topic (default)
lda.visualize_documents(
    method='pca',
    figsize=(12, 10),
    dpi=150,
    filename='documents_pca.png'
)

# t-SNE with document labels (sample per topic)
doc_labels = [f"Document_{i}" for i in range(len(documents))]
lda.visualize_documents(
    method='tsne',
    doc_labels=doc_labels,
    show_labels=True,
    label_strategy='sample',  # Show sample of labels per topic
    max_labels=5,             # Show up to 5 documents per topic
    figsize=(14, 12),
    dpi=200,
    filename='documents_tsne.png'
)

# K-means clustering with MDS (specify n_clusters to use k-means)
lda.visualize_documents(
    method='mds',
    n_clusters=3,  # Automatically uses k-means when n_clusters is set
    figsize=(10, 8),
    filename='documents_clusters.png'
)

# Interactive HTML visualization
lda.visualize_documents(
    method='pca',
    doc_labels=doc_labels,
    format='html',  # Creates interactive visualization
    filename='documents_interactive.html'
)

# Highlight specific topics (static plot)
lda.visualize_documents(
    method='pca',
    highlight=[0, 2, 5],  # Only topics 0, 2, and 5 shown in color
    figsize=(12, 10),
    filename='documents_highlighted.png'
)

# Custom number of words in legend (static plot)
lda.visualize_documents(
    method='pca',
    n_topic_words=6,      # Show 6 words per topic in legend
    figsize=(14, 10),     # Wider figure to accommodate longer legend
    filename='documents_6words.png'
)

# Interactive HTML with highlighting and custom topic words
lda.visualize_documents(
    method='tsne',
    doc_labels=doc_labels,
    format='html',
    highlight=[0, 2, 5],  # Initially highlight these topics
    n_topic_words=6,      # Show 6 words per topic in legend
    perplexity=50,        # Custom t-SNE parameter
    filename='documents_custom.html'
)

# UMAP with custom parameters (if umap-learn is installed)
try:
    lda.visualize_documents(
        method='umap',
        doc_labels=doc_labels,
        format='html',
        n_neighbors=15,       # UMAP parameter
        min_dist=0.1,         # UMAP parameter
        filename='documents_umap.html'
    )
except ImportError:
    print("Install umap-learn: pip install umap-learn")
```

**Interactive HTML Features:**

The HTML format creates a standalone file with:
- **Hover tooltips** showing document name/ID and top 3 topic probabilities
- **Click topics** in the legend to toggle highlighting on/off
- **Click points** on the canvas to toggle their topic's highlighting
- **Select All / Deselect All button** to quickly toggle all topics at once
- **Responsive legend** that updates based on highlighted topics
- All topics shown in legend (grayed when not highlighted)

This is useful for exploring large document collections and interactively focusing on specific topics.
