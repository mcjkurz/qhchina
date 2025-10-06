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
LDAGibbsSampler(n_topics=10, alpha=None, beta=None, iterations=1000, burnin=0, 
                random_state=None, log_interval=None, min_word_count=1, 
                max_vocab_size=None, min_word_length=1, stopwords=None, 
                use_cython=True, estimate_alpha=1)
```

### Parameters

- `n_topics` (int): Number of topics (default: 10)
- `alpha` (float): Document-topic prior. If None, uses `50/n_topics` (Griffiths & Steyvers, 2004) (default: None)
- `beta` (float): Topic-word prior. If None, uses `1/n_topics` (Griffiths & Steyvers, 2004) (default: None)
- `iterations` (int): Number of Gibbs sampling iterations (default: 1000)
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
- `metric` (str): Similarity metric ('jsd', 'hellinger', 'cosine', or 'kl')

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
save(filepath)
```

```python
load(filepath)
```

Save or load model to/from file.

**Parameters:**
- `filepath` (str): Path to save/load model

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
    iterations=1000,
    burnin=100,
    log_interval=100,
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
