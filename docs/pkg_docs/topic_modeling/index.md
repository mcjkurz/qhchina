---
layout: docs_with_sidebar
title: Topic Modeling
permalink: /pkg_docs/topic_modeling/
---

# Topic Modeling with qhChina

Topic modeling is a technique used to discover the hidden thematic structure in document collections. qhChina provides powerful topic modeling capabilities through the `LDAGibbsSampler` class, which implements Latent Dirichlet Allocation (LDA) with Gibbs sampling.

## Basic Usage

The `LDAGibbsSampler` class provides a simple interface for topic modeling:

```python
from qhchina.preprocessing.segmentation import create_segmenter
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_stopwords
import numpy as np
import matplotlib.pyplot as plt

# Example tokenized documents after Chinese word segmentation
documents = [
    # Technology document
    ["人工智能", "正在", "改变", "我们", "的", "生活", "方式", "和", "工作", "效率"],
    
    # Healthcare document
    ["医生", "建议", "患者", "多", "喝", "水", "每天", "适当", "运动"],
    
    # Cultural document
    ["中国", "传统", "文化", "源远流长", "需要", "年轻人", "传承"],
    
    # Economic document
    ["经济", "发展", "需要", "创新", "驱动", "和", "人才", "支撑"],
    
    # Education document
    ["在", "大学", "里", "学习", "不仅", "是", "获取", "知识", "更是", "培养", "能力"]
]

# Create and fit the model
# Note: alpha defaults to 50/n_topics as recommended by Griffiths and Steyvers (2004)
lda = LDAGibbsSampler(
    n_topics=5,
    iterations=100,
    burnin=10,            # Number of initial iterations before alpha optimization
    log_interval=10
)
lda.fit(documents)

# Get all topics with their top words
topics = lda.get_topics(n_words=10)
for i, topic in enumerate(topics):
    print(f"Topic {i}:")
    for word, prob in topic:
        print(f"  {word}: {prob:.4f}")
```

## Hyperparameter Selection

### Alpha and Beta Parameters

The `LDAGibbsSampler` uses default values for hyperparameters based on Griffiths and Steyvers (2004):

- **alpha**: Controls document-topic distributions (default: `50 / n_topics`)
- **beta**: Controls topic-word distributions (default: `1 / n_topics`)

A higher alpha makes documents more similar in their topic mixtures, while a lower alpha creates more distinct topic distributions per document.

You can override these defaults:

```python
# Override the default alpha and beta
lda = LDAGibbsSampler(
    n_topics=5,
    alpha=0.1,
    beta=0.01,
    iterations=1000
)
```

## Automatic Alpha Estimation

The `LDAGibbsSampler` can automatically estimate the alpha parameter during training. Use the `burnin` parameter to specify initial iterations before estimation begins:

```python
lda = LDAGibbsSampler(
    n_topics=5,
    iterations=1000,
    burnin=100,            # Run 100 initial iterations before alpha estimation
    estimate_alpha=1       # Estimate alpha every 1 iteration (0 to disable)
)
```

Set `estimate_alpha=0` to disable alpha estimation, or use higher values (e.g., `estimate_alpha=10`) to estimate less frequently.

## Using Stopwords

You can improve topic quality by removing common stopwords:

```python
# Load simplified Chinese stopwords
stopwords = load_stopwords(language="zh_sim")

# Create model with stopwords filtering
lda = LDAGibbsSampler(
    n_topics=5, 
    iterations=1000,
    stopwords=stopwords
)
lda.fit(documents)
```

## Analyzing Topics and Documents

The `LDAGibbsSampler` provides several methods to analyze topics and documents:

```python
# Get the top words for a specific topic
topic_id = 0
top_words = lda.get_topic_words(topic_id=topic_id, n_words=10)
print(f"Top words for Topic {topic_id}:")
for word, prob in top_words:
    print(f"  {word}: {prob:.4f}")

# Get topic distribution for a specific document
doc_id = 0
doc_topics = lda.get_document_topics(doc_id=doc_id, sort_by_prob=True)
print(f"Topic distribution for document {doc_id}:")
for topic_id, prob in doc_topics:
    print(f"  Topic {topic_id}: {prob:.4f}")

# Get the top documents for a specific topic
topic_id = 0
top_docs = lda.get_top_documents(topic_id=topic_id, n_docs=5)
print(f"Top documents for Topic {topic_id}:")
for doc_id, prob in top_docs:
    print(f"  Document {doc_id}: {prob:.4f}")
```

## Visualizing Topics

Visualize the top words for each topic using the built-in plotting functions:

```python
# Plot top words for all topics in a single figure (horizontal bars)
lda.plot_topic_words(
    n_words=10,
    figsize=(12, 20),
    fontsize=12,
    filename="topics.png",
    orientation="horizontal",  # or "vertical"
    dpi=100
)

# Create separate plot files for each topic
lda.plot_topic_words(
    n_words=10,
    figsize=(8, 6),
    fontsize=12,
    filename="topic.png",
    separate_files=True,
    dpi=100
)
```

## Saving and Loading Models

You can save your trained model and load it later:

```python
# Save the trained model
lda.save("lda_model.npy")

# Load the model later
loaded_lda = LDAGibbsSampler.load("lda_model.npy")

# Use the loaded model
topics = loaded_lda.get_topics(n_words=5)
```

## Inference on New Documents

Infer topic distributions for new documents:

```python
# New document
new_doc = ["人工智能", "技术", "在", "医疗", "领域", "大有", "可为"]

# Infer topic distribution
topic_dist = lda.inference(new_doc, inference_iterations=50)
print("Topic distribution for new document:")
for i, prob in enumerate(topic_dist):
    print(f"  Topic {i}: {prob:.4f}")
```

## Measuring Similarity

Calculate similarity between topics or documents:

```python
# Compare two topics
similarity = lda.topic_similarity(topic_i=0, topic_j=1, metric='jsd')
print(f"Topic similarity: {similarity:.4f}")

# Get similarity matrix for all topics
topic_matrix = lda.topic_correlation_matrix(metric='cosine')

# Compare two documents
doc_sim = lda.document_similarity(doc_i=0, doc_j=1, metric='jsd')
print(f"Document similarity: {doc_sim:.4f}")

# Get similarity matrix for specific documents
doc_ids = [0, 1, 2, 3]
doc_matrix = lda.document_similarity_matrix(doc_ids=doc_ids, metric='cosine')
```

Available metrics: `'jsd'` (Jensen-Shannon divergence), `'hellinger'` (Hellinger distance), `'cosine'` (cosine similarity), or `'kl'` (KL divergence).

## Advanced Configuration

The `LDAGibbsSampler` offers several parameters for fine-tuning:

```python
lda = LDAGibbsSampler(
    n_topics=10,             # Number of topics
    alpha=None,              # Document-topic prior (default: 50/n_topics)
    beta=None,               # Topic-word prior (default: 1/n_topics)
    iterations=2000,         # Number of Gibbs sampling iterations
    burnin=200,              # Number of burn-in iterations before estimation
    random_state=42,         # Random seed for reproducibility
    log_interval=100,        # Print perplexity every N iterations
    min_word_count=2,        # Minimum word count to include in vocabulary
    max_vocab_size=10000,    # Maximum vocabulary size
    min_word_length=2,       # Minimum word length to include
    stopwords=stopwords,     # Set of stopwords to exclude
    use_cython=True,         # Use Cython acceleration if available
    estimate_alpha=1         # Estimate alpha every N iterations (0 = disable)
)
```

## Performance Optimization

The `LDAGibbsSampler` can use optimized Cython implementations for faster sampling. By default, the module automatically checks for Cython availability and uses it when possible. If Cython is not available, a warning will be issued, and the pure Python implementation will be used.

You can control Cython usage with the `use_cython` parameter:

```python
lda = LDAGibbsSampler(
    n_topics=10,
    use_cython=True
)
```

The Cython version typically offers 10-50x better performance than the pure Python implementation, which is particularly valuable for large corpora or many topics.

## Complete Example

Here is a complete example that demonstrates many features of the `LDAGibbsSampler`:

```python
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_stopwords, load_texts
import jieba
import re

# Load stopwords
stopwords = load_stopwords("zh_sim")

# Load and preprocess text data
texts = load_texts([
    "data/texts/新闻报道.txt", 
    "data/texts/学术论文.txt", 
    "data/texts/网络评论.txt", 
    "data/texts/政府公告.txt"
])
documents = []

for text in texts:
    # Basic cleaning
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize with jieba
    tokens = jieba.cut(text)
    
    # Add to documents
    documents.append(tokens)

# Train the LDA model with default alpha=50/n_topics
lda = LDAGibbsSampler(
    n_topics=5,
    iterations=1000,
    burnin=100,
    log_interval=100,
    stopwords=stopwords,
    min_word_count=2,
    min_word_length=2,
    use_cython=True,
    estimate_alpha=1
)

lda.fit(documents)

# Analyze the model
print("Topic distribution across corpus:")
print(lda.get_topic_distribution())

# Visualize topics
lda.plot_topic_words(
    n_words=10,
    figsize=(12, 20),
    fontsize=12,
    filename="topics.png",
    dpi=100
)

# Save the model
lda.save("lda_model.npy")
```

## API Reference

### LDAGibbsSampler

```python
class LDAGibbsSampler:
    def __init__(self, n_topics=10, alpha=None, beta=None, iterations=1000, 
                 burnin=0, random_state=None, log_interval=None, min_word_count=1, 
                 max_vocab_size=None, min_word_length=1, stopwords=None,
                 use_cython=True, estimate_alpha=1):
        """Initialize the LDA model with Gibbs sampling."""
        
    def fit(self, documents):
        """Fit the LDA model to the given documents."""
        
    def get_topics(self, n_words=10):
        """Get the top words for each topic along with their probabilities."""
        
    def get_topic_words(self, topic_id, n_words=10):
        """Get the top n words for a specific topic."""
        
    def get_document_topics(self, doc_id, sort_by_prob=False):
        """Get topic distribution for a specific document."""
        
    def get_top_documents(self, topic_id, n_docs=10):
        """Get the top n documents for a specific topic."""
        
    def get_topic_distribution(self):
        """Get overall topic distribution across the corpus."""
        
    def inference(self, new_doc, inference_iterations=100):
        """Infer topic distribution for a new document."""
    
    def topic_similarity(self, topic_i, topic_j, metric='jsd'):
        """Calculate similarity between two topics."""
    
    def topic_correlation_matrix(self, metric='jsd'):
        """Calculate pairwise similarity between all topics."""
    
    def document_similarity(self, doc_i, doc_j, metric='jsd'):
        """Calculate similarity between two documents."""
    
    def document_similarity_matrix(self, doc_ids=None, metric='jsd'):
        """Calculate pairwise similarity between documents."""
        
    def plot_topic_words(self, n_words=10, figsize=(12, 8), fontsize=10, 
                         filename=None, separate_files=False, dpi=72, 
                         orientation='horizontal'):
        """Plot the top words for each topic as a bar chart."""
        
    def save(self, filepath):
        """Save the model to a file."""
        
    @classmethod
    def load(cls, filepath):
        """Load a model from a file.""" 
```