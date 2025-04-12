# qhChina Analytics Module

This module provides utilities for text analytics and topic modeling.

## Features

- **Collocation Analysis**: Find significant word co-occurrences in text
- **Corpus Comparison**: Statistically compare different corpora
- **Vector Operations**: Project and manipulate word embeddings
- **BERT-based Modeling**: Text classification and embedding with BERT
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
    min_count=2,         # Minimum word count to include in vocab
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
```

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