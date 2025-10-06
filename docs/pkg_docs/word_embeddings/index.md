---
layout: docs_with_sidebar
title: Word Embeddings
permalink: /pkg_docs/word_embeddings/
---

# Word Embeddings

The `qhchina.analytics.word2vec` module provides Word2Vec implementations for Chinese text analysis, including standard Word2Vec and TempRefWord2Vec for tracking semantic change over time.

## Word2Vec

Implementation of Word2Vec with both CBOW and Skip-gram architectures.

### Initialization

```python
Word2Vec(vector_size=100, window=5, min_word_count=5, sg=1, negative=5, 
         alpha=0.025, min_alpha=None, ns_exponent=0.75, max_vocab_size=None, 
         sample=1e-3, shrink_windows=True, seed=1, cbow_mean=True, 
         use_double_precision=False, use_cython=False, gradient_clip=1.0, 
         exp_table_size=1000, max_exp=6.0)
```

### Key Parameters

- `vector_size` (int): Dimensionality of word vectors (default: 100)
- `window` (int): Maximum distance between target and context words (default: 5)
- `min_word_count` (int): Ignores words with frequency below threshold (default: 5)
- `sg` (int): Training algorithm: 1 for Skip-gram, 0 for CBOW (default: 1)
- `alpha` (float): Initial learning rate (default: 0.025)
- `negative` (int): Number of negative samples (default: 5)
- `sample` (float): Threshold for downsampling frequent words (default: 1e-3)
- `seed` (int): Random seed for reproducibility (default: 1)
- `cbow_mean` (bool): Use mean (True) or sum (False) for context vectors in CBOW (default: True)
- `use_cython` (bool): Use Cython for performance-critical operations (default: False)

### Main Methods

```python
build_vocab(sentences, update=False)
```

Build vocabulary from tokenized sentences.

**Parameters:**
- `sentences` (list): List of tokenized sentences
- `update` (bool): Update existing vocabulary instead of replacing

```python
train(sentences, epochs=5, batch_size=None)
```

Train the Word2Vec model.

**Parameters:**
- `sentences` (list): List of tokenized sentences
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training (optional)

```python
get_vector(word)
```

Get the vector representation of a word.

**Parameters:**
- `word` (str): The word to get vector for

**Returns:** (numpy.ndarray) Word vector

```python
most_similar(word, topn=10)
```

Find the most similar words.

**Parameters:**
- `word` (str): Target word
- `topn` (int): Number of similar words to return

**Returns:** (list) List of (word, similarity) tuples

```python
similarity(word1, word2)
```

Calculate cosine similarity between two words.

**Parameters:**
- `word1` (str): First word
- `word2` (str): Second word

**Returns:** (float) Cosine similarity score

```python
save(filepath)
```

```python
load(filepath)
```

Save or load model to/from file.

**Parameters:**
- `filepath` (str): Path to save/load model

## TempRefWord2Vec

Specialized implementation for tracking semantic change over time. Creates temporal variants of target words in a single vector space.

### Initialization

```python
TempRefWord2Vec(corpora, labels, targets, vector_size=100, window=5, 
                min_word_count=5, sg=1, negative=5, alpha=0.025, seed=1, ...)
```

### Key Parameters

- `corpora` (list): List of corpora for different time periods (list of lists of sentences)
- `labels` (list): Labels for each time period (list of strings)
- `targets` (list): Words to track for semantic change (list of strings)
- Additional parameters: Same as Word2Vec

### Main Methods

```python
train(calculate_loss=True, batch_size=64)
```

Train the temporal reference model.

```python
get_vector(word)
```

Get vector for a word (including temporal variants like "word_1980").

```python
calculate_semantic_change(target_word)
```

Calculate semantic change for a target word across time periods.

**Parameters:**
- `target_word` (str): The target word to analyze

**Returns:** (dict) Dictionary mapping transitions to lists of (word, change_score) tuples

## Vector Analysis Functions

From `qhchina.analytics.vectors`:

```python
project_2d(vectors, method='pca', title=None, adjust_text_labels=True, 
           perplexity=30, **kwargs)
```

Project word vectors to 2D space for visualization.

**Parameters:**
- `vectors` (dict): Dictionary mapping words to vectors
- `method` (str): Projection method ('pca' or 'tsne')
- `title` (str): Plot title
- `adjust_text_labels` (bool): Adjust text labels to avoid overlap
- `perplexity` (int): Perplexity for t-SNE (if using t-SNE)

```python
calculate_bias(dimension_pairs, target_words, model)
```

Calculate bias scores along a semantic dimension.

**Parameters:**
- `dimension_pairs` (list): List of word pairs defining the dimension
- `target_words` (list): Words to calculate bias for
- `model`: Word2Vec model

**Returns:** (dict) Mapping of target words to bias scores

```python
align_vectors(model1, model2)
```

Align vectors from two models for direct comparison.

**Parameters:**
- `model1`: Reference Word2Vec model
- `model2`: Model to align to model1's space

## Examples

### Basic Word2Vec Training

```python
from qhchina.analytics.word2vec import Word2Vec

# Initialize model
model = Word2Vec(vector_size=100, window=5, min_word_count=5, sg=1, seed=42)

# Prepare tokenized sentences
sentences = [
    ["我", "喜欢", "这部", "电影"],
    ["这", "是", "一个", "有趣", "的", "故事"],
    # More sentences...
]

# Train model
model.train(sentences, epochs=5)

# Get similar words
similar_words = model.most_similar("电影", topn=10)
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# Calculate similarity
sim = model.similarity("电影", "电视")
print(f"Similarity: {sim:.4f}")
```

### Tracking Semantic Change Over Time

```python
from qhchina.analytics.word2vec import TempRefWord2Vec

# Prepare corpus data from different time periods
time_labels = ["1980", "1990", "2000", "2010"]
corpora = [corpus_1980, corpus_1990, corpus_2000, corpus_2010]  # Each is a list of tokenized sentences

# Words to track for semantic change
target_words = ["改革", "经济", "科技"]

# Initialize and train model
model = TempRefWord2Vec(
    corpora=corpora,
    labels=time_labels,
    targets=target_words,
    vector_size=100,
    window=5,
    sg=1,
    seed=42
)
model.train(calculate_loss=True)

# Access temporal variants
reform_1980s = model.get_vector("改革_1980")
reform_2010s = model.get_vector("改革_2010")

# Analyze semantic change
changes = model.calculate_semantic_change("改革")
for transition, word_changes in changes.items():
    print(f"\n{transition}:")
    print("Words moved towards:", word_changes[:10])
    print("Words moved away:", word_changes[-10:])
```
