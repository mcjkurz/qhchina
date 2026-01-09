---
layout: docs_with_sidebar
title: Word Embeddings
permalink: /pkg_docs/word_embeddings/
functions:
  - name: Word2Vec()
    anchor: word2vec
  - name: build_vocab()
    anchor: build_vocab
  - name: train()
    anchor: train
  - name: get_vector()
    anchor: get_vector
  - name: most_similar()
    anchor: most_similar
  - name: similarity()
    anchor: similarity
  - name: save() / load()
    anchor: save-load
  - name: TempRefWord2Vec()
    anchor: temprefword2vec
  - name: calculate_semantic_change()
    anchor: calculate_semantic_change
  - name: project_2d()
    anchor: project_2d
  - name: calculate_bias()
    anchor: calculate_bias
  - name: align_vectors()
    anchor: align_vectors
  - name: cosine_similarity()
    anchor: cosine_similarity
  - name: most_similar() (vectors)
    anchor: most_similar-vectors
  - name: project_bias()
    anchor: project_bias
---

# Word Embeddings

The `qhchina.analytics.word2vec` module provides Word2Vec implementations for Chinese text analysis, including standard Word2Vec and TempRefWord2Vec for tracking semantic change over time.

```python
from qhchina.analytics.word2vec import Word2Vec

model = Word2Vec(vector_size=100, window=5, min_word_count=5)
model.train(sentences, epochs=5)
similar = model.most_similar("经济", topn=10)  # Find words similar to "经济"
```

---

<h3 id="word2vec">Word2Vec()</h3>

Implementation of Word2Vec with both CBOW and Skip-gram architectures.

```python
Word2Vec(vector_size=100, window=5, min_word_count=5, sg=1, negative=5, 
         alpha=0.025, min_alpha=None, ns_exponent=0.75, max_vocab_size=None, 
         sample=1e-3, shrink_windows=True, seed=1, cbow_mean=True, 
         use_double_precision=False, use_cython=False, gradient_clip=1.0, 
         exp_table_size=1000, max_exp=6.0)
```

**Parameters:**
- `vector_size` (int): Dimensionality of word vectors (default: 100)
- `window` (int): Maximum distance between target and context words (default: 5)
- `min_word_count` (int): Ignores words with frequency below threshold (default: 5)
- `sg` (int): Training algorithm: 1 for Skip-gram, 0 for CBOW (default: 1)
- `alpha` (float): Initial learning rate (default: 0.025)
- `min_alpha` (float): Minimum learning rate for decay. When provided, enables learning rate decay from `alpha` down to `min_alpha` over training. This requires counting the total number of training examples upfront to calculate the decay schedule, so expect a brief delay before training begins.
- `negative` (int): Number of negative samples (default: 5)
- `sample` (float): Threshold for downsampling frequent words (default: 1e-3)
- `seed` (int): Random seed for reproducibility (default: 1)
- `cbow_mean` (bool): Use mean (True) or sum (False) for context vectors in CBOW (default: True)
- `use_cython` (bool): Use Cython for performance-critical operations (default: False)

<br>

<h3 id="build_vocab">build_vocab()</h3>

```python
build_vocab(sentences, update=False)
```

Build vocabulary from tokenized sentences.

**Parameters:**
- `sentences` (list): List of tokenized sentences
- `update` (bool): Update existing vocabulary instead of replacing

<br>

<h3 id="train">train()</h3>

```python
train(sentences, epochs=5, batch_size=2000, total_examples=None, verbose=None)
```

Train the Word2Vec model.

**Parameters:**
- `sentences` (list): List of tokenized sentences
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training (default: 2000)
- `total_examples` (int, optional): Total training examples per epoch. When provided with `min_alpha`, skips the automatic counting step and uses this value for learning rate decay scheduling
- `verbose` (int, optional): Controls logging frequency. If `None`, no batch-level logging in simple mode. If an integer, logs progress every `verbose` batches (e.g., `verbose=100` logs every 100 batches)

<br>

<h3 id="get_vector">get_vector()</h3>

```python
get_vector(word)
```

Get the vector representation of a word.

**Parameters:**
- `word` (str): The word to get vector for

**Returns:** (numpy.ndarray) Word vector

<br>

<h3 id="most_similar">most_similar()</h3>

```python
most_similar(word, topn=10)
```

Find the most similar words.

**Parameters:**
- `word` (str): Target word
- `topn` (int): Number of similar words to return

**Returns:** (list) List of (word, similarity) tuples

<br>

<h3 id="similarity">similarity()</h3>

```python
similarity(word1, word2)
```

Calculate cosine similarity between two words.

**Parameters:**
- `word1` (str): First word
- `word2` (str): Second word

**Returns:** (float) Cosine similarity score

<br>

<h3 id="save-load">save() / load()</h3>

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

<h3 id="temprefword2vec">TempRefWord2Vec()</h3>

Specialized implementation for tracking semantic change over time. Creates temporal variants of target words in a single vector space.

```python
TempRefWord2Vec(corpora, labels, targets, vector_size=100, window=5, 
                min_word_count=5, sg=1, negative=5, alpha=0.025, seed=1, ...)
```

**Parameters:**
- `corpora` (list): List of corpora for different time periods (list of lists of sentences)
- `labels` (list): Labels for each time period (list of strings)
- `targets` (list): Words to track for semantic change (list of strings)
- Additional parameters: Same as Word2Vec

<br>

```python
train(calculate_loss=True, batch_size=2000)
```

Train the temporal reference model.

<br>

```python
get_vector(word)
```

Get vector for a word (including temporal variants like "word_1980").

<br>

<h3 id="calculate_semantic_change">calculate_semantic_change()</h3>

```python
calculate_semantic_change(target_word)
```

Calculate semantic change for a target word across time periods.

**Parameters:**
- `target_word` (str): The target word to analyze

**Returns:** (dict) Dictionary mapping transitions to lists of (word, change_score) tuples

<br>

<h3 id="project_2d">project_2d()</h3>

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

<br>

<h3 id="calculate_bias">calculate_bias()</h3>

```python
calculate_bias(anchors, targets, word_vectors)
```

Calculate bias scores for target words along an axis defined by anchor pairs.

**Parameters:**
- `anchors` (tuple or list): Either a tuple like `("man", "woman")` or a list of tuples like `[("king", "queen"), ("man", "woman")]` defining the bias dimension
- `targets` (list): List of words to calculate bias for
- `word_vectors`: Dictionary-like object mapping words to vectors (e.g., `model.wv` from Word2Vec)

**Returns:** (numpy.ndarray) Array of bias scores (dot products) for each target word

<br>

<h3 id="align_vectors">align_vectors()</h3>

```python
align_vectors(source_vectors, target_vectors)
```

Align source vectors with target vectors using Procrustes analysis.

**Parameters:**
- `source_vectors` (numpy.ndarray): Vectors to be aligned
- `target_vectors` (numpy.ndarray): Vectors to align to

**Returns:** (tuple) 
- `aligned_vectors`: The aligned source vectors
- `transformation_matrix`: The orthogonal transformation matrix that can be used to align other vectors

<br>

<h3 id="cosine_similarity">cosine_similarity()</h3>

```python
cosine_similarity(v1, v2)
```

Compute the cosine similarity between vectors. Returns 0.0 if either vector has zero norm.

**Parameters:**
- `v1` (numpy.ndarray or list): First vector or matrix of vectors
- `v2` (numpy.ndarray or list): Second vector or matrix of vectors

**Returns:** (float or numpy.ndarray) Cosine similarity score(s)

<br>

<h3 id="most_similar-vectors">most_similar() (vectors module)</h3>

```python
most_similar(target_vector, vectors, labels=None, metric='cosine', top_n=None)
```

Find the most similar vectors to a target vector.

**Parameters:**
- `target_vector` (numpy.ndarray): The reference vector to compare against
- `vectors` (list or numpy.ndarray): List of vectors to compare with the target
- `labels` (list): Optional labels corresponding to the vectors
- `metric` (str or callable): Similarity metric - `'cosine'` or a custom callable
- `top_n` (int): Number of top results to return (None for all)

**Returns:** (list) List of (label/index, score) tuples sorted by similarity in descending order

<br>

<h3 id="project_bias">project_bias()</h3>

```python
project_bias(x, y, targets, word_vectors, title=None, color=None, 
             figsize=(8,8), fontsize=12, filename=None, 
             adjust_text_labels=False, disperse_y=False)
```

Plot words on a 1D or 2D chart by projecting them onto bias axes.

**Parameters:**
- `x` (tuple or list): Bias axis for x-dimension, e.g., `("man", "woman")` or list of tuples
- `y` (tuple, list, or None): Bias axis for y-dimension. If None, creates 1D plot
- `targets` (list): List of words to plot
- `word_vectors`: Keyed vectors (e.g., `model.wv` from Word2Vec)
- `title` (str): Plot title
- `color` (str or list): Color(s) for points
- `figsize` (tuple): Figure size
- `fontsize` (int): Font size for labels
- `filename` (str): Path to save the figure
- `adjust_text_labels` (bool): Adjust text labels to avoid overlap (requires `adjustText`)
- `disperse_y` (bool): Add random y-dispersion in 1D plot for readability

---

## Examples

**Basic Word2Vec Training**

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

**Tracking Semantic Change Over Time**

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
