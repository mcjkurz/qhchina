---
layout: docs_with_sidebar
title: Word Embeddings
permalink: /pkg_docs/word_embeddings/
functions:
  - name: Word2Vec
    anchor: word2vec
  - name: Word2Vec.build_vocab()
    anchor: word2vec-build_vocab
  - name: Word2Vec.generate_cbow_examples()
    anchor: word2vec-generate_cbow_examples
  - name: Word2Vec.generate_skipgram_examples()
    anchor: word2vec-generate_skipgram_examples
  - name: Word2Vec.get_vector()
    anchor: word2vec-get_vector
  - name: Word2Vec.most_similar()
    anchor: word2vec-most_similar
  - name: Word2Vec.save()
    anchor: word2vec-save
  - name: Word2Vec.similarity()
    anchor: word2vec-similarity
  - name: Word2Vec.train()
    anchor: word2vec-train
  - name: TempRefWord2Vec
    anchor: temprefword2vec
  - name: TempRefWord2Vec.build_vocab()
    anchor: temprefword2vec-build_vocab
  - name: TempRefWord2Vec.calculate_semantic_change()
    anchor: temprefword2vec-calculate_semantic_change
  - name: TempRefWord2Vec.generate_cbow_examples()
    anchor: temprefword2vec-generate_cbow_examples
  - name: TempRefWord2Vec.generate_skipgram_examples()
    anchor: temprefword2vec-generate_skipgram_examples
  - name: TempRefWord2Vec.get_available_targets()
    anchor: temprefword2vec-get_available_targets
  - name: TempRefWord2Vec.get_period_vocab_counts()
    anchor: temprefword2vec-get_period_vocab_counts
  - name: TempRefWord2Vec.get_time_labels()
    anchor: temprefword2vec-get_time_labels
  - name: TempRefWord2Vec.save()
    anchor: temprefword2vec-save
  - name: TempRefWord2Vec.train()
    anchor: temprefword2vec-train
  - name: sample_sentences_to_token_count()
    anchor: sample_sentences_to_token_count
  - name: add_corpus_tags()
    anchor: add_corpus_tags
  - name: project_2d()
    anchor: project_2d
  - name: get_bias_direction()
    anchor: get_bias_direction
  - name: calculate_bias()
    anchor: calculate_bias
  - name: project_bias()
    anchor: project_bias
  - name: cosine_similarity()
    anchor: cosine_similarity
  - name: cosine_distance()
    anchor: cosine_distance
  - name: most_similar()
    anchor: most_similar
  - name: align_vectors()
    anchor: align_vectors
import_from: ['qhchina.analytics.word2vec', 'qhchina.analytics.vectors']
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

## API Reference

<!-- API-START -->

<h3 id="word2vec">Word2Vec</h3>

```python
Word2Vec(
    vector_size: int = 100,
    window: int = 5,
    min_word_count: int = 5,
    negative: int = 5,
    ns_exponent: float = 0.75,
    cbow_mean: bool = True,
    sg: int = 0,
    seed: int = 1,
    alpha: float = 0.025,
    min_alpha: Optional[float] = None,
    sample: float = 0.001,
    shrink_windows: bool = True,
    exp_table_size: int = 1000,
    max_exp: float = 6.0,
    max_vocab_size: Optional[int] = None,
    use_double_precision: bool = False,
    use_cython: bool = True,
    gradient_clip: float = 1.0
)
```

Implementation of Word2Vec algorithm with sample-based training approach.

This class implements both Skip-gram and CBOW architectures:
- Skip-gram (sg=1): Each training example is (input_idx, output_idx) where input is the center word
  and output is a context word.
- CBOW (sg=0): Each training example is (input_indices, output_idx) where inputs are context words
  and output is the center word.

Training is performed one example at a time, with negative examples generated for each positive example.

Features:
- CBOW and Skip-gram architectures with appropriate example generation
- Training with individual examples (one by one)
- Explicit negative sampling for each training example
- Subsampling of frequent words
- Dynamic window sizing with shrink_windows parameter
- Properly managed learning rate decay
- Sigmoid precomputation for faster training
- Vocabulary size restriction with max_vocab_size parameter
- Optional double precision for numerical stability
- Optional Cython acceleration for significantly faster training

Performance options:
- Use double precision (use_double_precision=True) for better numerical stability (slightly slower)
- Use Cython acceleration (use_cython=True) for much faster training (requires Cython extension)

<h4 id="word2vec-build_vocab">Word2Vec.build_vocab()</h4>

```python
build_vocab(sentences: Union[List[List[str]], Iterator[List[str]]])
```

Build vocabulary from a list or iterator of sentences.

Parameters:
-----------
sentences: List or iterator of tokenized sentences (each sentence is a list of words)

Returns:
--------
None

<h4 id="word2vec-generate_cbow_examples">Word2Vec.generate_cbow_examples()</h4>

```python
generate_cbow_examples(sentences: Union[List[List[str]], Iterator[List[str]]])
```

Generate CBOW training examples from sentences.

A CBOW example is a tuple (input_indices, output_idx) where:
- input_indices is a list of indices of context words
- output_idx is the index of the center word

For each positive example, the caller should generate negative examples using the noise distribution.

Parameters:
-----------
sentences: List or iterator of sentences (lists of words)

Returns:
--------
Generator yielding (input_indices, output_idx) tuples for positive examples

<h4 id="word2vec-generate_skipgram_examples">Word2Vec.generate_skipgram_examples()</h4>

```python
generate_skipgram_examples(sentences: Union[List[List[str]], Iterator[List[str]]])
```

Generate Skip-gram training examples from sentences.

A Skip-gram example is a tuple (input_idx, output_idx) where:
- input_idx is the index of the center word
- output_idx is the index of a context word

For each positive example, the caller should generate negative examples using the noise distribution.

Parameters:
-----------
sentences: List or iterator of sentences (lists of words)

Returns:
--------
Generator yielding (input_idx, output_idx) tuples for positive examples

<h4 id="word2vec-get_vector">Word2Vec.get_vector()</h4>

```python
get_vector(word: str, normalize: bool = False)
```

Get the vector for a word.

Parameters:
-----------
word: Input word
normalize: If True, return the normalized vector (unit length)

Returns:
--------
Word vector

<h4 id="word2vec-most_similar">Word2Vec.most_similar()</h4>

```python
most_similar(word: str, topn: int = 10)
```

Find the topn most similar words to the given word.

Parameters:
-----------
word: Input word
topn: Number of similar words to return

Returns:
--------
List of (word, similarity) tuples

<h4 id="word2vec-save">Word2Vec.save()</h4>

```python
save(path: str)
```

Save the model to a file.

Parameters:
-----------
path: Path to save the model

Returns:
--------
None

<h4 id="word2vec-similarity">Word2Vec.similarity()</h4>

```python
similarity(word1: str, word2: str)
```

Calculate cosine similarity between two words.

Parameters:
-----------
word1: First word
word2: Second word

Returns:
--------
Cosine similarity between the two words (float between -1 and 1)

Raises:
-------
KeyError: If either word is not in the vocabulary

<h4 id="word2vec-train">Word2Vec.train()</h4>

```python
train(sentences: Union[List[List[str]], Iterator[List[str]]], epochs: int = 1, alpha: Optional[float] = None, min_alpha: Optional[float] = None, total_examples: Optional[int] = None, batch_size: int = 2000, callbacks: List[Callable] = None, calculate_loss: bool = True, verbose: Optional[int] = None)
```

Train word2vec model on given sentences.

Parameters:
-----------
sentences: List or iterator of tokenized sentences (lists of words)
epochs: Number of training iterations over the corpus
alpha: Initial learning rate
min_alpha: Minimum allowed learning rate. When provided, enables learning rate decay
    from `alpha` down to `min_alpha` over the course of training. By default, this
    requires counting the total number of training examples upfront to calculate
    the decay schedule (which can be slow). Use `total_examples` to skip counting.
total_examples: Total number of training examples per epoch. When provided along
    with `min_alpha`, skips the automatic counting step and uses this value for
    learning rate decay scheduling. This is useful when you already know the
    corpus size or want to control the decay rate independently of corpus size.
batch_size: Batch size for training; if 0, no batching is used; default is 2000
callbacks: List of callback functions to call after each epoch
calculate_loss: Whether to calculate and return the final loss
verbose: Controls logging frequency. If None, no batch-level logging in simple mode.
    If an integer, logs progress every `verbose` batches (when batched) or every
    `verbose * 1000` examples (when not batched).
    
Returns:
--------
Final loss value if calculate_loss is True, None otherwise

<br>

<h3 id="temprefword2vec">TempRefWord2Vec</h3>

```python
TempRefWord2Vec(
    corpora: List[List[List[str]]],
    labels: List[str],
    targets: List[str],
    balance: bool = True,
    **kwargs
)
```

Implementation of Word2Vec with Temporal Referencing (TR) for tracking semantic change.

This class extends Word2Vec to implement temporal referencing, where target words
are represented with time period indicators (e.g., "bread_1800" for period 1800s) when used
as target words, but remain unchanged when used as context words.

The class takes multiple corpora corresponding to different time periods and automatically
creates temporal references for specified target words.

Usage:
------
1. Initialize with corpora from different time periods, labels for the periods,
   and target words to track for semantic change
2. The model will process, balance, and combine the corpora
3. Call train() without arguments to train on the preprocessed data
4. Access semantic change through most_similar() or by directly analyzing the word vectors
   of temporal variants (e.g., "bread_1800" vs "bread_1900")

**Example:**
```python
```

<h4 id="temprefword2vec-build_vocab">TempRefWord2Vec.build_vocab()</h4>

```python
build_vocab(sentences: List[List[str]])
```

Extends the parent build_vocab method to handle temporal word variants.

Explicitly adds base words to the vocabulary even if they don't appear in the corpus.

Parameters:
-----------
sentences: List of tokenized sentences

<h4 id="temprefword2vec-calculate_semantic_change">TempRefWord2Vec.calculate_semantic_change()</h4>

```python
calculate_semantic_change(target_word: str, labels: Optional[List[str]] = None)
```

Calculate semantic change by comparing cosine similarities across time periods.

Parameters:
-----------
target_word: Target word to analyze (must be one of the targets specified during initialization)
labels: Time period labels (optional, defaults to labels from model initialization)

Returns:
--------
Dict mapping transition names to lists of (word, change) tuples, sorted by change score (descending)

Example:
--------
>>> changes = model.calculate_semantic_change("人民")
>>> for transition, word_changes in changes.items():
>>>     print(f"\n{transition}:")
>>>     print("Words moved towards:", word_changes[:5])  # Top 5 increases
>>>     print("Words moved away:", word_changes[-5:])   # Top 5 decreases

<h4 id="temprefword2vec-generate_cbow_examples">TempRefWord2Vec.generate_cbow_examples()</h4>

```python
generate_cbow_examples(sentences: List[List[str]])
```

Override parent method to implement temporal referencing in CBOW model.

For CBOW, temporal referencing means:
- Context words (inputs) should be converted to their base forms
- Target words (outputs) are already handled by data preprocessing (with temporal variants)

This implementation calls the parent's implementation and then modifies the yielded
examples by converting any temporal variant context words to their base form.

Parameters:
-----------
sentences: List of sentences (lists of words)

Returns:
--------
Generator yielding (input_indices, output_idx) tuples for positive examples

<h4 id="temprefword2vec-generate_skipgram_examples">TempRefWord2Vec.generate_skipgram_examples()</h4>

```python
generate_skipgram_examples(sentences: List[List[str]])
```

Override parent method to implement temporal referencing in Skip-gram model.

For Skip-gram, temporal referencing means that target words (inputs) are replaced
with their temporal variants, while context words (outputs) remain unchanged.

This implementation calls the parent's implementation and then modifies the yielded
examples by converting any temporal variant context words to their base form.

Parameters:
-----------
sentences: List of sentences (lists of words)

Returns:
--------
Generator yielding (input_idx, output_idx) tuples for positive examples

<h4 id="temprefword2vec-get_available_targets">TempRefWord2Vec.get_available_targets()</h4>

```python
get_available_targets()
```

Get the list of target words available for semantic change analysis.

**Returns:**


<h4 id="temprefword2vec-get_period_vocab_counts">TempRefWord2Vec.get_period_vocab_counts()</h4>

```python
get_period_vocab_counts(period: Optional[str] = None)
```

Get vocabulary counts for a specific period or all periods.

Parameters:
-----------
period : str, optional
    The period label to get vocab counts for. If None, returns all periods.
    
Returns:
--------
Union[Dict[str, Counter], Counter]
    If period is None: dictionary mapping period labels to Counter objects
    If period is specified: Counter object for that specific period
    
Raises:
-------
ValueError
    If the specified period is not found in the model

<h4 id="temprefword2vec-get_time_labels">TempRefWord2Vec.get_time_labels()</h4>

```python
get_time_labels()
```

Get the list of time period labels used in the model.

**Returns:**


<h4 id="temprefword2vec-save">TempRefWord2Vec.save()</h4>

```python
save(path: str)
```

Save the TempRefWord2Vec model to a file, including vocab counts and temporal metadata.

This overrides the parent save method to also save:
- Period-specific vocabulary counts
- Target words and labels  
- Temporal word mappings
- All other model parameters from the parent class

Note: The combined corpus is NOT saved to reduce file size.

Parameters:
-----------
path : str
    Path to save the model file
    
Returns:
--------
None

<h4 id="temprefword2vec-train">TempRefWord2Vec.train()</h4>

```python
train(sentences: Optional[List[str]] = None, **kwargs)
```

Train the TempRefWord2Vec model using the preprocessed combined corpus.

Unlike the parent Word2Vec class, TempRefWord2Vec always uses its internal combined_corpus
that was created and preprocessed during initialization. This ensures the training
data has the proper temporal references.

Parameters:
-----------
sentences: Ignored in TempRefWord2Vec, will use self.combined_corpus instead
**kwargs: All additional arguments are passed to the parent's train method
         (epochs, batch_size, alpha, min_alpha, callbacks, calculate_loss, etc.)

Returns:
--------
Final loss value if calculate_loss is True in kwargs, None otherwise

<br>

<h3 id="sample_sentences_to_token_count">sample_sentences_to_token_count()</h3>

```python
sample_sentences_to_token_count(
    corpus: List[List[str]],
    target_tokens: int,
    seed: Optional[int] = None
)
```

Samples sentences from a corpus until the target token count is reached.

This function randomly selects sentences from the corpus until the total number
of tokens reaches or slightly exceeds the target count. This is useful for balancing
corpus sizes when comparing different time periods or domains.

Parameters:
-----------
corpus : List[List[str]]
    A list of sentences, where each sentence is a list of tokens
target_tokens : int
    The target number of tokens to sample
seed : Optional[int]
    Random seed for reproducibility. If None, uses global seed.
    
Returns:
--------
List[List[str]]
    A list of sampled sentences with token count close to target_tokens

<br>

<h3 id="add_corpus_tags">add_corpus_tags()</h3>

```python
add_corpus_tags(
    corpora: List[List[List[str]]],
    labels: List[str],
    target_words: List[str]
)
```

Add corpus-specific tags to target words in all corpora at once.

**Parameters:**
- `corpora`: List of corpora (each corpus is list of tokenized sentences)
- `labels`: List of corpus labels
- `target_words`: List of words to tag

**Returns:**
List of processed corpora where target words have been tagged with their corpus label

<br>

<h3 id="project_2d">project_2d()</h3>

```python
project_2d(
    vectors: Union[List[numpy.ndarray], Dict[str, numpy.ndarray], numpy.ndarray],
    labels: Optional[List[str]] = None,
    method: str = 'pca',
    title: Optional[str] = None,
    color: Union[str, List[str], NoneType] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = 12,
    perplexity: Optional[float] = None,
    filename: Optional[str] = None,
    adjust_text_labels: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1
)
```

Projects high-dimensional vectors into 2D using PCA, t-SNE, or UMAP and visualizes them.

Parameters:
vectors (list of vectors or dict {label: vector}): Vectors to project.
labels (list of str, optional): List of labels for the vectors. Defaults to None.
method (str, optional): Method to use for projection ('pca', 'tsne', or 'umap'). Defaults to 'pca'.
title (str, optional): Title of the plot. Defaults to None.
color (list of str or str, optional): List of colors for the vectors or a single color. Defaults to None.
figsize (tuple, optional): Figure size as (width, height). Defaults to (8, 8).
fontsize (int, optional): Font size for labels. Defaults to 12.
perplexity (float, optional): Perplexity parameter for t-SNE. Required if method is 'tsne'.
filename (str, optional): Path to save the figure. Defaults to None.
adjust_text_labels (bool, optional): Whether to adjust text labels to avoid overlap. Defaults to False.
n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
min_dist (float, optional): Minimum distance between points for UMAP. Defaults to 0.1.

<br>

<h3 id="get_bias_direction">get_bias_direction()</h3>

```python
get_bias_direction(
    anchors: Union[Tuple[numpy.ndarray, numpy.ndarray], List[Tuple[numpy.ndarray, numpy.ndarray]]]
)
```

Given either a single tuple (pos_anchor, neg_anchor) or a list of tuples,

compute the direction vector for measuring bias by taking the mean of 
differences between positive and negative anchor pairs.

Parameters:
anchors: A tuple (pos_vector, neg_vector) or list of such tuples
        Each vector in the pairs should be a numpy array

Returns:
numpy array representing the bias direction vector (unnormalized)

<br>

<h3 id="calculate_bias">calculate_bias()</h3>

```python
calculate_bias(
    anchors: Union[Tuple[str, str], List[Tuple[str, str]]],
    targets: List[str],
    word_vectors: Any
)
```

Calculate bias scores for target words along an axis defined by anchor pairs.

Parameters:
anchors: tuple or list of tuples, e.g. ("man", "woman") or [("king", "queen"), ("man", "woman")]
targets: list of words to calculate bias for
word_vectors: keyed vectors (e.g. from word2vec_model.wv)

Returns:
numpy array of bias scores (dot products) for each target word

<br>

<h3 id="project_bias">project_bias()</h3>

```python
project_bias(
    x,
    y,
    targets,
    word_vectors,
    title=None,
    color=None,
    figsize=(8, 8),
    fontsize=12,
    filename=None,
    adjust_text_labels=False,
    disperse_y=False
)
```

Plots words on either a 1D or 2D chart by projecting them onto:

- axis_x: derived from x (single tuple or list of tuples)
  - axis_y: derived from y (single tuple or list of tuples), if provided

Parameters remain the same as before, but calculation of bias scores is now handled separately.

<br>

<h3 id="cosine_similarity">cosine_similarity()</h3>

```python
cosine_similarity(
    v1: Union[numpy.ndarray, List[float]],
    v2: Union[numpy.ndarray, List[float]]
)
```

Compute the cosine similarity between vectors.

If v1 and v2 are single vectors, computes similarity between them.
If either is a matrix of vectors, uses sklearn's implementation for efficiency.

Returns 0.0 if either vector has zero norm (to avoid division by zero).

Parameters:
-----------
v1 : numpy.ndarray or list
    First vector or matrix of vectors
v2 : numpy.ndarray or list  
    Second vector or matrix of vectors
    
Returns:
--------
float or numpy.ndarray
    Cosine similarity score(s). For single vectors, returns a float in range [-1, 1].
    For matrices, returns a 2D similarity matrix.

<br>

<h3 id="cosine_distance">cosine_distance()</h3>

```python
cosine_distance(
    v1: Union[numpy.ndarray, List[float]],
    v2: Union[numpy.ndarray, List[float]]
)
```

Compute the cosine distance between vectors (1 - cosine_similarity).

Cosine distance is a dissimilarity measure where 0 means identical vectors
and 2 means opposite vectors.

Parameters:
-----------
v1 : numpy.ndarray or list
    First vector or matrix of vectors
v2 : numpy.ndarray or list  
    Second vector or matrix of vectors
    
Returns:
--------
float or numpy.ndarray
    Cosine distance score(s). For single vectors, returns a float in range [0, 2].
    For matrices, returns a 2D distance matrix.

<br>

<h3 id="most_similar">most_similar()</h3>

```python
most_similar(
    target_vector: numpy.ndarray,
    vectors: Union[List[numpy.ndarray], numpy.ndarray],
    labels: Optional[List[str]] = None,
    metric: Union[str, Callable[[numpy.ndarray, numpy.ndarray], float]] = 'cosine',
    top_n: Optional[int] = None
)
```

Find the most similar vectors to a target vector using the specified similarity metric.

Parameters:
target_vector (numpy.ndarray): The reference vector to compare against
vectors (list or numpy.ndarray): List of vectors to compare with the target
labels (list, optional): Labels corresponding to the vectors. If provided, returns (label, score) pairs
metric (str or callable, optional): Similarity metric to use. Can be 'cosine' or a callable that takes two vectors
top_n (int, optional): Number of top results to return. If None, returns all results

Returns:
If labels provided: List of (label, score) tuples sorted by similarity score in descending order
If no labels: List of (index, score) tuples sorted by similarity score in descending order

<br>

<h3 id="align_vectors">align_vectors()</h3>

```python
align_vectors(source_vectors: numpy.ndarray, target_vectors: numpy.ndarray)
```

Align source vectors with target vectors using Procrustes analysis.

**Parameters:**
- `source_vectors`: numpy array of vectors to be aligned
- `target_vectors`: numpy array of vectors to align to

**Returns:**
Tuple of (aligned_vectors, transformation_matrix)
- aligned_vectors: The aligned source vectors
- transformation_matrix: The orthogonal transformation matrix that can be used to align other vectors

<br>

<!-- API-END -->

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
