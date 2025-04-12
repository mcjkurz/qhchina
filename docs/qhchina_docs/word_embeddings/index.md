---
layout: default
title: Word Embeddings
permalink: /qhchina_docs/word_embeddings/
---

# Word Embeddings in qhChina

This page documents the word embeddings functionality in the qhChina package, with a focus on the customized Word2Vec implementation.

## Word2Vec Implementation
qhChina provides a custom implementation of Word2Vec with both CBOW (Continuous Bag of Words) and Skip-gram architectures, designed specifically for research in humanities and social sciences with Chinese text.

### Basic Usage

```python
from qhchina.analytics import Word2Vec

# Initialize a Word2Vec model
model = Word2Vec(
    vector_size=100,  # Dimensionality of word vectors
    window=5,         # Context window size
    min_count=5,      # Minimum word frequency threshold
    sg=1,             # 1 for Skip-gram; 0 for CBOW
    negative=5,       # Number of negative samples
    alpha=0.025,      # Initial learning rate
    seed=42           # Random seed for reproducibility
)

# Prepare tokenized sentences
sentences = [
    ["我", "喜欢", "这部", "电影"],
    ["这", "是", "一个", "有趣", "的", "故事"],
    # More sentences...
]

# Train the model
model.train(sentences, epochs=5)

# Get word vector
vector = model.get_vector("电影")

# Find similar words
similar_words = model.most_similar("电影", topn=10)
```

### Key Features

#### Architecture Options

- **CBOW (Continuous Bag of Words)**: Predicts the target word from context words
- **Skip-gram**: Predicts context words from the target word

```python
# CBOW model (default)
cbow_model = Word2Vec(sg=0)

# Skip-gram model
skipgram_model = Word2Vec(sg=1)
```

#### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `vector_size` | Dimensionality of word vectors (default: 100) |
| `window` | Maximum distance between target and context words (default: 5) |
| `min_count` | Ignores words with frequency below this threshold (default: 5) |
| `alpha` | Initial learning rate (default: 0.025) |
| `min_alpha` | Final learning rate (default: None) |
| `negative` | Number of negative samples for each positive sample (default: 5) |
| `ns_exponent` | Exponent for negative sampling distribution (default: 0.75) |
| `max_vocab_size` | Maximum vocabulary size (default: None) |
| `sample` | Threshold for downsampling frequent words (default: 1e-3) |
| `shrink_windows` | Whether to use dynamic window size (default: True) |
| `seed` | Random seed for reproducibility (default: 1) |
| `cbow_mean` | Whether to use mean or sum for context word vectors in CBOW (default: True) |
| `use_double_precision` | Whether to use double precision for calculations (default: False) |
| `use_cython` | Whether to use Cython for performance-critical operations (default: False) |
| `gradient_clip` | Clipping value for gradients (default: 1.0) |
| `exp_table_size` | Size of the precomputed sigmoid table (default: 1000) |
| `max_exp` | Maximum value in the precomputed sigmoid table (default: 6.0) |

#### Batch Training

The Word2Vec implementation supports batch-based training for better performance:

```python
# Train with batching
model.train(sentences, epochs=5, batch_size=64)
```

#### Advanced Methods

```python
# Save and load models
model.save("my_model.model")
loaded_model = Word2Vec.load("my_model.model")

# Update model with new sentences
model.build_vocab(new_sentences, update=True)
model.train(new_sentences, epochs=3)

# Word similarity
similarity = model.similarity("电影", "电视")

# Get most similar words
similar_words = model.most_similar("中国", topn=10)
```

## Temporal Reference Word2Vec

qhChina provides a specialized implementation called `TempRefWord2Vec` for tracking semantic change over time. This model does not require training separate models for each time period. Instead, it creates temporal variants of target words in a single vector space using a specialized training approach.

### Basic Usage

```python
from qhchina.analytics import TempRefWord2Vec

# Prepare corpus data from different time periods
time_labels = ["1980", "1990", "2000", "2010"]
corpora = [corpus_1980, corpus_1990, corpus_2000, corpus_2010]

# Target words to track for semantic change
target_words = ["改革", "经济", "科技", "人民"]

# Initialize and train the model in one step
model = TempRefWord2Vec(
    corpora=corpora,          # List of corpora for different time periods
    labels=time_labels,       # Labels for each time period
    targets=target_words,     # Words to track for semantic change
    vector_size=256,
    window=5,
    min_count=5,
    sg=1,                     # Use Skip-gram model
    negative=10,
    seed=42
)

# Train the model
model.train(calculate_loss=True, batch_size=64)

# Access temporal variants of words
reform_1980s = model.get_vector("改革_1980")
reform_2010s = model.get_vector("改革_2010")

# Find words similar to a target in a specific time period
similar_to_reform_1980s = model.most_similar("改革_1980", topn=10)
similar_to_reform_2010s = model.most_similar("改革_2010", topn=10)
```

### How It Works

The `TempRefWord2Vec` model works by:

1. Creating temporal variants of target words by appending time period labels (e.g., "改革_1980s", "改革_2010s")
2. Training a single Word2Vec model with all corpora, but making each target word specific to its time period
3. Maintaining the shared vector space for all non-target words across all time periods
4. This allows direct comparison of how a word's semantic associations change over time

### Analyzing Semantic Change

```python
def calculate_semantic_change(model, target_word, labels, limit_top_similar=200, min_length=2):
    """
    Calculate semantic change by comparing cosine similarities across time periods.
    
    Parameters:
    -----------
    model: Trained TempRefWord2Vec model
    target_word: Target word to analyze
    labels: Time period labels
    limit_top_similar: Number of most similar words to consider
    min_length: Minimum word length to include
    
    Returns:
    --------
    Dict mapping transition names to lists of (word, change) tuples
    """
    results = {}
    
    # Get all words in vocabulary (excluding temporal variants)
    all_words = [word for word in model.vocab.keys() 
                if word not in model.reverse_temporal_map]
    
    # Get embeddings for all words
    all_word_vectors = np.array([model.get_vector(word) for word in all_words])

    # For each adjacent pair of time periods
    for i in range(len(labels) - 1):
        from_period = labels[i]
        to_period = labels[i+1]
        transition = f"{from_period}_to_{to_period}"
        
        # Get temporal variants for the target word
        from_variant = f"{target_word}_{from_period}"
        to_variant = f"{target_word}_{to_period}"
        
        # Get vectors for the target word in each period
        from_vector = model.get_vector(from_variant).reshape(1, -1)
        to_vector = model.get_vector(to_variant).reshape(1, -1)
        
        # Calculate cosine similarity for all words with the target word in each period
        from_sims = cosine_similarity(from_vector, all_word_vectors)[0]
        to_sims = cosine_similarity(to_vector, all_word_vectors)[0]
        
        # Calculate differences in similarity
        sim_diffs = to_sims - from_sims
        
        # Create word-change pairs and sort by change
        word_changes = [(all_words[i], float(sim_diffs[i])) for i in range(len(all_words))]
        word_changes.sort(key=lambda x: x[1], reverse=True)
        
        # Consider only words that were among the most similar in either period
        most_similar_from = model.most_similar(from_variant, topn=limit_top_similar)
        most_similar_to = model.most_similar(to_variant, topn=limit_top_similar)
        
        considered_words = set(word for word, _ in most_similar_from) | set(word for word, _ in most_similar_to)
        
        # Filter results based on considered words and length
        word_changes = [change for change in word_changes 
                      if change[0] in considered_words and len(change[0]) >= min_length]
        
        results[transition] = word_changes
    
    return results

# Example usage
target_word = "人民"
changes = calculate_semantic_change(model, target_word, time_labels)

# Display words that became more associated with "人民"
for transition, word_changes in changes.items():
    print(f"\nTransition: {transition}")
    
    # Words with increased similarity (moved towards)
    print("Words moved towards:")
    for word, change in word_changes[:10]:
        print(f"  {word}: {change:.4f}")
    
    # Words with decreased similarity (moved away)
    print("\nWords moved away from:")
    for word, change in word_changes[-10:]:
        print(f"  {word}: {change:.4f}")
```

### Visualization Examples

You can visualize the semantic change using the standard vector projection tools:

```python
from qhchina.analytics.vectors import project_2d
from sklearn.decomposition import PCA

# Get vectors for target word across all time periods
target_word = "改革"
vectors = {}
for period in time_labels:
    temporal_variant = f"{target_word}_{period}"
    vectors[temporal_variant] = model.get_vector(temporal_variant)

# Add common words to the visualization
common_words = ["政策", "开放", "经济", "发展", "市场"]
for word in common_words:
    vectors[word] = model.get_vector(word)

# Project to 2D
project_2d(
    vectors=vectors,
    method="pca",
    title=f"Semantic Change of '{target_word}' Over Time",
    adjust_text_labels=True
)
```

## Vector Analysis

qhChina provides tools for analyzing and visualizing word embeddings.

### Vector Projection

```python
from qhchina.analytics.vectors import project_2d

# Project vectors to 2D space using PCA
project_2d(
    vectors={word: model.get_vector(word) for word in ["中国", "美国", "俄罗斯", "日本", "德国"]},
    method="pca",
    title="Countries in Vector Space"
)

# Using t-SNE for better clustering visualization
project_2d(
    vectors={word: model.get_vector(word) for word in words_list},
    method="tsne",
    perplexity=5,
    title="t-SNE Projection of Word Vectors"
)
```

### Bias Analysis

```python
from qhchina.analytics.vectors import calculate_bias, project_bias

# Define gender dimension
gender_pairs = [("男人", "女人"), ("他", "她"), ("父亲", "母亲")]

# Calculate bias scores along gender dimension
target_words = ["医生", "护士", "工程师", "教师", "科学家"]
bias_scores = calculate_bias(gender_pairs, target_words, model)

# Project words on the gender dimension
project_bias(
    x=gender_pairs,
    y=None,
    targets=target_words,
    word_vectors=model,
    title="Gender Bias in Profession Words"
)
```

### Vector Alignment

When comparing word vectors across different models (e.g., from different training runs), you can align them to enable direct comparison:

```python
from qhchina.analytics.vectors import align_vectors

# Align model2's vectors to model1's vector space
align_vectors(model1, model2)

# Now you can directly compare vectors
vector1 = model1.get_vector("电影")
vector2 = model2.get_vector("电影")
```

## Practical Examples

### Analyzing Conceptual Change

```python
# Initialize model with specific parameters for historical analysis
model = Word2Vec(
    vector_size=200,
    window=10,
    min_count=10,
    sg=1,
    negative=10
)

# Train on early period corpus
model.build_vocab(early_period_texts)
model.train(early_period_texts, epochs=5)
early_model = model.copy()

# Update model with later period corpus
model.build_vocab(later_period_texts, update=True)
model.train(later_period_texts, epochs=5)

# Compare semantic neighborhoods
early_neighbors = early_model.most_similar("革命", topn=20)
modern_neighbors = model.most_similar("革命", topn=20)
```

### Creating Semantic Fields

```python
# Get all words similar to a concept
economy_terms = model.most_similar("经济", topn=50)

# Find clusters within a semantic field
from sklearn.cluster import KMeans
from qhchina.analytics.vectors import cosine_similarity

# Get vectors for economy-related terms
vectors = [model.get_vector(word) for word, _ in economy_terms]

# Cluster vectors
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(vectors)

# Group words by cluster
semantic_fields = {}
for i, (word, _) in enumerate(economy_terms):
    cluster = clusters[i]
    if cluster not in semantic_fields:
        semantic_fields[cluster] = []
    semantic_fields[cluster].append(word)
```

## Performance Considerations

- For large corpora, increase `max_vocab_size` to limit memory usage
- Use `sample` parameter to downsample frequent words for better results
- For very large vocabularies, consider filtering words before training
- Set `shrink_windows=True` for more diverse contexts during training

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26. 
3. Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. arXiv preprint arXiv:1605.09096. 