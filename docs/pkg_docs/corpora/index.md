---
layout: docs_with_sidebar
title: Corpus Analysis
permalink: /pkg_docs/corpora/
---

# Corpus Analysis in qhChina

qhChina provides a suite of tools for analyzing corpus data, with a focus on comparing corpora and identifying linguistic patterns.

## Comparing Corpora

The `compare_corpora` function allows you to identify statistically significant differences in word usage between two corpora. This is particularly useful for studying language variation across different text collections, such as texts from different time periods, regions, or sources.

### Basic Usage

```python
from qhchina.analytics import compare_corpora

# Example data
corpus_a = ["中国", "经济", "发展", "改革", "经济", "政策", "中国", "市场"]
corpus_b = ["美国", "经济", "市场", "金融", "美国", "贸易", "进口", "出口"]

# Compare the corpora
results = compare_corpora(
    corpusA=corpus_a, 
    corpusB=corpus_b, 
    method="fisher",
    filters={"min_word_count": 10},
    as_dataframe=True
)

# Sort by statistical significance
results = results.sort_values("p_value")
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `corpusA` | List of tokens from the first corpus |
| `corpusB` | List of tokens from the second corpus |
| `method` | Statistical test to use: 'fisher' (default), 'chi2', or 'chi2_corrected' |
| `filters` | Dictionary of filters to apply to results (see Filtering Options) |
| `as_dataframe` | Whether to return results as a pandas DataFrame (default: True) |

### Filtering Options

The `filters` parameter accepts a dictionary with the following options:

| Filter | Description |
|--------|-------------|
| `min_word_count` | Minimum count for a word to be included (int or tuple of two ints) |
| `max_p` | Maximum p-value threshold for statistical significance |
| `stopwords` | List of words to exclude from results |
| `min_word_length` | Minimum character length for words |

### Results Interpretation

The function returns a DataFrame (or list of dictionaries), with each entry containing:

- `word`: The word being compared
- `abs_freqA`: Absolute frequency in corpus A
- `abs_freqB`: Absolute frequency in corpus B
- `rel_freqA`: Relative frequency in corpus A
- `rel_freqB`: Relative frequency in corpus B
- `rel_ratio`: Ratio of relative frequencies (A:B)
- `p_value`: Statistical significance of the difference

A small p-value indicates that the difference in word frequency between the two corpora is statistically significant.

### Example Analysis

```python
# Apply multiple filters during corpus comparison
filtered_results = compare_corpora(
    corpusA=corpus_a,
    corpusB=corpus_b,
    method="fisher",
    filters={
        "min_word_count": 3,  # Minimum count in both corpora
        "max_p": 0.05,        # Only statistically significant differences
        "stopwords": ["的", "了", "和"],  # Exclude common words
        "min_word_length": 2  # Only include words with at least 2 characters
    }
)

# Identify words that are significantly more common in corpus A
words_overrepresented_in_A = filtered_results[
    (filtered_results["p_value"] < 0.05) & 
    (filtered_results["rel_ratio"] > 1)
]

# Identify words that are significantly more common in corpus B
words_overrepresented_in_B = filtered_results[
    (filtered_results["p_value"] < 0.05) & 
    (filtered_results["rel_ratio"] < 1)
]

# Visualize the most significant differences
import matplotlib.pyplot as plt
import numpy as np

top_words = filtered_results.sort_values("p_value").head(10)
plt.figure(figsize=(10, 6))
plt.barh(
    top_words["word"],
    np.log2(top_words["rel_ratio"]),
    color=[("blue" if ratio > 1 else "red") for ratio in top_words["rel_ratio"]]
)
plt.axvline(x=0, color="black", linestyle="-")
plt.xlabel("Log2 Ratio (Corpus A / Corpus B)")
plt.title("Most Significant Word Frequency Differences")
plt.tight_layout()
plt.show()
```

## Co-occurrence Matrix

The `cooc_matrix` function allows you to create a co-occurrence matrix from a collection of documents, capturing how frequently words occur together within a context.

### Basic Usage

```python
from qhchina.analytics.collocations import cooc_matrix

# Example data - list of tokenized documents
documents = [
    ["中国", "经济", "发展", "改革"],
    ["美国", "经济", "市场", "金融"],
    ["中国", "市场", "贸易", "改革"],
    # More documents...
]

# Create co-occurrence matrix using window method
cooc = cooc_matrix(
    documents=documents,
    method="window",
    horizon=2,           # Context window size
    min_abs_count=2,     # Minimum word frequency
    as_dataframe=True    # Return as pandas DataFrame
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `documents` | List of tokenized documents, where each document is a list of tokens |
| `method` | Co-occurrence method: 'window' (default) or 'document' |
| `horizon` | Size of the context window (only used if `method='window'`) |
| `min_abs_count` | Minimum absolute count for a word to be included |
| `min_doc_count` | Minimum number of documents a word must appear in |
| `vocab_size` | Maximum vocabulary size (optional) |
| `binary` | Count co-occurrences as binary (0/1) rather than frequencies |
| `as_dataframe` | Return matrix as a pandas DataFrame |
| `vocab` | Predefined vocabulary to use (optional) |
| `use_sparse` | Use a sparse matrix for better memory efficiency with large vocabularies |

### Word Co-occurrence Analysis

```python
# Find words that frequently co-occur with a target word
target_word = "经济"
if target_word in cooc.index:
    cooc_with_target = cooc[target_word].sort_values(ascending=False)
    print(f"Words co-occurring with '{target_word}':")
    print(cooc_with_target.head(10))

# Visualize co-occurrence network
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph from the co-occurrence matrix
G = nx.Graph()

# Add nodes for each word
for word in cooc.index:
    G.add_node(word)

# Add edges for co-occurrences above a threshold
threshold = 3  # Minimum co-occurrence count
for word1 in cooc.index:
    for word2 in cooc.columns:
        if word1 != word2 and cooc.loc[word1, word2] >= threshold:
            G.add_edge(word1, word2, weight=cooc.loc[word1, word2])

# Draw the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=100)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.axis("off")
plt.title("Word Co-occurrence Network")
plt.tight_layout()
plt.show()
```

## Temporal Analysis with TempRefWord2Vec

For analyzing corpus data over time, qhChina provides the `TempRefWord2Vec` model and supporting functions. This allows you to track semantic change in specific words across different time periods.

```python
from qhchina.analytics import TempRefWord2Vec

# Prepare corpus data from different time periods
time_labels = ["1980s", "1990s", "2000s", "2010s"]
corpora = [corpus_1980s, corpus_1990s, corpus_2000s, corpus_2010s]

# Target words to track for semantic change
target_words = ["改革", "经济", "科技", "人民"]

# Initialize and train the model
model = TempRefWord2Vec(
    corpora=corpora,          # List of corpora for different time periods
    labels=time_labels,       # Labels for each time period
    targets=target_words,     # Words to track for semantic change
    balance=True,             # Balance corpus sizes
    vector_size=100,
    window=5,
    min_word_count=5,
    sg=1                      # Use Skip-gram model
)

# Train the model
model.train(epochs=5)

# Access temporal variants of words
reform_1980s = model.get_vector("改革_1980s")
reform_2010s = model.get_vector("改革_2010s")

# Find similar words for a target in different time periods
similar_to_reform_1980s = model.most_similar("改革_1980s", topn=10)
similar_to_reform_2010s = model.most_similar("改革_2010s", topn=10)
```

## Combining with Word Embeddings

You can use the corpus analysis tools in combination with word embeddings for more sophisticated analyses:

```python
from qhchina.analytics.word2vec import Word2Vec
from qhchina.analytics.corpora import compare_corpora

# Train Word2Vec model on your corpus
model = Word2Vec(vector_size=100, window=5, min_word_count=5)
model.build_vocab(all_sentences)
model.train(all_sentences, epochs=5)

# Analyze differences between two corpora
comparison = compare_corpora(
    corpusA=corpus_a, 
    corpusB=corpus_b,
    filters={"min_word_count": 5, "max_p": 0.05}
)
significant_words = comparison[comparison["p_value"] < 0.05]["word"].tolist()

# Examine the semantic relationships between significant words
from qhchina.analytics.vectors import project_2d

# Get vectors for significant words that appear in the model
significant_vectors = {}
for word in significant_words:
    if word in model.vocab:
        significant_vectors[word] = model.get_vector(word)

# Visualize the semantic space
if significant_vectors:
    project_2d(
        vectors=significant_vectors,
        method="tsne",
        perplexity=5,
        title="Semantic Space of Statistically Significant Words"
    )
```

## Practical Examples

### Comparative Analysis of Historical Texts

```python
# Comparing language use in different historical periods
modern_corpus = ["现代", "科技", "发展", "技术", "电脑", "系统", ...]
classical_corpus = ["古代", "诗词", "文学", "礼仪", "制度", ...]

# Find distinctive vocabulary in each period
period_comparison = compare_corpora(
    modern_corpus, 
    classical_corpus,
    min_count=5
)
```

### Topic-Focused Analysis

```python
# Extract all sentences containing a specific term
from qhchina.helpers import texts

economy_sentences = texts.extract_sentences_with_term(
    all_sentences, 
    "经济"
)

# Compare sentences with the term to the broader corpus
economy_comparison = compare_corpora(
    [token for sentence in economy_sentences for token in sentence],
    [token for sentence in all_sentences for token in sentence],
    min_count=5
)
```

## Performance Considerations

- For very large corpora, consider using `min_word_count` to filter out rare words
- When creating co-occurrence matrices for large vocabularies, use `use_sparse=True`
- The `compare_corpora` function stores results in memory, so for extremely large corpora, consider processing in batches

## References

1. Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. Computational linguistics, 19(1), 61-74.
2. Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. Computational linguistics, 16(1), 22-29.
3. Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. arXiv preprint arXiv:1605.09096. 