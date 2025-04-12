---
layout: default
title: Collocation Analysis
permalink: /qhchina_docs/collocations/
---

# Collocation Analysis in qhChina

This page documents the collocation analysis functionality in the qhChina package, which allows you to identify and analyze words that frequently occur together in text.

## Finding Collocates

The `find_collocates` function allows you to identify words that co-occur with specific target words more frequently than would be expected by chance.

### Basic Usage

```python
from qhchina.analytics.collocations import find_collocates

# Find collocates of "经济"
collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    method="window",
    horizon=3,
    as_dataframe=True
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `sentences` | List of tokenized sentences (each a list of tokens) |
| `target_words` | List of target words (or a single word) to find collocates for |
| `method` | Method to use: 'window' (default) or 'sentence' |
| `horizon` | Context window size (only used with `method='window'`) |
| `filters` | Dictionary of filters to apply to results (see section on Filtering Results) |
| `as_dataframe` | Whether to return results as a pandas DataFrame (default: True) |

### Results Interpretation

The function returns a list of dictionaries (or a DataFrame), with each entry containing:

- `target`: The target word
- `collocate`: A word that co-occurs with the target
- `exp_local`: Expected frequency of co-occurrence (if independent)
- `obs_local`: Observed frequency of co-occurrence
- `ratio_local`: Ratio of observed to expected frequency
- `obs_global`: Total frequency of the collocate in the corpus
- `p_value`: Statistical significance of the association

A small p-value indicates that the association between the target word and the collocate is statistically significant. The `ratio_local` value indicates the strength of the association, with higher values indicating stronger associations.

### Example Analysis

```python
# Get top 10 most significant collocates
top_collocates = collocates.head(10)
print("Top collocates of target word:")
for _, row in top_collocates.iterrows():
    print(f"{row['collocate']}: observed={row['obs_local']}, expected={row['exp_local']:.2f}, ratio={row['ratio_local']}")
```

## Collocation Methods

qhChina provides two methods for finding collocates:

### Window-based Collocation

The window-based method (`method='window'`) looks for words that appear within a specified distance (horizon) of the target word. This method is better for identifying words that have a close syntactic relationship with the target word.

```python
# Find words that appear within 3 words of "中国"
window_collocates = find_collocates(
    sentences=sentences,
    target_words=["中国"],
    method="window",
    horizon=3
)
```

### Sentence-based Collocation

The sentence-based method (`method='sentence'`) looks for words that appear in the same sentence as the target word. This method is better for identifying broader thematic associations.

```python
# Find words that appear in the same sentences as "改革"
sentence_collocates = find_collocates(
    sentences=sentences,
    target_words=["改革"],
    method="sentence"
)
```

## Multiple Target Words

You can analyze collocates for multiple target words simultaneously:

```python
# Find collocates for multiple target words
multi_collocates = find_collocates(
    sentences=sentences,
    target_words=["中国", "美国", "日本"],
    method="window",
    horizon=3
)
china_collocates = china_collocates[china_collocates["target"] == "中国"]
```

## Filtering Results

The `filters` parameter allows you to apply multiple filters to the results:

```python
# Define filters
filters = {
    'max_p': 0.05,              # Maximum p-value threshold
    'stopwords': ["的", "了", "在", "是", "和", "有", "被"],  # Words to exclude
    'min_length': 2             # Minimum character length
}

# Find collocates with filters
filtered_collocates = find_collocates(
    sentences=sentences,
    target_words=["经济"],
    filters=filters
)
```

### Available Filters

| Filter | Description |
|--------|-------------|
| `max_p` | Maximum p-value threshold for statistical significance |
| `stopwords` | List of words to exclude from results |
| `min_length` | Minimum character length for collocates |

### Filtering After Results

You can also apply filters after obtaining the results:

```python
# Get only statistically significant collocates (p < 0.05)
significant_collocates = collocates[collocates["p_value"] < 0.05]

# Sort by strength of association
significant_collocates = significant_collocates.sort_values("ratio_local", ascending=False)
```

## Visualizing Collocations

### Bar Chart of Top Collocates

```python
import matplotlib.pyplot as plt

# Get top 10 collocates by significance
top10 = collocates.sort_values("p_value").head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top10["collocate"][::-1],  # Reverse for bottom-to-top display
    top10["ratio_local"][::-1]
)
plt.xlabel("Observed/Expected Ratio")
plt.title(f"Top Collocates of '{top10['target'].iloc[0]}'")
plt.tight_layout()
plt.show()
```

### Network Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a network of collocations
G = nx.Graph()

# Add the target word as a central node
target = "经济"
G.add_node(target, size=20)

# Add edges to significant collocates
significant = collocates[collocates["p_value"] < 0.01]
significant = significant.sort_values("ratio_local", ascending=False).head(15)

for _, row in significant.iterrows():
    collocate = row["collocate"]
    weight = row["ratio_local"]
    G.add_node(collocate, size=10)
    G.add_edge(target, collocate, weight=weight)

# Draw the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)

# Draw nodes with different sizes
node_sizes = [G.nodes[node]["size"] * 50 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

# Draw edges with weights affecting width
edge_weights = [G[u][v]["weight"] / 5 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=12)

plt.axis("off")
plt.title(f"Collocation Network for '{target}'")
plt.tight_layout()
plt.show()
```

## Co-occurrence Matrix

qhChina also provides a function to compute co-occurrence matrices:

```python
from qhchina.analytics.collocations import cooc_matrix

# Get co-occurrence matrix
context_words = ["经济", "发展", "科技", "创新", "改革", "政策"]
target_matrix, row_labels, col_labels = cooc_matrix(
    tokenized_docs, 
    context_words=context_words,
    window_size=5
)
```

### Matrix Parameters

| Parameter | Description |
|-----------|-------------|
| `documents` | List of tokenized documents |
| `method` | Method to use: 'window' (default) or 'document' |
| `horizon` | Context window size (only with method='window') |
| `min_abs_count` | Minimum absolute count for inclusion |
| `min_doc_count` | Minimum document count for inclusion |
| `vocab_size` | Maximum vocabulary size |
| `binary` | Whether to count co-occurrences as binary (0/1) |
| `as_dataframe` | Return as pandas DataFrame (default: True) |
| `vocab` | Predefined vocabulary to use |
| `use_sparse` | Use sparse matrix for memory efficiency |

## Practical Examples

### Comparing Collocations Across Corpora

```python
# Find collocates in two different corpora
collocates_corpus1 = find_collocates(
    sentences=corpus1_sentences,
    target_words=["经济"],
    as_dataframe=True
)

collocates_corpus2 = find_collocates(
    sentences=corpus2_sentences,
    target_words=["经济"],
    as_dataframe=True
)

# Merge the dataframes to compare
collocates_corpus1["corpus"] = "Corpus 1"
collocates_corpus2["corpus"] = "Corpus 2"
combined = pd.concat([collocates_corpus1, collocates_corpus2])

# Find collocates that appear in both corpora
collocates1 = set(collocates_corpus1["collocate"])
collocates2 = set(collocates_corpus2["collocate"])
common_collocates = collocates1.intersection(collocates2)

# Compare the strength of association in both corpora
comparison = combined[combined["collocate"].isin(common_collocates)]
pivot = comparison.pivot(index="collocate", columns="corpus", values="ratio_local")
```

### Tracking Collocations Over Time

```python
# Assume we have corpora from different time periods
periods = ["1980s", "1990s", "2000s", "2010s"]
period_data = {period: sentences_for_period for period, sentences_for_period in zip(periods, all_period_sentences)}

# Track collocations of a term over time
target_word = "改革"
collocations_over_time = {}

for period, sentences in period_data.items():
    collocates = find_collocates(
        sentences=sentences,
        target_words=[target_word],
        as_dataframe=True
    )
    collocations_over_time[period] = collocates.sort_values("p_value").head(10)["collocate"].tolist()

# See which collocates appear in multiple periods
all_collocates = set()
for period_collocates in collocations_over_time.values():
    all_collocates.update(period_collocates)

presence_matrix = {collocate: [] for collocate in all_collocates}
for period in periods:
    for collocate in all_collocates:
        presence_matrix[collocate].append(collocate in collocations_over_time[period])

# Convert to DataFrame for easy viewing
collocate_tracking = pd.DataFrame(presence_matrix, index=periods).T
```

## Performance Considerations

- For large corpora, consider processing sentences in batches
- When analyzing multiple target words, the computation time increases linearly with the number of targets
- The sentence-based method is generally faster than the window-based method

## References

1. Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. Computational linguistics, 16(1), 22-29.
2. Evert, S. (2008). Corpora and collocations. Corpus linguistics. An international handbook, 2, 1212-1248. 