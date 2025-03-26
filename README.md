# QHChina

**Quantitative Humanities China Lab** - A Python package for NLP tasks related to Chinese text analysis.

## Features

- **Collocation Analysis**: Find significant word co-occurrences in text
- **Corpus Comparison**: Statistically compare different corpora
- **Word Embeddings**: Work with Word2Vec and other embedding models
- **Text Classification**: BERT-based classification and analysis
- **Topic Modeling**: Fast LDA implementation with Cython acceleration

## Installation

```bash
pip install qhchina
```

## Usage Examples

### Topic Modeling with LDA

```python
from qhchina.analytics import LDAGibbsSampler

# Each document is a list of tokens
documents = [
    ["word1", "word2", "word3"],
    ["word2", "word4", "word5"],
    # ...
]

# Initialize and train the model
lda = LDAGibbsSampler(
    n_topics=10,
    iterations=500
)
lda.fit(documents)

# Get top words for each topic
for i, topic in enumerate(lda.get_topic_words(10)):
    print(f"Topic {i}: {[word for word, _ in topic]}")
```

For more examples, see the module documentation.

## Documentation

For complete API documentation and tutorials, visit:
https://mcjkurz.github.io/qhchina/

## License

This project is licensed under the MIT License - see the LICENSE file for details.