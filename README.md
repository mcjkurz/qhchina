# QHChina Lab

**Quantitative Humanities China Lab** - A research lab and Python package for NLP tasks related to Chinese text analysis.

## Repository Structure

This repository contains:

1. **Python Package**: The `qhchina` package for Chinese text analysis in humanities research
2. **Lab Website**: The QHChina Lab website located in the `/docs` folder with information about our research, projects, resources, and documentation

## Python Package Features

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

## Website

The lab website is built with Jekyll and includes:

- Information about the QHChina Lab and our research
- Project descriptions and updates
- Resources for Chinese humanities research
- Comprehensive documentation for the qhchina package
- Blog with updates and announcements

To run the website locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

## Documentation

For complete API documentation and tutorials, visit:
https://mcjkurz.github.io/qhchina/

## License

This project is licensed under the MIT License - see the LICENSE file for details.