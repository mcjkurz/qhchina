# qhChina

A Python toolkit for computational analysis of Chinese texts in humanities research.

## Features

- **Preprocessing**: Chinese text segmentation with multiple backends (spaCy, Jieba, BERT, LLM)
- **Word Embeddings**: Word2Vec training and temporal semantic change analysis (TempRefWord2Vec)
- **Topic Modeling**: LDA with Gibbs sampling and Cython acceleration
- **Stylometry**: Authorship attribution and document clustering
- **Collocations**: Statistical collocation analysis and co-occurrence matrices
- **Corpus Comparison**: Identify significant vocabulary differences between corpora
- **Helpers**: CJK font management, text loading, stopwords

## Installation

```bash
pip install qhchina
```

## Building from Source

```bash
git clone https://github.com/mcjkurz/qhchina.git
cd qhchina
pip install -e .
```

This will compile the Cython extensions and install the package in editable mode.

## Documentation

Full documentation and examples: [www.qhchina.org/docs/](https://www.qhchina.org/docs/)

## Tests

```bash
pip install pytest
pytest tests/
```

## License

MIT License
