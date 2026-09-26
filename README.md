# qhChina

A Python toolkit for computational analysis of Chinese texts in humanities research.

## Modules

- **Preprocessing**: Chinese text segmentation with multiple backends (spaCy, Jieba, BERT, LLM), plus normalization helpers.
- **Analytics**: Core analysis tools including:
  - **Word Embeddings**: `Word2Vec`, `TempRefWord2Vec`, `DynamicWord2Vec`, and `GloVe`.
  - **Topic Modeling**: `LDAGibbsSampler` with Gibbs sampling and Cython acceleration.
  - **Stylometry**: Authorship attribution, similarity analysis, and document clustering.
  - **Collocations**: Statistical collocation analysis, co-occurrence matrices, and collocate comparison across corpora.
  - **Text Reuse**: Shared sequence detection across documents/corpora.
  - **Vector Utilities**: Similarity, distance, alignment, and visualization helpers.
- **Educational**: Interactive learning tools and visualizations for basic NLP concepts.
- **Helpers**: Font management, text loading, stopword utilities, and corpus download helpers.

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

Issues can be posted here: [qhchina issues](https://github.com/mcjkurz/qhchina/issues).

## Tests

```bash
pip install pytest
pytest tests/
```

## License

MIT License
