# AI Agent Guide

> Toolkit for computational analysis of Chinese texts in humanities research.

## Quick Orientation

| Directory | Purpose |
|-----------|---------|
| `qhchina/` | Main package |
| `qhchina/analytics/` | Core analytics (LDA, Word2Vec, stylometry, collocations) |
| `qhchina/analytics/cython_ext/` | Cython extensions (`.pyx`) for performance |
| `qhchina/preprocessing/` | Text segmentation (spaCy, Jieba, BERT, LLM backends) |
| `qhchina/helpers/` | Utilities (fonts, text loading, stopwords) |
| `tests/` | Pytest test suite |

## Key Classes

- **`Corpus`** (`corpus.py`): Core data structure. Collection of `Document` objects. Iterable (yields token lists). Supports filtering, grouping, serialization.
- **`Word2Vec`** (`analytics/word2vec.py`): Word embeddings with CBOW/Skip-gram.
- **`TempRefWord2Vec`** (`analytics/word2vec.py`): Temporal semantic change analysis.
- **`LDAGibbsSampler`** (`analytics/topicmodels.py`): Topic modeling with Gibbs sampling.
- **`Stylometry`** (`analytics/stylometry.py`): Authorship attribution, corpus comparison.
- **`SegmentationWrapper`** (`preprocessing/segmentation.py`): Chinese text segmentation.

## Important Patterns

### Cython Extensions
Performance-critical code in `.pyx` files with Python fallbacks:
```python
try:
    from .cython_ext.module import func
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    # Python fallback implementation
```

### Random Seed Management
Global seed via `config.py`. Use `get_rng()` for isolated RNG instances:
```python
from qhchina.config import get_rng, set_random_seed
rng = get_rng("my_module")  # Isolated, reproducible RNG
```

### Logging
Module-level loggers: `logger = logging.getLogger("qhchina.module_name")`

### Type Hints
Uses modern syntax: `int | None`, `list[str]`, `TYPE_CHECKING` for conditional imports.

## Development Commands

```bash
# Always use venv
./venv/bin/python -m pytest tests/           # Run tests
./venv/bin/python setup.py build_ext --inplace  # Build Cython
./venv/bin/pip install -e .                  # Dev install
```

## Build System

- `pyproject.toml`: Package metadata, dependencies (numpy, scipy, scikit-learn, pandas)
- `setup.py`: Cython extension compilation (3 extensions: lda_sampler, word2vec, collocations)
- Python 3.11+ required

## Test Data

- `tests/texts/`: Historical Chinese texts (宋史.txt, 明史.txt) for realistic testing
- `tests/conftest.py`: Shared fixtures (`sample_corpus`, `sample_tokenized`, etc.)

## Common Entry Points

```python
from qhchina import Corpus, load_text, load_stopwords, set_log_level
from qhchina.analytics import Word2Vec, LDAGibbsSampler, Stylometry
from qhchina.preprocessing import create_segmenter
```
