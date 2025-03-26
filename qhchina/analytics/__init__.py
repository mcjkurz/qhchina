"""Analytics module for text and vector operations.

This module provides tools for:
- Collocation analytics
- Corpus comparison
- Vector operations and projections
- BERT-based modeling and classification
- Topic modeling
"""

# Collocation analytics
from .collocations import (
    find_collocates,
    cooc_matrix,
)

# Corpus comparison
from .corpora import compare_corpora

# Vector operations
from .vectors import (
    project_2d,
    project_bias,
    cosine_similarity,
    get_bias_direction,
    calculate_bias,
    most_similar,
    align_vectors
)

# Word2Vec operations   
from .word2vec import (
    Word2Vec, 
    TempRefWord2Vec
)

# BERT modeling and classification
from .modeling import (
    train_bert_classifier,
    evaluate,
    TextDataset,
    get_device,
    predict,
    bert_encode,
    make_datasets,
)

# Topic modeling
from .topicmodels import (
    LDAGibbsSampler
)

# Cython extension compilation
from .compile_extensions import (
    compile_cython_extensions
)

# Make all functions available at module level
__all__ = [
    # Collocations
    'find_collocates', 'cooc_matrix', '_calculate_collocations_window', '_calculate_collocations_sentence',
    # Corpora
    'compare_corpora',
    # Vectors
    'project_2d', 'project_bias', 'cosine_similarity', 'get_bias_direction', 'calculate_bias', 'most_similar',
    # Word2Vec
    'Word2Vec', 'TempRefWord2Vec',
    # Modeling
    'train_bert_classifier', 'evaluate', 'TextDataset', 'get_device', 'predict', 'bert_encode', 
    'align_vectors', 'make_datasets',
    # Topic modeling
    'LDAGibbsSampler',
    # Cython extensions
    'compile_cython_extensions',
    # Examples
    'lda_example', 'load_and_save_example',
]