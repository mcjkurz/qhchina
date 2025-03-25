"""Analytics module for text and vector operations.

This module provides tools for:
- Collocation analytics
- Corpus comparison
- Vector operations and projections
- BERT-based modeling and classification
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
)

# Word2Vec operations   
from .word2vec import (
    Word2Vec, 
    TemporalWord2Vec
)

# BERT modeling and classification
from .modeling import (
    train_bert_classifier,
    evaluate,
    TextDataset,
    set_device,
    predict,
    bert_encode,
    semantic_change,
    align_vectors,
    visualize_semantic_trajectory,
    visualize_semantic_trajectory_full,
    make_datasets,
    add_corpus_tags,
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
    'Word2Vec', 'TemporalWord2Vec',
    # Modeling
    'train_bert_classifier', 'evaluate', 'TextDataset', 'set_device', 'predict', 'bert_encode', 
    'semantic_change', 'align_vectors', 'visualize_semantic_trajectory', 'visualize_semantic_trajectory_full',
    'make_datasets', 'add_corpus_tags',
]