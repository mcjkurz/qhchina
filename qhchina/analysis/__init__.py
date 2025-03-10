"""Analysis module for text and vector operations.

This module provides tools for:
- Collocation analysis
- Corpus comparison
- Vector operations and projections
- BERT-based modeling and classification
"""

# Collocation analysis
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

# BERT modeling and classification
from .modeling import (
    train_bert_classifier,
    evaluate,
    TextDataset,
    set_device,
    predict,
    bert_encode,
)