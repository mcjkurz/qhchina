"""qhchina: A package for Chinese text analysis and educational tools

Core analysis functionality is available directly.
For more specialized functions, import from specific modules:
- qhchina.analysis: Text analysis and modeling
- qhchina.preprocessing: Text preprocessing utilities
- qhchina.helpers: Utility functions
- qhchina.educational: Educational visualization tools
"""

__version__ = "0.0.26"

# Most commonly used analysis functions
from .analysis import (
    # Collocation analysis
    find_collocates,
    cooc_matrix,
    # Corpus comparison
    compare_corpora,
    # Vector operations
    project_2d,
    calculate_bias,
    project_bias,
    get_bias_direction,
    cosine_similarity,
    most_similar,
)

# BERT-related functionality
from .analysis import (
    train_bert_classifier,
    evaluate,
    TextDataset,
    set_device,
    predict,
    bert_encode,
)

# Text preprocessing
from .preprocessing import split_into_chunks

# Helper functions
from .helpers import (
    install_package,
    load_texts,
    load_fonts,
    set_font,
)

# Educational tools
from .educational import show_vectors

# For explicit access to submodules
from . import analysis
from . import preprocessing
from . import helpers
from . import educational