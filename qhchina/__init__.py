# qhchina/__init__.py
__version__ = "0.0.23"

from .analysis import (find_collocates, 
                       cooc_matrix, 
                       compare_corpora, 
                       project_2d,
                       calculate_bias,
                       project_bias,
                       get_bias_direction,
                       cosine_similarity, 
                       train_bert_classifier,
                       evaluate)
from .preprocessing import split_into_chunks
from .helpers import (install_package, 
                      load_texts, 
                      load_fonts, 
                      set_font)
from .educational import show_vectors