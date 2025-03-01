# qhchina/__init__.py
__version__ = "0.0.19"

from .analysis import (find_collocations, 
                       cooc_matrix, 
                       compare_corpora, 
                       project_2d, 
                       project_bias,
                       cosine_similarity)
from .preprocessing import split_into_chunks
from .helpers import (install_package, 
                      load_texts, 
                      load_fonts, 
                      set_font)
from .educational import show_vectors