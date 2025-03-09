from .collocations import find_collocates, cooc_matrix
from .corpora import compare_corpora
from .vectors import (project_2d, 
                      project_bias,
                      cosine_similarity,
                      get_bias_direction,
                      calculate_bias)
from .modeling import train_bert_classifier, evaluate, TextDataset, set_device