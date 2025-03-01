import numpy as np
from scipy.sparse import coo_matrix
from collections import Counter

# Initialize co-occurrence matrix
n = len(word_to_index)

if use_sparse:
    # For sparse matrix, we'll use a dictionary to accumulate co-occurrence counts
    # to avoid duplicate entries in the sparse matrix
    cooc_dict = {}
    
    # Function to update co-occurrence counts
    def update_cooc(word1_idx, word2_idx, count=1):
        key = (word1_idx, word2_idx)
        if binary:
            cooc_dict[key] = 1
        else:
            cooc_dict[key] = cooc_dict.get(key, 0) + count
else:
    # For dense matrix, initialize with zeros
    cooc_matrix_array = np.zeros((n, n), dtype=int)
    
    # Function to update co-occurrence counts
    def update_cooc(word1_idx, word2_idx, count=1):
        if binary:
            cooc_matrix_array[word1_idx, word2_idx] = 1
        else:
            cooc_matrix_array[word1_idx, word2_idx] += count

# Calculate co-occurrences
if method == 'window':
    for document in filtered_documents:
        for i, word1 in enumerate(document):
            idx1 = word_to_index[word1]
            start = max(0, i - horizon)
            end = min(len(document), i + horizon + 1)
            
            # Get context words (excluding the word itself)
            context_words = document[start:i] + document[i+1:end]
            
            if binary:
                # For binary counting, we only need to count each unique context word once
                # Create a set of unique context words
                unique_context_words = set(context_words)
                
                # Update co-occurrence counts for each unique context word
                for word2 in unique_context_words:
                    idx2 = word_to_index[word2]
                    update_cooc(idx1, idx2, 1)
            else:
                # For frequency counting, count each occurrence
                for word2 in context_words:
                    idx2 = word_to_index[word2]
                    update_cooc(idx1, idx2, 1)

elif method == 'document':
    for document in filtered_documents:
        # Pre-count words in document for efficiency
        doc_word_counts = Counter(document)
        unique_words = set(document)
        
        for word1 in unique_words:
            idx1 = word_to_index[word1]
            for word2 in unique_words:
                if word2 != word1:
                    idx2 = word_to_index[word2]
                    if binary:
                        update_cooc(idx1, idx2, 1)
                    else:
                        update_cooc(idx1, idx2, doc_word_counts[word2])

# Create the sparse matrix if needed
if use_sparse:
    # Convert the dictionary to COO format
    row_indices = []
    col_indices = []
    data_values = []
    
    for (i, j), value in cooc_dict.items():
        row_indices.append(i)
        col_indices.append(j)
        data_values.append(value)
        
    cooc_matrix_array = coo_matrix(
        (data_values, (row_indices, col_indices)), 
        shape=(n, n)
    ).tocsr()  # Convert to CSR for efficient operations

# Return results based on parameters 