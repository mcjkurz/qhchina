# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Collocation counting operations implemented in Cython.

This module provides implementations of the core counting operations
for collocation analysis using Cython.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from cython.operator cimport dereference, postincrement

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.int64_t LONG_t

def build_vocabulary(list tokenized_sentences):
    """
    Build vocabulary from tokenized sentences.
    
    Builds word-to-index and index-to-word mappings from tokenized sentences.
    
    Args:
        tokenized_sentences: List of tokenized sentences (list of lists of strings)
        
    Returns:
        tuple: (word2idx, idx2word) - dictionaries for bidirectional mapping
    """
    cdef dict word2idx = {}
    cdef dict idx2word = {}
    cdef int idx = 0
    cdef list sentence
    cdef str word
    cdef int i
    cdef int n_sentences = len(tokenized_sentences)
    
    # Build word2idx mapping
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
    
    return word2idx, idx2word

def convert_sentences_to_indices(list tokenized_sentences, dict word2idx, int max_sentence_length=256):
    """
    Convert tokenized sentences to indexed arrays.
    
    Converts tokenized sentences to padded integer arrays for efficient processing.
    
    Args:
        tokenized_sentences: List of tokenized sentences (list of lists of strings)
        word2idx: Dictionary mapping words to integer indices
        max_sentence_length: Maximum sentence length to pad to (default 256)
        
    Returns:
        tuple: (sentences_indices, sentence_lengths)
            - sentences_indices: 2D numpy array (n_sentences, max_length) of int32
            - sentence_lengths: 1D numpy array (n_sentences,) of int32
    """
    cdef int n_sentences = len(tokenized_sentences)
    cdef int i, j, actual_len, sent_len
    cdef int max_length = 0
    cdef list sentence
    cdef str word
    
    # First pass: find max length
    for i in range(n_sentences):
        sent_len = len(tokenized_sentences[i])
        if sent_len > max_length:
            max_length = sent_len
    
    # Cap max_length to avoid memory bloat
    if max_length > max_sentence_length:
        max_length = max_sentence_length
    
    # Allocate arrays
    cdef np.ndarray[INT_t, ndim=2] sentences_indices = np.zeros((n_sentences, max_length), dtype=np.int32)
    cdef np.ndarray[INT_t, ndim=1] sentence_lengths = np.zeros(n_sentences, dtype=np.int32)
    
    # Second pass: convert to indices
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        actual_len = len(sentence)
        if actual_len > max_length:
            actual_len = max_length
        sentence_lengths[i] = actual_len
        
        for j in range(actual_len):
            word = sentence[j]
            sentences_indices[i, j] = word2idx[word]
    
    return sentences_indices, sentence_lengths

def calculate_collocations_window(list tokenized_sentences, list target_words, 
                                  int horizon=5, int max_sentence_length=256, 
                                  int batch_size=10000):
    """
    Window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List of target words
        horizon: Window size on each side
        max_sentence_length: Maximum sentence length (default 256)
        batch_size: Batch size for processing (default 10000)
        
    Returns:
        tuple: (T_count, candidate_counts, token_counter, total_tokens, word2idx, idx2word, target_indices)
    """
    # All variable declarations at the top
    cdef dict word2idx, idx2word
    cdef int vocab_size, n_sentences, n_targets, n_batches
    cdef int batch_start, batch_end, batch_idx, t_idx, candidate_idx
    cdef list batch_sentences, target_words_filtered
    cdef str word
    cdef INT_t[:, ::1] batch_indices
    cdef INT_t[::1] batch_lengths, target_indices_array
    cdef LONG_t[::1] T_count_batch, token_counter_batch
    cdef LONG_t[:, ::1] candidate_counts_batch
    cdef LONG_t total_tokens_batch
    cdef np.ndarray[LONG_t, ndim=1] T_count_total, token_counter_total
    cdef np.ndarray[LONG_t, ndim=2] candidate_counts_total
    cdef np.ndarray[INT_t, ndim=1] target_indices
    cdef LONG_t total_tokens = 0
    
    # Build vocabulary
    word2idx, idx2word = build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    n_sentences = len(tokenized_sentences)
    
    # Filter target words and convert to indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if len(target_words_filtered) == 0:
        return None, None, None, 0, word2idx, idx2word, np.array([], dtype=np.int32)
    
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    target_indices_array = target_indices
    n_targets = len(target_indices)
    
    # Initialize accumulation arrays
    T_count_total = np.zeros(n_targets, dtype=np.int64)
    candidate_counts_total = np.zeros((n_targets, vocab_size), dtype=np.int64)
    token_counter_total = np.zeros(vocab_size, dtype=np.int64)
    
    # Process batches - all in Cython
    n_batches = (n_sentences + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_sentences)
        batch_sentences = tokenized_sentences[batch_start:batch_end]
        
        # Convert batch to indices
        batch_indices, batch_lengths = convert_sentences_to_indices(
            batch_sentences, word2idx, max_sentence_length
        )
        
        # Calculate counts for this batch
        T_count_batch, candidate_counts_batch, token_counter_batch, total_tokens_batch = calculate_window_counts(
            batch_indices, batch_lengths, target_indices_array, horizon, vocab_size
        )
        
        # Accumulate results in Cython
        for t_idx in range(n_targets):
            T_count_total[t_idx] += T_count_batch[t_idx]
            for candidate_idx in range(vocab_size):
                candidate_counts_total[t_idx, candidate_idx] += candidate_counts_batch[t_idx, candidate_idx]
        
        for candidate_idx in range(vocab_size):
            token_counter_total[candidate_idx] += token_counter_batch[candidate_idx]
        
        total_tokens += total_tokens_batch
    
    return T_count_total, candidate_counts_total, token_counter_total, total_tokens, word2idx, idx2word, target_indices

def calculate_collocations_sentence(list tokenized_sentences, list target_words,
                                    int max_sentence_length=256, int batch_size=10000):
    """
    Sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List of target words
        max_sentence_length: Maximum sentence length (default 256)
        batch_size: Batch size for processing (default 10000)
        
    Returns:
        tuple: (candidate_sentences, sentences_with_token, total_sentences, word2idx, idx2word, target_indices)
    """
    # All variable declarations at the top
    cdef dict word2idx, idx2word
    cdef int vocab_size, n_sentences, n_targets, n_batches
    cdef int batch_start, batch_end, batch_idx, t_idx, candidate_idx
    cdef list batch_sentences, target_words_filtered
    cdef str word
    cdef INT_t[:, ::1] batch_indices
    cdef INT_t[::1] batch_lengths, target_indices_array
    cdef LONG_t[:, ::1] candidate_sentences_batch
    cdef LONG_t[::1] sentences_with_token_batch
    cdef LONG_t n_sentences_batch
    cdef np.ndarray[LONG_t, ndim=2] candidate_sentences_total
    cdef np.ndarray[LONG_t, ndim=1] sentences_with_token_total
    cdef np.ndarray[INT_t, ndim=1] target_indices
    cdef LONG_t total_sentences = 0
    
    # Build vocabulary
    word2idx, idx2word = build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    n_sentences = len(tokenized_sentences)
    
    # Filter target words and convert to indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if len(target_words_filtered) == 0:
        return None, None, 0, word2idx, idx2word, np.array([], dtype=np.int32)
    
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    target_indices_array = target_indices
    n_targets = len(target_indices)
    
    # Initialize accumulation arrays
    candidate_sentences_total = np.zeros((n_targets, vocab_size), dtype=np.int64)
    sentences_with_token_total = np.zeros(vocab_size, dtype=np.int64)
    
    # Process batches - all in Cython
    n_batches = (n_sentences + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_sentences)
        batch_sentences = tokenized_sentences[batch_start:batch_end]
        
        # Convert batch to indices
        batch_indices, batch_lengths = convert_sentences_to_indices(
            batch_sentences, word2idx, max_sentence_length
        )
        
        # Calculate counts for this batch
        candidate_sentences_batch, sentences_with_token_batch, n_sentences_batch = calculate_sentence_counts(
            batch_indices, batch_lengths, target_indices_array, vocab_size
        )
        
        # Accumulate results in Cython
        for t_idx in range(n_targets):
            for candidate_idx in range(vocab_size):
                candidate_sentences_total[t_idx, candidate_idx] += candidate_sentences_batch[t_idx, candidate_idx]
        
        for candidate_idx in range(vocab_size):
            sentences_with_token_total[candidate_idx] += sentences_with_token_batch[candidate_idx]
        
        total_sentences += n_sentences_batch
    
    return candidate_sentences_total, sentences_with_token_total, total_sentences, word2idx, idx2word, target_indices

def calculate_window_counts(
    INT_t[:, ::1] sentences_indices,
    INT_t[::1] sentence_lengths,
    INT_t[::1] target_indices,
    int horizon,
    int vocab_size
):
    """
    Window-based collocation counting using sparse C++ hash maps.
    
    Args:
        sentences_indices: 2D array of sentence tokens as indices (n_sentences, max_length)
        sentence_lengths: Actual length of each sentence (n_sentences,)
        target_indices: Indices of target words (n_targets,)
        horizon: Window size on each side
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (T_count, candidate_in_context, token_counter, total_tokens)
            - T_count: For each target, count of positions with target in context
            - candidate_in_context: For each target, count of each candidate in those positions
            - token_counter: Global count of each token
            - total_tokens: Total number of token positions
    """
    cdef int n_sentences = sentences_indices.shape[0]
    cdef int n_targets = target_indices.shape[0]
    cdef int i, j, s, t, start, end, doc_len, token_idx, target_idx, context_idx
    cdef LONG_t total_tokens = 0
    cdef char* is_target
    cdef char* context_has_target
    cdef INT_t* target_idx_map
    
    # T_count remains as NumPy array (small, based on n_targets)
    cdef np.ndarray[LONG_t, ndim=1] T_count_arr = np.zeros(n_targets, dtype=np.int64)
    cdef LONG_t[::1] T_count = T_count_arr
    
    # Declare C++ hash maps for sparse counting
    cdef unordered_map[INT_t, LONG_t] token_counter_map
    cdef vector[unordered_map[INT_t, LONG_t]] candidate_counts_maps
    
    # Initialize vector of maps (one per target)
    candidate_counts_maps.resize(n_targets)
    
    # Build target lookup table for O(1) checking
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    # Build reverse mapping: vocab_idx -> target_position for O(1) lookup
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    # Allocate buffer for checking which targets are in context
    context_has_target = <char*>malloc(n_targets * sizeof(char))
    if context_has_target == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for context_has_target")
    
    # Main counting loop - release GIL for parallel processing potential
    # Use C++ hash maps for sparse counting - better cache performance
    with nogil:
        for s in range(n_sentences):
            doc_len = sentence_lengths[s]
            
            for i in range(doc_len):
                token_idx = sentences_indices[s, i]
                total_tokens += 1
                token_counter_map[token_idx] += 1
                
                # Define window bounds (excluding center token) - inline comparisons
                start = i - horizon if i >= horizon else 0
                end = i + horizon + 1 if i + horizon + 1 <= doc_len else doc_len
                
                # Reset context target flags
                memset(context_has_target, 0, n_targets * sizeof(char))
                
                # Check which targets are in this context
                for j in range(start, end):
                    if j == i:  # Skip center token
                        continue
                    
                    context_idx = sentences_indices[s, j]
                    if context_idx < vocab_size and is_target[context_idx]:
                        # O(1) lookup instead of O(n_targets) scan
                        context_has_target[target_idx_map[context_idx]] = 1
                
                # For each target that was in the context, count this token position
                for t in range(n_targets):
                    if context_has_target[t]:
                        T_count[t] += 1
                        candidate_counts_maps[t][token_idx] += 1
    
    # Free allocated memory
    free(is_target)
    free(target_idx_map)
    free(context_has_target)
    
    # Transfer sparse map data to dense NumPy arrays for return
    cdef np.ndarray[LONG_t, ndim=2] candidate_counts_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] token_counter_arr = np.zeros(vocab_size, dtype=np.int64)
    cdef LONG_t[:, ::1] candidate_counts = candidate_counts_arr
    cdef LONG_t[::1] token_counter = token_counter_arr
    
    # Populate token_counter array from map
    cdef unordered_map[INT_t, LONG_t].iterator it_token = token_counter_map.begin()
    while it_token != token_counter_map.end():
        token_idx = dereference(it_token).first
        if token_idx < vocab_size:
            token_counter[token_idx] = dereference(it_token).second
        postincrement(it_token)
    
    # Populate candidate_counts array from maps
    cdef unordered_map[INT_t, LONG_t].iterator it_cand
    for t in range(n_targets):
        it_cand = candidate_counts_maps[t].begin()
        while it_cand != candidate_counts_maps[t].end():
            token_idx = dereference(it_cand).first
            if token_idx < vocab_size:
                candidate_counts[t, token_idx] = dereference(it_cand).second
            postincrement(it_cand)
    
    return T_count_arr, candidate_counts_arr, token_counter_arr, total_tokens


def calculate_sentence_counts(
    INT_t[:, ::1] sentences_indices,
    INT_t[::1] sentence_lengths,
    INT_t[::1] target_indices,
    int vocab_size
):
    """
    Sentence-based collocation counting using sparse C++ hash maps.
    
    Args:
        sentences_indices: 2D array of sentence tokens as indices (n_sentences, max_length)
        sentence_lengths: Actual length of each sentence (n_sentences,)
        target_indices: Indices of target words (n_targets,)
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (candidate_in_sentences, sentences_with_token, n_sentences)
            - candidate_in_sentences: For each target, count of sentences containing both
            - sentences_with_token: Count of sentences containing each token
            - n_sentences: Total number of sentences
    """
    cdef int n_sentences = sentences_indices.shape[0]
    cdef int n_targets = target_indices.shape[0]
    cdef int max_doc_len = sentences_indices.shape[1]
    cdef int i, s, t, doc_len, token_idx, unique_count, idx
    cdef char* is_target
    cdef char* token_in_sentence
    cdef char* target_in_sentence
    cdef INT_t* unique_tokens
    cdef INT_t* target_idx_map
    
    # Declare C++ hash maps for sparse counting
    cdef unordered_map[INT_t, LONG_t] sentences_with_token_map
    cdef vector[unordered_map[INT_t, LONG_t]] candidate_sentences_maps
    
    # Initialize vector of maps (one per target)
    candidate_sentences_maps.resize(n_targets)
    
    # Build target lookup table
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    # Build reverse mapping for O(1) target lookup
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    # Allocate buffers for tracking tokens in current sentence
    token_in_sentence = <char*>malloc(vocab_size * sizeof(char))
    if token_in_sentence == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for token_in_sentence")
    
    target_in_sentence = <char*>malloc(n_targets * sizeof(char))
    if target_in_sentence == NULL:
        free(is_target)
        free(target_idx_map)
        free(token_in_sentence)
        raise MemoryError("Failed to allocate memory for target_in_sentence")
    
    # Buffer to store unique token indices in current sentence (avoid full vocab scan)
    unique_tokens = <INT_t*>malloc(max_doc_len * sizeof(INT_t))
    if unique_tokens == NULL:
        free(is_target)
        free(target_idx_map)
        free(token_in_sentence)
        free(target_in_sentence)
        raise MemoryError("Failed to allocate memory for unique_tokens")
    
    # Main counting loop - release GIL for parallel processing potential
    # Use C++ hash maps for sparse counting - better cache performance
    with nogil:
        for s in range(n_sentences):
            doc_len = sentence_lengths[s]
            
            # Reset target flags (small buffer, OK to memset)
            memset(target_in_sentence, 0, n_targets * sizeof(char))
            unique_count = 0
            
            # Mark unique tokens in this sentence and track them
            for i in range(doc_len):
                token_idx = sentences_indices[s, i]
                if token_idx < vocab_size:
                    if not token_in_sentence[token_idx]:
                        token_in_sentence[token_idx] = 1
                        unique_tokens[unique_count] = token_idx
                        unique_count += 1
                        
                        # Check if this token is a target (O(1) lookup)
                        if is_target[token_idx]:
                            target_in_sentence[target_idx_map[token_idx]] = 1
            
            # Update global sentence counts using hash map - only iterate unique tokens
            for i in range(unique_count):
                token_idx = unique_tokens[i]
                sentences_with_token_map[token_idx] += 1
            
            # For each target in sentence, count all unique tokens using hash maps
            for t in range(n_targets):
                if target_in_sentence[t]:
                    for i in range(unique_count):
                        token_idx = unique_tokens[i]
                        candidate_sentences_maps[t][token_idx] += 1
            
            # Clear only the tokens we actually used (much faster than memset entire vocab)
            for i in range(unique_count):
                token_in_sentence[unique_tokens[i]] = 0
    
    # Free allocated memory
    free(is_target)
    free(target_idx_map)
    free(token_in_sentence)
    free(target_in_sentence)
    free(unique_tokens)
    
    # Transfer sparse map data to dense NumPy arrays for return
    cdef np.ndarray[LONG_t, ndim=2] candidate_sentences_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] sentences_with_token_arr = np.zeros(vocab_size, dtype=np.int64)
    cdef LONG_t[:, ::1] candidate_sentences = candidate_sentences_arr
    cdef LONG_t[::1] sentences_with_token = sentences_with_token_arr
    
    # Populate sentences_with_token array from map
    cdef unordered_map[INT_t, LONG_t].iterator it_token = sentences_with_token_map.begin()
    while it_token != sentences_with_token_map.end():
        token_idx = dereference(it_token).first
        if token_idx < vocab_size:
            sentences_with_token[token_idx] = dereference(it_token).second
        postincrement(it_token)
    
    # Populate candidate_sentences array from maps
    cdef unordered_map[INT_t, LONG_t].iterator it_cand
    for t in range(n_targets):
        it_cand = candidate_sentences_maps[t].begin()
        while it_cand != candidate_sentences_maps[t].end():
            token_idx = dereference(it_cand).first
            if token_idx < vocab_size:
                candidate_sentences[t, token_idx] = dereference(it_cand).second
            postincrement(it_cand)
    
    return candidate_sentences_arr, sentences_with_token_arr, n_sentences

