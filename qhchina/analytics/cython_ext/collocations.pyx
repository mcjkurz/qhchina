# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Collocation counting operations implemented in Cython.

This module provides highly optimized implementations of core counting operations
for collocation analysis. Key optimizations include:
- C-level sentence encoding: Flattened contiguous buffer with offset array for O(1) access
- Zero dictionary access in hot loops: All counting operates on pure C-level integer arrays
- Active targets tracking: Window routine only iterates over targets actually seen in window
- Epoch-based uniqueness: O(1) duplicate detection without linear search
- Dense NumPy arrays for direct counting in nogil blocks (no sparse-to-dense conversion)
- GIL-free hot loops with while-based iteration for maximum performance
- Complete separation: Python dictionary access happens before any counting routine
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libcpp.vector cimport vector

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.int64_t LONG_t

# C-level struct for encoded corpus
cdef struct EncodedCorpus:
    INT_t* tokens        # Flattened token array
    LONG_t* offsets      # Sentence start offsets (n_sentences + 1)
    int n_sentences      # Number of sentences
    LONG_t total_tokens  # Total number of tokens

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

cdef EncodedCorpus* encode_corpus_to_c_buffer(list tokenized_sentences, dict word2idx, int max_sentence_length=256) except NULL:
    """
    Encode corpus into compact C-level buffers (flattened tokens + offsets).
    
    This function performs all word-to-index lookups upfront and stores the result
    in contiguous C memory for maximum cache efficiency. No per-sentence allocations.
    
    Args:
        tokenized_sentences: List of tokenized sentences (list of lists of strings)
        word2idx: Dictionary mapping words to indices
        max_sentence_length: Maximum sentence length to process (default 256)
        
    Returns:
        EncodedCorpus*: Pointer to encoded corpus structure (caller must free)
    """
    cdef int n_sentences = len(tokenized_sentences)
    cdef int i, j, sent_len, token_idx
    cdef LONG_t estimated_tokens = 0
    cdef list sentence
    cdef str word
    cdef object word2idx_get = word2idx.get
    cdef vector[INT_t] tokens_vec
    cdef vector[LONG_t] offsets_vec
    cdef LONG_t current_offset = 0
    cdef EncodedCorpus* corpus = NULL
    
    # Calculate precise token count for optimal reservation
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        sent_len = len(sentence)
        if sent_len > max_sentence_length:
            sent_len = max_sentence_length
        estimated_tokens += sent_len
    
    # Reserve exact space needed
    offsets_vec.reserve(n_sentences + 1)
    tokens_vec.reserve(estimated_tokens)
    
    # Build flattened token array and offsets
    offsets_vec.push_back(0)
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        sent_len = len(sentence)
        
        # Cap sentence length if needed
        if sent_len > max_sentence_length:
            sent_len = max_sentence_length
        
        # Convert words to indices and append to flat buffer
        for j in range(sent_len):
            word = sentence[j]
            token_idx = word2idx_get(word, -1)
            if token_idx >= 0:
                tokens_vec.push_back(token_idx)
                current_offset += 1
        
        # Store offset for next sentence
        offsets_vec.push_back(current_offset)
    
    # Allocate C struct
    corpus = <EncodedCorpus*>malloc(sizeof(EncodedCorpus))
    if corpus == NULL:
        raise MemoryError("Failed to allocate EncodedCorpus")
    
    # Allocate and copy token array
    corpus.total_tokens = <LONG_t>tokens_vec.size()
    corpus.tokens = <INT_t*>malloc(corpus.total_tokens * sizeof(INT_t))
    if corpus.tokens == NULL:
        free(corpus)
        raise MemoryError("Failed to allocate tokens array")
    
    for i in range(corpus.total_tokens):
        corpus.tokens[i] = tokens_vec[i]
    
    # Allocate and copy offsets array
    corpus.n_sentences = n_sentences
    corpus.offsets = <LONG_t*>malloc((n_sentences + 1) * sizeof(LONG_t))
    if corpus.offsets == NULL:
        free(corpus.tokens)
        free(corpus)
        raise MemoryError("Failed to allocate offsets array")
    
    for i in range(n_sentences + 1):
        corpus.offsets[i] = offsets_vec[i]
    
    return corpus

cdef void free_encoded_corpus(EncodedCorpus* corpus) nogil:
    """
    Free memory allocated for encoded corpus.
    
    Args:
        corpus: Pointer to encoded corpus structure
    """
    if corpus != NULL:
        if corpus.tokens != NULL:
            free(corpus.tokens)
        if corpus.offsets != NULL:
            free(corpus.offsets)
        free(corpus)

def calculate_collocations_window(list tokenized_sentences, list target_words, 
                                  int left_horizon=5, int right_horizon=5, int max_sentence_length=256):
    """
    Window-based collocation calculation.
    
    Encodes corpus into C-level buffers once, then processes them in fully optimized
    hot loops without any Python dictionary access. Uses active targets tracking.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List of target words
        left_horizon: Window size on the left side
        right_horizon: Window size on the right side
        max_sentence_length: Maximum sentence length (default 256)
        
    Returns:
        tuple: (T_count, candidate_counts, token_counter, total_tokens, word2idx, idx2word, target_indices)
    """
    # All variable declarations at the top
    cdef dict word2idx, idx2word
    cdef int vocab_size, n_targets
    cdef list target_words_filtered
    cdef INT_t[::1] target_indices_array
    cdef np.ndarray[INT_t, ndim=1] target_indices
    cdef EncodedCorpus* corpus = NULL
    cdef tuple result
    
    # Build vocabulary
    word2idx, idx2word = build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    
    # Filter target words and convert to indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if len(target_words_filtered) == 0:
        return None, None, None, 0, word2idx, idx2word, np.array([], dtype=np.int32)
    
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    target_indices_array = target_indices
    n_targets = len(target_indices)
    
    # Encode corpus to C-level buffers (single allocation, contiguous memory)
    corpus = encode_corpus_to_c_buffer(tokenized_sentences, word2idx, max_sentence_length)
    
    try:
        # Calculate counts from C-level buffers (no dictionary access, active targets optimization)
        result = calculate_window_counts(
            corpus, target_indices_array, left_horizon, right_horizon, vocab_size
        )
        return result[0], result[1], result[2], result[3], word2idx, idx2word, target_indices
    finally:
        # Always free the corpus memory
        free_encoded_corpus(corpus)

def calculate_collocations_sentence(list tokenized_sentences, list target_words, int max_sentence_length=256):
    """
    Optimized sentence-based collocation calculation.
    
    Encodes corpus into C-level buffers once, then processes them in fully optimized
    hot loops without any Python dictionary access. Uses epoch-based uniqueness detection.
    
    Args:
        tokenized_sentences: List of tokenized sentences (list of lists of strings)
        target_words: List of target words
        max_sentence_length: Maximum sentence length (default 256)
        
    Returns:
        tuple: (candidate_sentences, sentences_with_token, total_sentences, word2idx, idx2word, target_indices)
            - candidate_sentences: 2D array (n_targets, vocab_size) counting co-occurrences
            - sentences_with_token: 1D array (vocab_size,) counting sentences per token
            - total_sentences: Total number of sentences processed
            - word2idx: Dictionary mapping words to indices
            - idx2word: Dictionary mapping indices to words
            - target_indices: Array of target word indices
    """
    # Variable declarations
    cdef dict word2idx, idx2word
    cdef int vocab_size, n_targets
    cdef list target_words_filtered
    cdef np.ndarray[INT_t, ndim=1] target_indices
    cdef INT_t[::1] target_indices_array
    cdef EncodedCorpus* corpus = NULL
    cdef tuple result
    
    # Build vocabulary
    word2idx, idx2word = build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    
    # Filter target words and convert to indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if len(target_words_filtered) == 0:
        return None, None, 0, word2idx, idx2word, np.array([], dtype=np.int32)
    
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    target_indices_array = target_indices
    n_targets = len(target_indices)
    
    # Encode corpus to C-level buffers (single allocation, contiguous memory)
    corpus = encode_corpus_to_c_buffer(tokenized_sentences, word2idx, max_sentence_length)
    
    try:
        # Call optimized counting function with C-level buffers (epoch-based uniqueness)
        result = calculate_sentence_counts(
            corpus, target_indices_array, vocab_size, max_sentence_length, word2idx, idx2word, target_indices
        )
        return result
    finally:
        # Always free the corpus memory
        free_encoded_corpus(corpus)

cdef calculate_sentence_counts(
    EncodedCorpus* corpus,
    INT_t[::1] target_indices,
    int vocab_size,
    int max_sentence_length,
    dict word2idx,
    dict idx2word,
    np.ndarray[INT_t, ndim=1] target_indices_np
):
    """
    Core sentence counting logic operating on C-level encoded corpus.
    
    Processes corpus from contiguous C buffers without any Python dictionary access.
    Uses epoch-based uniqueness detection for O(1) duplicate checking with overflow guard.
    Uses dense NumPy arrays for counting directly in nogil blocks.
    All hot loops are GIL-free and operate purely on C-level memory.
    """
    cdef int n_targets = target_indices.shape[0]
    cdef int n_sentences = corpus.n_sentences
    cdef int s, t, i, j, token_idx, n_unique, n_targets_in_sent
    cdef LONG_t sent_start, sent_end, sent_len
    cdef char* is_target = NULL
    cdef INT_t* target_idx_map = NULL
    cdef INT_t* seen_epoch = NULL  # Epoch-based uniqueness tracking
    cdef INT_t current_epoch = 1
    cdef vector[INT_t] unique_tokens_buffer
    cdef vector[INT_t] targets_in_sent_buffer
    
    # Allocate all NumPy arrays upfront
    cdef np.ndarray[LONG_t, ndim=2] candidate_sentences_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] sentences_with_token_arr = np.zeros(vocab_size, dtype=np.int64)
    
    # Create typed memoryviews for direct access in nogil blocks
    cdef LONG_t[:, ::1] candidate_sentences = candidate_sentences_arr
    cdef LONG_t[::1] sentences_with_token = sentences_with_token_arr
    
    # Build target lookup table for O(1) checking
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    # Build reverse mapping: vocab_idx -> target_position (initialize to -1 for safety)
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    memset(target_idx_map, 0xFF, vocab_size * sizeof(INT_t))  # -1 for all entries
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    # Allocate epoch array for O(1) uniqueness detection
    seen_epoch = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if seen_epoch == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for seen_epoch")
    memset(seen_epoch, 0, vocab_size * sizeof(INT_t))
    
    # Reserve space for buffers to minimize reallocations
    # Use max_sentence_length since that's the maximum possible unique tokens in a sentence
    unique_tokens_buffer.reserve(max_sentence_length)
    # Reserve space for targets buffer (estimate: min of n_targets or max_sentence_length)
    targets_in_sent_buffer.reserve(n_targets if n_targets < max_sentence_length else max_sentence_length)
    
    # Main counting loop - Process C-level encoded corpus (fully nogil!)
    with nogil:
        for s in range(n_sentences):
            # Check for epoch overflow and reset if necessary
            if current_epoch == 2147483647:  # INT32_MAX
                memset(seen_epoch, 0, vocab_size * sizeof(INT_t))
                current_epoch = 1
            
            # Get sentence bounds from offset array
            sent_start = corpus.offsets[s]
            sent_end = corpus.offsets[s + 1]
            sent_len = sent_end - sent_start
            
            # Clear buffers for this sentence
            unique_tokens_buffer.clear()
            targets_in_sent_buffer.clear()
            
            # Build list of unique token indices using epoch-based detection (O(1))
            for i in range(sent_len):
                token_idx = corpus.tokens[sent_start + i]
                
                if token_idx >= vocab_size:
                    continue
                
                # O(1) uniqueness check using epochs
                if seen_epoch[token_idx] != current_epoch:
                    seen_epoch[token_idx] = current_epoch
                    unique_tokens_buffer.push_back(token_idx)
                    
                    # Check if this token is a target
                    if is_target[token_idx]:
                        targets_in_sent_buffer.push_back(token_idx)
            
            n_unique = <int>unique_tokens_buffer.size()
            
            # Update global sentence counts for all unique tokens
            i = 0
            while i < n_unique:
                token_idx = unique_tokens_buffer[i]
                sentences_with_token[token_idx] += 1
                i += 1
            
            # For each target in this sentence, count all unique tokens
            n_targets_in_sent = <int>targets_in_sent_buffer.size()
            i = 0
            while i < n_targets_in_sent:
                token_idx = targets_in_sent_buffer[i]
                t = target_idx_map[token_idx]
                
                # Safety check (should never be -1 if logic is correct)
                if t >= 0:
                    # Count all unique tokens in sentence for this target
                    j = 0
                    while j < n_unique:
                        candidate_sentences[t, unique_tokens_buffer[j]] += 1
                        j += 1
                i += 1
            
            # Increment epoch for next sentence
            current_epoch += 1
    
    # Free allocated memory
    free(is_target)
    free(target_idx_map)
    free(seen_epoch)
    
    # Return the arrays directly (word2idx passed from caller)
    return candidate_sentences_arr, sentences_with_token_arr, n_sentences, word2idx, idx2word, target_indices_np

cdef calculate_window_counts(
    EncodedCorpus* corpus,
    INT_t[::1] target_indices,
    int left_horizon,
    int right_horizon,
    int vocab_size
):
    """
    Window-based collocation counting operating on C-level encoded corpus.
    
    Processes corpus from contiguous C buffers without any Python dictionary access.
    Uses epoch-based active targets tracking: eliminates memset per position and only
    iterates over targets actually seen in window. Hot loops operate without GIL
    on C-level memory. Uses dense NumPy arrays for counting directly in nogil blocks.
    
    Args:
        corpus: Pointer to encoded corpus structure
        target_indices: Indices of target words (n_targets,)
        left_horizon: Window size on the left side
        right_horizon: Window size on the right side
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (T_count, candidate_in_context, token_counter, total_tokens)
            - T_count: For each target, count of positions with target in context
            - candidate_in_context: For each target, count of each candidate in those positions
            - token_counter: Global count of each token
            - total_tokens: Total number of token positions
    """
    cdef int n_sentences = corpus.n_sentences
    cdef int n_targets = target_indices.shape[0]
    cdef int i, j, s, t, start, end, token_idx, context_idx, max_active
    cdef LONG_t sent_start, sent_end, sent_len
    cdef LONG_t total_tokens = 0
    cdef char* is_target = NULL
    cdef INT_t* target_idx_map = NULL
    cdef INT_t* target_epoch = NULL  # Epoch-based tracking instead of memset
    cdef INT_t window_epoch = 1
    cdef vector[INT_t] active_targets  # Only targets actually in window
    cdef int n_active
    
    # Allocate all NumPy arrays upfront
    cdef np.ndarray[LONG_t, ndim=1] T_count_arr = np.zeros(n_targets, dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=2] candidate_counts_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] token_counter_arr = np.zeros(vocab_size, dtype=np.int64)
    
    # Create typed memoryviews for direct access in nogil blocks
    cdef LONG_t[::1] T_count = T_count_arr
    cdef LONG_t[:, ::1] candidate_counts = candidate_counts_arr
    cdef LONG_t[::1] token_counter = token_counter_arr
    
    # Build target lookup table for O(1) checking
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    # Build reverse mapping: vocab_idx -> target_position (initialize to -1 for safety)
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    memset(target_idx_map, 0xFF, vocab_size * sizeof(INT_t))  # -1 for all entries
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    # Allocate epoch array for tracking which targets are in current window
    target_epoch = <INT_t*>malloc(n_targets * sizeof(INT_t))
    if target_epoch == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for target_epoch")
    memset(target_epoch, 0, n_targets * sizeof(INT_t))
    
    # Reserve space for active targets buffer - at most min(n_targets, window_size)
    max_active = n_targets if n_targets < (left_horizon + right_horizon + 1) else (left_horizon + right_horizon + 1)
    active_targets.reserve(max_active)
    
    # Main counting loop - Process C-level encoded corpus (fully nogil!)
    with nogil:
        for s in range(n_sentences):
            # Get sentence bounds from offset array
            sent_start = corpus.offsets[s]
            sent_end = corpus.offsets[s + 1]
            sent_len = sent_end - sent_start
            
            # Process each position in sentence with epoch-based active targets
            for i in range(sent_len):
                token_idx = corpus.tokens[sent_start + i]
                total_tokens += 1
                token_counter[token_idx] += 1
                
                # Define window bounds (excluding center token)
                start = i - left_horizon if i >= left_horizon else 0
                end = i + right_horizon + 1 if i + right_horizon + 1 <= sent_len else sent_len
                
                # Clear active targets for this window (no memset needed!)
                active_targets.clear()
                
                # Check for epoch overflow and reset if necessary
                if window_epoch == 2147483647:  # INT32_MAX
                    memset(target_epoch, 0, n_targets * sizeof(INT_t))
                    window_epoch = 1
                
                # Scan window and collect active targets using epoch tracking
                for j in range(start, end):
                    if j != i:  # Skip center token
                        context_idx = corpus.tokens[sent_start + j]
                        if context_idx < vocab_size and is_target[context_idx]:
                            t = target_idx_map[context_idx]
                            # O(1) check: has this target been seen in current window?
                            if t >= 0 and target_epoch[t] != window_epoch:
                                target_epoch[t] = window_epoch
                                active_targets.push_back(t)
                
                # Only iterate over active targets (not all targets!)
                n_active = <int>active_targets.size()
                for j in range(n_active):
                    t = active_targets[j]
                    T_count[t] += 1
                    candidate_counts[t, token_idx] += 1
                
                # Increment window epoch for next position
                window_epoch += 1
    
    # Free allocated memory
    free(is_target)
    free(target_idx_map)
    free(target_epoch)
    
    # Return the arrays directly (no sparse-to-dense conversion needed)
    return T_count_arr, candidate_counts_arr, token_counter_arr, total_tokens
