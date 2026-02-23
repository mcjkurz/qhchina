# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Collocation and co-occurrence counting operations implemented in Cython.

This module provides optimized implementations of core counting operations:

Collocation counting (find_collocates):
- C-level sentence encoding: Flattened contiguous buffer with offset array for O(1) access
- Zero dictionary access in hot loops: All counting operates on pure C-level integer arrays
- Active targets tracking: Window routine only iterates over targets actually seen in window
- Epoch-based uniqueness: O(1) duplicate detection without linear search
- Dense NumPy arrays for direct counting in nogil blocks
- GIL-free hot loops for maximum performance

Co-occurrence matrix (cooc_matrix):
- C++ unordered_map accumulation: Deduplicates co-occurrence pairs in-place during counting,
  using O(nnz) memory instead of O(total_pairs). Avoids the scipy COO->CSR sort bottleneck.
- Streaming batches: Each batch is encoded, counted into a persistent map (nogil), and freed.
- Direct CSR construction: Map entries are sorted once by (row, col) and written directly
  into CSR arrays — no intermediate COO format.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.pair cimport pair

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.int64_t LONG_t

# C-level struct for encoded sentences
cdef struct EncodedSentences:
    INT_t* tokens        # Flattened token array
    LONG_t* offsets      # Sentence start offsets (n_sentences + 1)
    int n_sentences      # Number of sentences
    LONG_t total_tokens  # Total number of tokens
    int max_sent_len     # Maximum sentence length (for buffer allocation)

cdef EncodedSentences* encode_sentences_to_c_buffer(list tokenized_sentences, dict word2idx, bint keep_oov=False) except NULL:
    """
    Encode sentences into compact C-level buffers (flattened tokens + offsets).
    
    Performs all word-to-index lookups upfront and stores the result in contiguous
    C memory for maximum cache efficiency. No per-sentence allocations.
    Also computes the maximum sentence length for buffer allocation hints.
    
    Args:
        tokenized_sentences: List of tokenized sentences (list of lists of strings)
        word2idx: Dictionary mapping words to indices
        keep_oov: If True, OOV words are kept as -1 sentinel values to preserve
                  positional distances. If False (default), OOV words are skipped.
        
    Returns:
        EncodedSentences*: Pointer to encoded sentences structure (caller must free)
    """
    cdef int n_sentences = len(tokenized_sentences)
    cdef int i, j, sent_len, token_idx
    cdef int max_sent_len = 0
    cdef LONG_t estimated_tokens = 0
    cdef list sentence
    cdef str word
    cdef object word2idx_get = word2idx.get
    cdef vector[INT_t] tokens_vec
    cdef vector[LONG_t] offsets_vec
    cdef LONG_t current_offset = 0
    cdef EncodedSentences* encoded = NULL
    
    # Calculate precise token count and max sentence length
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        sent_len = len(sentence)
        estimated_tokens += sent_len
        if sent_len > max_sent_len:
            max_sent_len = sent_len
    
    # Reserve exact space needed
    offsets_vec.reserve(n_sentences + 1)
    tokens_vec.reserve(estimated_tokens)
    
    # Build flattened token array and offsets
    offsets_vec.push_back(0)
    for i in range(n_sentences):
        sentence = tokenized_sentences[i]
        sent_len = len(sentence)
        
        # Convert words to indices and append to flat buffer
        for j in range(sent_len):
            word = sentence[j]
            token_idx = word2idx_get(word, -1)
            if keep_oov:
                tokens_vec.push_back(token_idx)
                current_offset += 1
            elif token_idx >= 0:
                tokens_vec.push_back(token_idx)
                current_offset += 1
        
        # Store offset for next sentence
        offsets_vec.push_back(current_offset)
    
    # Allocate C struct
    encoded = <EncodedSentences*>malloc(sizeof(EncodedSentences))
    if encoded == NULL:
        raise MemoryError("Failed to allocate EncodedSentences")
    
    # Allocate and copy token array
    encoded.total_tokens = <LONG_t>tokens_vec.size()
    encoded.tokens = <INT_t*>malloc(encoded.total_tokens * sizeof(INT_t))
    if encoded.tokens == NULL:
        free(encoded)
        raise MemoryError("Failed to allocate tokens array")
    
    for i in range(encoded.total_tokens):
        encoded.tokens[i] = tokens_vec[i]
    
    # Allocate and copy offsets array
    encoded.n_sentences = n_sentences
    encoded.offsets = <LONG_t*>malloc((n_sentences + 1) * sizeof(LONG_t))
    if encoded.offsets == NULL:
        free(encoded.tokens)
        free(encoded)
        raise MemoryError("Failed to allocate offsets array")
    
    for i in range(n_sentences + 1):
        encoded.offsets[i] = offsets_vec[i]
    
    encoded.max_sent_len = max_sent_len
    
    return encoded

cdef void free_encoded_sentences(EncodedSentences* encoded) noexcept nogil:
    """
    Free memory allocated for encoded sentences.
    
    Args:
        encoded: Pointer to encoded sentences structure
    """
    if encoded != NULL:
        if encoded.tokens != NULL:
            free(encoded.tokens)
        if encoded.offsets != NULL:
            free(encoded.offsets)
        free(encoded)

cdef calculate_sentence_counts(
    EncodedSentences* encoded,
    INT_t[::1] target_indices,
    int vocab_size,
):
    """
    Core sentence counting logic operating on C-level encoded sentences.
    
    Processes encoded sentences from contiguous C buffers without any Python
    dictionary access. Uses epoch-based uniqueness detection for O(1) duplicate
    checking with overflow guard. All hot loops are GIL-free.
    
    Args:
        encoded: Pointer to encoded sentences structure.
        target_indices: Memoryview of target word indices.
        vocab_size: Size of the vocabulary.
    
    Returns:
        tuple: (candidate_sentences_arr, sentences_with_token_arr, n_sentences)
    """
    cdef int n_targets = target_indices.shape[0]
    cdef int n_sentences = encoded.n_sentences
    cdef int max_sent_len = encoded.max_sent_len
    cdef int s, t, i, j, token_idx, n_unique, n_targets_in_sent
    cdef LONG_t sent_start, sent_end, sent_len
    cdef char* is_target = NULL
    cdef INT_t* target_idx_map = NULL
    cdef INT_t* seen_epoch = NULL
    cdef INT_t current_epoch = 1
    cdef vector[INT_t] unique_tokens_buffer
    cdef vector[INT_t] targets_in_sent_buffer
    
    cdef np.ndarray[LONG_t, ndim=2] candidate_sentences_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] sentences_with_token_arr = np.zeros(vocab_size, dtype=np.int64)
    
    cdef LONG_t[:, ::1] candidate_sentences = candidate_sentences_arr
    cdef LONG_t[::1] sentences_with_token = sentences_with_token_arr
    
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    memset(target_idx_map, 0xFF, vocab_size * sizeof(INT_t))
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    seen_epoch = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if seen_epoch == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for seen_epoch")
    memset(seen_epoch, 0, vocab_size * sizeof(INT_t))
    
    unique_tokens_buffer.reserve(max_sent_len)
    targets_in_sent_buffer.reserve(n_targets if n_targets < max_sent_len else max_sent_len)
    
    with nogil:
        for s in range(n_sentences):
            if current_epoch == 2147483647:  # INT32_MAX
                memset(seen_epoch, 0, vocab_size * sizeof(INT_t))
                current_epoch = 1
            
            sent_start = encoded.offsets[s]
            sent_end = encoded.offsets[s + 1]
            sent_len = sent_end - sent_start
            
            unique_tokens_buffer.clear()
            targets_in_sent_buffer.clear()
            
            for i in range(sent_len):
                token_idx = encoded.tokens[sent_start + i]
                
                if token_idx >= vocab_size:
                    continue
                
                if seen_epoch[token_idx] != current_epoch:
                    seen_epoch[token_idx] = current_epoch
                    unique_tokens_buffer.push_back(token_idx)
                    
                    if is_target[token_idx]:
                        targets_in_sent_buffer.push_back(token_idx)
            
            n_unique = <int>unique_tokens_buffer.size()
            
            i = 0
            while i < n_unique:
                token_idx = unique_tokens_buffer[i]
                sentences_with_token[token_idx] += 1
                i += 1
            
            n_targets_in_sent = <int>targets_in_sent_buffer.size()
            i = 0
            while i < n_targets_in_sent:
                token_idx = targets_in_sent_buffer[i]
                t = target_idx_map[token_idx]
                
                if t >= 0:
                    j = 0
                    while j < n_unique:
                        candidate_sentences[t, unique_tokens_buffer[j]] += 1
                        j += 1
                i += 1
            
            current_epoch += 1
    
    free(is_target)
    free(target_idx_map)
    free(seen_epoch)
    
    return candidate_sentences_arr, sentences_with_token_arr, n_sentences

cdef calculate_window_counts(
    EncodedSentences* encoded,
    INT_t[::1] target_indices,
    int left_horizon,
    int right_horizon,
    int vocab_size
):
    """
    Window-based collocation counting operating on C-level encoded sentences.
    
    Processes encoded sentences from contiguous C buffers without any Python
    dictionary access. Uses epoch-based active targets tracking. Hot loops
    operate without GIL on C-level memory.
    
    Args:
        encoded: Pointer to encoded sentences structure
        target_indices: Indices of target words (n_targets,)
        left_horizon: Window size on the left side
        right_horizon: Window size on the right side
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (T_count, candidate_in_context, token_counter, total_tokens)
    """
    cdef int n_sentences = encoded.n_sentences
    cdef int n_targets = target_indices.shape[0]
    cdef int i, j, s, t, start, end, token_idx, context_idx, max_active
    cdef LONG_t sent_start, sent_end, sent_len
    cdef LONG_t total_tokens = 0
    cdef char* is_target = NULL
    cdef INT_t* target_idx_map = NULL
    cdef INT_t* target_epoch = NULL
    cdef INT_t window_epoch = 1
    cdef vector[INT_t] active_targets
    cdef int n_active
    
    cdef np.ndarray[LONG_t, ndim=1] T_count_arr = np.zeros(n_targets, dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=2] candidate_counts_arr = np.zeros((n_targets, vocab_size), dtype=np.int64)
    cdef np.ndarray[LONG_t, ndim=1] token_counter_arr = np.zeros(vocab_size, dtype=np.int64)
    
    cdef LONG_t[::1] T_count = T_count_arr
    cdef LONG_t[:, ::1] candidate_counts = candidate_counts_arr
    cdef LONG_t[::1] token_counter = token_counter_arr
    
    is_target = <char*>malloc(vocab_size * sizeof(char))
    if is_target == NULL:
        raise MemoryError("Failed to allocate memory for is_target")
    memset(is_target, 0, vocab_size * sizeof(char))
    
    target_idx_map = <INT_t*>malloc(vocab_size * sizeof(INT_t))
    if target_idx_map == NULL:
        free(is_target)
        raise MemoryError("Failed to allocate memory for target_idx_map")
    memset(target_idx_map, 0xFF, vocab_size * sizeof(INT_t))
    
    for t in range(n_targets):
        if target_indices[t] < vocab_size:
            is_target[target_indices[t]] = 1
            target_idx_map[target_indices[t]] = t
    
    target_epoch = <INT_t*>malloc(n_targets * sizeof(INT_t))
    if target_epoch == NULL:
        free(is_target)
        free(target_idx_map)
        raise MemoryError("Failed to allocate memory for target_epoch")
    memset(target_epoch, 0, n_targets * sizeof(INT_t))
    
    max_active = n_targets if n_targets < (left_horizon + right_horizon + 1) else (left_horizon + right_horizon + 1)
    active_targets.reserve(max_active)
    
    with nogil:
        for s in range(n_sentences):
            sent_start = encoded.offsets[s]
            sent_end = encoded.offsets[s + 1]
            sent_len = sent_end - sent_start
            
            for i in range(sent_len):
                token_idx = encoded.tokens[sent_start + i]
                total_tokens += 1
                token_counter[token_idx] += 1
                
                start = i - left_horizon if i >= left_horizon else 0
                end = i + right_horizon + 1 if i + right_horizon + 1 <= sent_len else sent_len
                
                active_targets.clear()
                
                if window_epoch == 2147483647:  # INT32_MAX
                    memset(target_epoch, 0, n_targets * sizeof(INT_t))
                    window_epoch = 1
                
                for j in range(start, end):
                    if j != i:
                        context_idx = encoded.tokens[sent_start + j]
                        if context_idx < vocab_size and is_target[context_idx]:
                            t = target_idx_map[context_idx]
                            if t >= 0 and target_epoch[t] != window_epoch:
                                target_epoch[t] = window_epoch
                                active_targets.push_back(t)
                
                n_active = <int>active_targets.size()
                for j in range(n_active):
                    t = active_targets[j]
                    T_count[t] += 1
                    candidate_counts[t, token_idx] += 1
                
                window_epoch += 1
    
    free(is_target)
    free(target_idx_map)
    free(target_epoch)
    
    return T_count_arr, candidate_counts_arr, token_counter_arr, total_tokens


# =============================================================================
# Batch Processing Wrappers
# =============================================================================

def calculate_window_counts_batch(
    list batch,
    dict word2idx,
    np.ndarray[INT_t, ndim=1] target_indices,
    int left_horizon,
    int right_horizon,
    int vocab_size
):
    """
    Process a single batch of sentences for window-based collocation counting.
    
    Thin wrapper that encodes sentences to C buffers, runs the nogil counting
    loop, and frees the buffer. Designed to be called repeatedly from Python
    with accumulators summed across batches.
    
    Args:
        batch: List of tokenized sentences (one batch worth)
        word2idx: Pre-built word-to-index mapping (shared across batches)
        target_indices: Array of target word indices
        left_horizon: Window size on the left side
        right_horizon: Window size on the right side
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (T_count, candidate_counts, token_counter, total_tokens)
            All arrays are fresh per batch; caller accumulates with +=.
    """
    cdef EncodedSentences* encoded = NULL
    cdef INT_t[::1] target_indices_view = target_indices
    
    encoded = encode_sentences_to_c_buffer(batch, word2idx)
    try:
        return calculate_window_counts(
            encoded, target_indices_view, left_horizon, right_horizon, vocab_size
        )
    finally:
        free_encoded_sentences(encoded)


def calculate_sentence_counts_batch(
    list batch,
    dict word2idx,
    np.ndarray[INT_t, ndim=1] target_indices,
    int vocab_size
):
    """
    Process a single batch of sentences for sentence-based collocation counting.
    
    Thin wrapper that encodes sentences to C buffers, runs the nogil counting
    loop, and frees the buffer. Designed to be called repeatedly from Python
    with accumulators summed across batches.
    
    Args:
        batch: List of tokenized sentences (one batch worth)
        word2idx: Pre-built word-to-index mapping (shared across batches)
        target_indices: Array of target word indices
        vocab_size: Size of the vocabulary
        
    Returns:
        tuple: (candidate_sentences, sentences_with_token, n_sentences)
            Arrays are fresh per batch; caller accumulates with +=.
    """
    cdef EncodedSentences* encoded = NULL
    cdef INT_t[::1] target_indices_view = target_indices
    
    encoded = encode_sentences_to_c_buffer(batch, word2idx)
    try:
        return calculate_sentence_counts(
            encoded, target_indices_view, vocab_size,
        )
    finally:
        free_encoded_sentences(encoded)


# =============================================================================
# Co-occurrence Matrix Functions (hash-map accumulation)
# =============================================================================

# Struct for sorting map entries into CSR order
cdef struct CoocEntry:
    INT_t row
    INT_t col
    LONG_t value

cdef inline bint _cooc_entry_less(const CoocEntry& a, const CoocEntry& b) noexcept nogil:
    if a.row != b.row:
        return a.row < b.row
    return a.col < b.col


cdef void _count_window_pairs(
    EncodedSentences* encoded,
    unordered_map[long long, long long]* counts,
    int vocab_size,
    int left_horizon,
    int right_horizon,
) noexcept nogil:
    """
    Window-based co-occurrence counting into an unordered_map.
    
    Iterates through encoded sentences with OOV sentinel values (-1). Sentinels
    occupy positions (preserving distances) but are skipped for counting.
    Accumulates directly into the map — no duplicate pairs are stored.
    
    Args:
        encoded: Pointer to encoded sentences structure (with keep_oov=True encoding).
        counts: Pointer to persistent map; caller owns lifetime.
        vocab_size: Used to compute composite key (row * vocab_size + col).
        left_horizon: Window size on the left side.
        right_horizon: Window size on the right side.
    """
    cdef int n_docs = encoded.n_sentences
    cdef int i, j, s, start, end, center_idx, context_idx
    cdef LONG_t doc_start, doc_end, doc_len
    cdef long long key
    cdef long long vs = <long long>vocab_size
    
    for s in range(n_docs):
        doc_start = encoded.offsets[s]
        doc_end = encoded.offsets[s + 1]
        doc_len = doc_end - doc_start
        
        for i in range(<int>doc_len):
            center_idx = encoded.tokens[doc_start + i]
            
            if center_idx < 0:
                continue
            
            start = i - left_horizon if i >= left_horizon else 0
            end = i + right_horizon + 1 if i + right_horizon + 1 <= <int>doc_len else <int>doc_len
            
            for j in range(start, end):
                if j != i:
                    context_idx = encoded.tokens[doc_start + j]
                    
                    if context_idx < 0:
                        continue
                    
                    key = <long long>center_idx * vs + <long long>context_idx
                    counts[0][key] += 1


cdef void _count_document_pairs(
    EncodedSentences* encoded,
    unordered_map[long long, long long]* counts,
    int vocab_size,
    bint binary,
) noexcept nogil:
    """
    Document-based co-occurrence counting into an unordered_map.
    
    For each document, deduplicates tokens and collects per-document frequencies.
    Then for all unique pairs (i, j) where i < j, accumulates symmetric entries
    into the map with weight freq[i] * freq[j] (or 1 if binary).
    
    Args:
        encoded: Pointer to encoded sentences structure (with keep_oov=False encoding).
        counts: Pointer to persistent map; caller owns lifetime.
        vocab_size: Used to compute composite key.
        binary: If True, each pair contributes 1 regardless of frequency.
    """
    cdef int n_docs = encoded.n_sentences
    cdef int s, i, j, token_idx, n_unique
    cdef LONG_t doc_start, doc_end, doc_len
    cdef LONG_t weight
    cdef INT_t idx_a, idx_b
    cdef long long key_ab, key_ba
    cdef long long vs = <long long>vocab_size
    
    cdef LONG_t* freq = <LONG_t*>malloc(vocab_size * sizeof(LONG_t))
    if freq == NULL:
        return
    memset(freq, 0, vocab_size * sizeof(LONG_t))
    
    cdef vector[INT_t] unique_tokens
    unique_tokens.reserve(encoded.max_sent_len if encoded.max_sent_len < vocab_size else vocab_size)
    
    for s in range(n_docs):
        doc_start = encoded.offsets[s]
        doc_end = encoded.offsets[s + 1]
        doc_len = doc_end - doc_start
        
        if doc_len == 0:
            continue
        
        unique_tokens.clear()
        
        for i in range(<int>doc_len):
            token_idx = encoded.tokens[doc_start + i]
            if freq[token_idx] == 0:
                unique_tokens.push_back(token_idx)
            freq[token_idx] += 1
        
        n_unique = <int>unique_tokens.size()
        
        for i in range(n_unique):
            idx_a = unique_tokens[i]
            for j in range(i + 1, n_unique):
                idx_b = unique_tokens[j]
                
                key_ab = <long long>idx_a * vs + <long long>idx_b
                key_ba = <long long>idx_b * vs + <long long>idx_a
                
                if binary:
                    counts[0][key_ab] = 1
                    counts[0][key_ba] = 1
                else:
                    weight = freq[idx_a] * freq[idx_b]
                    counts[0][key_ab] += weight
                    counts[0][key_ba] += weight
        
        for i in range(n_unique):
            freq[unique_tokens[i]] = 0
    
    free(freq)


cdef _hashmap_to_csr(unordered_map[long long, long long]& counts, int vocab_size):
    """
    Extract an unordered_map of (row*V+col)->value into CSR arrays.
    
    1. Collects all entries into a vector of CoocEntry structs
    2. Sorts by (row, col)
    3. Builds indptr/indices/data arrays in a single pass
    
    Returns:
        tuple: (indptr, indices, data) numpy arrays for scipy.sparse.csr_matrix
    """
    cdef LONG_t nnz = <LONG_t>counts.size()
    cdef long long vs = <long long>vocab_size
    
    if nnz == 0:
        return (
            np.zeros(vocab_size + 1, dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
        )
    
    cdef vector[CoocEntry] entries
    entries.reserve(nnz)
    
    cdef CoocEntry entry
    cdef pair[long long, long long] item
    for item in counts:
        entry.row = <INT_t>(item.first / vs)
        entry.col = <INT_t>(item.first % vs)
        entry.value = item.second
        entries.push_back(entry)
    
    cpp_sort(entries.begin(), entries.end(), &_cooc_entry_less)
    
    cdef np.ndarray[LONG_t, ndim=1] indptr_arr = np.zeros(vocab_size + 1, dtype=np.int64)
    cdef np.ndarray[INT_t, ndim=1] indices_arr = np.empty(nnz, dtype=np.int32)
    cdef np.ndarray[LONG_t, ndim=1] data_arr = np.empty(nnz, dtype=np.int64)
    
    cdef LONG_t[::1] indptr = indptr_arr
    cdef INT_t[::1] indices = indices_arr
    cdef LONG_t[::1] data = data_arr
    
    cdef LONG_t k
    for k in range(nnz):
        indices[k] = entries[k].col
        data[k] = entries[k].value
        indptr[entries[k].row + 1] += 1
    
    # Convert counts-per-row to cumulative offsets
    cdef int r
    for r in range(vocab_size):
        indptr[r + 1] += indptr[r]
    
    return indptr_arr, indices_arr, data_arr


def calculate_cooc_matrix(documents, dict word_to_index, str method,
                          int left_horizon=0, int right_horizon=0,
                          bint binary=False, int batch_words=100_000):
    """
    Cython-accelerated co-occurrence matrix calculation with hash-map accumulation.
    
    Processes documents in streaming batches. Each batch is encoded to C-level
    buffers and counted in a nogil loop that accumulates into a C++ unordered_map.
    After all batches, the map is extracted into sorted CSR arrays. This avoids
    the O(P log P) scipy COO->CSR sort on raw pairs and uses O(nnz) memory
    instead of O(total_pairs).
    
    Args:
        documents: Iterable of tokenized documents (list of lists of strings).
            Consumed once in a single pass.
        word_to_index: Dictionary mapping vocabulary words to matrix indices.
        method: 'window' for sliding-window co-occurrence, 'document' for
            bag-of-words co-occurrence within each document.
        left_horizon: Window size on the left side (window method only).
        right_horizon: Window size on the right side (window method only).
        binary: If True, all co-occurrence counts are clamped to 1.
        batch_words: Target number of tokens per processing batch.
    
    Returns:
        tuple: (indptr, indices, data) numpy arrays for direct construction
            of a scipy.sparse.csr_matrix with shape (vocab_size, vocab_size).
    """
    from qhchina.utils import iter_batches
    
    cdef int vocab_size = len(word_to_index)
    cdef unordered_map[long long, long long] counts
    cdef EncodedSentences* encoded = NULL
    cdef bint is_window = (method == 'window')
    
    if vocab_size == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
        )
    
    for batch in iter_batches(documents, batch_words, max_length=None):
        encoded = encode_sentences_to_c_buffer(batch, word_to_index, keep_oov=is_window)
        try:
            if is_window:
                with nogil:
                    _count_window_pairs(encoded, &counts, vocab_size, left_horizon, right_horizon)
            else:
                with nogil:
                    _count_document_pairs(encoded, &counts, vocab_size, binary)
        finally:
            free_encoded_sentences(encoded)
    
    if binary and is_window:
        for item in counts:
            counts[item.first] = 1
    
    return _hashmap_to_csr(counts, vocab_size)
