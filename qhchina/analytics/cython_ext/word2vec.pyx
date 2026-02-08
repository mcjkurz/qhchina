"""
Fast Word2Vec training operations implemented in Cython.

This module provides optimized implementations of the core training
operations for Word2Vec with minimal Python/Cython boundary crossings.

THREAD SAFETY: This module is fully thread-safe. All state is passed as
parameters to functions - there are no global mutable variables.

Key optimizations:
- Direct BLAS function calls via scipy.linalg.cython_blas
- Vocabulary lookup happens inside Cython (with GIL, but no intermediate allocations)
- Fixed-size stack-allocated buffers (like Gensim) for maximum performance
- GIL released for actual training computations
- Conditional loss computation (only when needed)
"""

# Compiler directives for maximum performance
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

import cython
import numpy as np
cimport numpy as np
from libc.string cimport memset
from libc.math cimport exp, log

# Import BLAS functions for optimized linear algebra (direct cython bindings)
from scipy.linalg.cython_blas cimport sdot, saxpy, sscal

# Define C types
ctypedef np.float32_t REAL_t
ctypedef np.int32_t ITYPE_t
ctypedef np.uint32_t UITYPE_t

# Constants
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6.0

# Maximum words per batch - fixed at compile time (like Gensim's MAX_SENTENCE_LEN)
# This determines the size of stack-allocated buffers for training
DEF MAX_WORDS_IN_BATCH = 10000

# Export to Python so the Word2Vec class can use it
MAX_BATCH_SIZE = MAX_WORDS_IN_BATCH

# =============================================================================
# Sigmoid Lookup Tables (read-only, initialized once at module load)
# =============================================================================

# These are the only module-level variables. They are:
# 1. Read-only after initialization
# 2. Thread-safe (no writes during training)
# 3. Initialized at module import time
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE
cdef bint _tables_initialized = False

cdef void _init_tables() noexcept:
    """Initialize sigmoid lookup tables (called once at module load)."""
    global _tables_initialized
    if _tables_initialized:
        return
    
    cdef int i
    cdef REAL_t x
    for i in range(EXP_TABLE_SIZE):
        x = (i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP
        EXP_TABLE[i] = <REAL_t>(1.0 / (1.0 + exp(-x)))
        LOG_TABLE[i] = <REAL_t>log(max(EXP_TABLE[i], 1e-10))
    
    _tables_initialized = True

# Initialize tables at module import
_init_tables()


# =============================================================================
# Fast LCG Random Number Generator
# =============================================================================

cdef inline unsigned long long random_int32(unsigned long long *next_random) noexcept nogil:
    """Fast LCG random number generator."""
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random


# =============================================================================
# Binary Search for Negative Sampling
# =============================================================================

cdef inline unsigned long long bisect_left(
    UITYPE_t *a, 
    unsigned long long x, 
    unsigned long long lo, 
    unsigned long long hi
) noexcept nogil:
    """Binary search on cumulative table."""
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo


# =============================================================================
# Core Skip-gram Training (nogil)
# =============================================================================

cdef inline unsigned long long train_sg_pair(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    const UITYPE_t word_index,
    const UITYPE_t context_index,
    const REAL_t alpha,
    REAL_t *work,
    unsigned long long next_random,
    # Negative sampling parameters (passed, not global)
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    # Loss computation
    const int _compute_loss,
    REAL_t *running_loss
) noexcept nogil:
    """Train on a single (center, context) pair with negative sampling."""
    cdef long long row1 = <long long>context_index * <long long>size
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d
    
    memset(work, 0, size * cython.sizeof(REAL_t))
    
    for d in range(negative + 1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len - 1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0
        
        row2 = <long long>target_index * <long long>size
        f_dot = sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        
        # Accumulate loss only if requested
        if _compute_loss == 1:
            f_dot_for_loss = (f_dot if d == 0 else -f_dot)
            if f_dot_for_loss <= -MAX_EXP or f_dot_for_loss >= MAX_EXP:
                pass  # Skip this loss term
            else:
                log_e_f_dot = LOG_TABLE[<int>((f_dot_for_loss + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                running_loss[0] = running_loss[0] - log_e_f_dot
        
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
    
    saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)
    
    return next_random


# =============================================================================
# Core CBOW Training (nogil)
# =============================================================================

cdef inline unsigned long long train_cbow_pair(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *context_indices,
    int context_len,
    const UITYPE_t center_index,
    const REAL_t alpha,
    REAL_t *neu1,
    REAL_t *work,
    unsigned long long next_random,
    bint cbow_mean,
    # Negative sampling parameters (passed, not global)
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    # Loss computation
    const int _compute_loss,
    REAL_t *running_loss
) noexcept nogil:
    """Train on a single CBOW example with negative sampling."""
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, count, inv_count, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d, m
    
    # Build combined context vector
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(context_len):
        saxpy(&size, &ONEF, &syn0[<long long>context_indices[m] * <long long>size], &ONE, neu1, &ONE)
        count += ONEF
    
    if count < <REAL_t>0.5:
        return next_random
    
    inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)
    
    memset(work, 0, size * cython.sizeof(REAL_t))
    
    for d in range(negative + 1):
        if d == 0:
            target_index = center_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len - 1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == center_index:
                continue
            label = <REAL_t>0.0
        
        row2 = <long long>target_index * <long long>size
        f_dot = sdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        
        # Accumulate loss only if requested
        if _compute_loss == 1:
            f_dot_for_loss = (f_dot if d == 0 else -f_dot)
            if f_dot_for_loss <= -MAX_EXP or f_dot_for_loss >= MAX_EXP:
                pass  # Skip this loss term
            else:
                log_e_f_dot = LOG_TABLE[<int>((f_dot_for_loss + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                running_loss[0] = running_loss[0] - log_e_f_dot
        
        saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    
    if not cbow_mean:
        sscal(&size, &inv_count, work, &ONE)
    
    for m in range(context_len):
        saxpy(&size, &ONEF, work, &ONE, &syn0[<long long>context_indices[m] * <long long>size], &ONE)
    
    return next_random


# =============================================================================
# Training batch processor (nogil inner loop)
# =============================================================================

cdef void train_batch_sg(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *indexes,
    UITYPE_t *sentence_idx,
    ITYPE_t *reduced_windows,
    int num_sentences,
    int window,
    REAL_t alpha,
    REAL_t *work,
    unsigned long long *next_random,
    # Negative sampling parameters
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    # Loss computation
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a batch of indexed sentences (nogil)."""
    cdef int sent_idx, i, j, k
    cdef int idx_start, idx_end
    
    for sent_idx in range(num_sentences):
        idx_start = sentence_idx[sent_idx]
        idx_end = sentence_idx[sent_idx + 1]
        
        for i in range(idx_start, idx_end):
            j = i - window + reduced_windows[i]
            if j < idx_start:
                j = idx_start
            k = i + window + 1 - reduced_windows[i]
            if k > idx_end:
                k = idx_end
            
            for j in range(j, k):
                if j == i:
                    continue
                
                next_random[0] = train_sg_pair(
                    syn0, syn1neg, size,
                    indexes[i], indexes[j],
                    alpha, work, next_random[0],
                    cum_table, cum_table_len, negative,
                    _compute_loss, running_loss
                )
                examples_trained[0] += 1


cdef void train_batch_cbow(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *indexes,
    UITYPE_t *sentence_idx,
    ITYPE_t *reduced_windows,
    int num_sentences,
    int window,
    REAL_t alpha,
    REAL_t *neu1,
    REAL_t *work,
    UITYPE_t *context_buffer,
    unsigned long long *next_random,
    bint cbow_mean,
    # Negative sampling parameters
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    # Loss computation
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train CBOW on a batch of indexed sentences (nogil)."""
    cdef int sent_idx, i, j, k, m, ctx_count
    cdef int idx_start, idx_end
    
    for sent_idx in range(num_sentences):
        idx_start = sentence_idx[sent_idx]
        idx_end = sentence_idx[sent_idx + 1]
        
        for i in range(idx_start, idx_end):
            j = i - window + reduced_windows[i]
            if j < idx_start:
                j = idx_start
            k = i + window + 1 - reduced_windows[i]
            if k > idx_end:
                k = idx_end
            
            # Collect context indices
            ctx_count = 0
            for m in range(j, k):
                if m == i:
                    continue
                context_buffer[ctx_count] = indexes[m]
                ctx_count += 1
            
            if ctx_count == 0:
                continue
            
            next_random[0] = train_cbow_pair(
                syn0, syn1neg, size,
                context_buffer, ctx_count, indexes[i],
                alpha, neu1, work, next_random[0],
                cbow_mean,
                cum_table, cum_table_len, negative,
                _compute_loss, running_loss
            )
            examples_trained[0] += 1


# =============================================================================
# Main Training Functions - Full Epoch in One Call
# =============================================================================

def train_batch(
    np.float32_t[:, :] syn0,
    np.float32_t[:, :] syn1neg,
    list sentences,
    dict word2idx,
    np.uint32_t[:] sample_ints,
    np.uint32_t[:] cum_table,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float alpha,
    bint sg,
    int negative,
    bint cbow_mean,
    unsigned long long random_seed,
    bint compute_loss=True,
):
    """
    Train Word2Vec (Skip-gram or CBOW) on a batch of sentences.
    
    THREAD SAFE: All state is passed as parameters. No global variables are used.
    
    This function processes a single batch of sentences that has already been
    batched by word count in Python. It uses fixed-size stack-allocated buffers
    (like Gensim) for maximum performance.
    
    IMPORTANT: The batch should contain at most MAX_BATCH_SIZE (10000) words.
    Python is responsible for batching by word count before calling this function.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size)
        syn1neg: Output word vectors (vocab_size x vector_size)
        sentences: List of tokenized sentences (list of list of str)
        word2idx: Dictionary mapping words to indices
        sample_ints: Pre-computed subsampling thresholds (uint32)
        cum_table: Cumulative distribution table for negative sampling
        use_subsampling: Whether to apply subsampling
        window: Context window size
        shrink_windows: Whether to randomly shrink windows
        alpha: Learning rate for this batch
        sg: If True, use Skip-gram; if False, use CBOW
        negative: Number of negative samples
        cbow_mean: If True, average context vectors (for CBOW)
        random_seed: Seed for random number generator
        compute_loss: Whether to compute and track loss (default True)
    
    Returns:
        Tuple of (batch_loss, examples_trained, words_processed, final_random_state)
    """
    cdef int vector_size = syn0.shape[1]
    cdef int num_sentences = len(sentences)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef int _sg = 1 if sg else 0
    cdef unsigned long long cum_table_len = len(cum_table)
    
    # Get raw pointers to weight matrices
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef UITYPE_t *cum_table_ptr = &cum_table[0]
    
    # Fixed-size stack-allocated buffers (like Gensim)
    cdef UITYPE_t indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t sentence_idx[MAX_WORDS_IN_BATCH + 1]
    cdef ITYPE_t reduced_windows[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t context_buffer[50]  # Max window is typically < 25, so 2*25 = 50
    
    # Work buffers need to be dynamically allocated based on vector_size
    cdef np.ndarray[REAL_t, ndim=1] work = np.zeros(vector_size, dtype=np.float32)
    cdef np.ndarray[REAL_t, ndim=1] neu1 = np.zeros(vector_size, dtype=np.float32)
    cdef REAL_t *work_ptr = <REAL_t *>np.PyArray_DATA(work)
    cdef REAL_t *neu1_ptr = <REAL_t *>np.PyArray_DATA(neu1)
    
    # State variables (local, not global)
    cdef unsigned long long next_random = random_seed
    cdef REAL_t running_loss = 0.0
    cdef long long total_examples = 0
    cdef long long total_words = 0
    
    # Batch processing state
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    cdef int word_idx
    cdef int sent_global_idx
    cdef int i
    
    # Process all sentences in this batch
    sentence_idx[0] = 0
    
    for sent_global_idx in range(num_sentences):
        sent = sentences[sent_global_idx]
        
        if sent is None or len(sent) == 0:
            continue
        
        # Index this sentence (vocab lookup WITH GIL, but no allocations)
        for token in sent:
            if token not in word2idx:
                continue
            
            word_idx = word2idx[token]
            total_words += 1
            
            # Subsampling
            if use_subsampling:
                if sample_ints_ptr[word_idx] < (random_int32(&next_random) & 0xFFFFFFFF):
                    continue
            
            indexes[effective_words] = <UITYPE_t>word_idx
            effective_words += 1
            
            # Safety check: don't exceed buffer size
            if effective_words >= MAX_WORDS_IN_BATCH:
                break
        
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words
        
        # Safety check: don't exceed buffer size
        if effective_words >= MAX_WORDS_IN_BATCH:
            break
    
    # Train on all words in this batch
    if effective_words > 0 and effective_sentences > 0:
        if shrink_windows:
            for i in range(effective_words):
                reduced_windows[i] = <ITYPE_t>(random_int32(&next_random) % window)
        else:
            memset(reduced_windows, 0, effective_words * cython.sizeof(ITYPE_t))
        
        with nogil:
            if _sg == 1:
                train_batch_sg(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes, sentence_idx, reduced_windows,
                    effective_sentences, window, alpha,
                    work_ptr, &next_random,
                    cum_table_ptr, cum_table_len, negative,
                    _compute_loss, &running_loss, &total_examples
                )
            else:
                train_batch_cbow(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes, sentence_idx, reduced_windows,
                    effective_sentences, window, alpha,
                    neu1_ptr, work_ptr, context_buffer,
                    &next_random, cbow_mean,
                    cum_table_ptr, cum_table_len, negative,
                    _compute_loss, &running_loss, &total_examples
                )
    
    return running_loss, total_examples, total_words, next_random


# =============================================================================
# Temporal Referencing Variants (for TempRefWord2Vec)
# =============================================================================

def train_batch_temporal(
    np.float32_t[:, :] syn0,
    np.float32_t[:, :] syn1neg,
    list sentences,
    dict word2idx,
    np.int32_t[:] temporal_index_map,
    np.uint32_t[:] sample_ints,
    np.uint32_t[:] cum_table,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float alpha,
    int negative,
    unsigned long long random_seed,
    bint compute_loss=True,
):
    """
    Train Word2Vec Skip-gram with temporal mapping on a batch of sentences.
    
    THREAD SAFE: All state is passed as parameters. No global variables are used.
    
    Context words are mapped to base forms, center words keep temporal variant.
    Uses fixed-size stack-allocated buffers (like Gensim) for maximum performance.
    
    IMPORTANT: The batch should contain at most MAX_BATCH_SIZE (10000) words.
    Python is responsible for batching by word count before calling this function.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size)
        syn1neg: Output word vectors (vocab_size x vector_size)
        sentences: List of tokenized sentences (list of list of str)
        word2idx: Dictionary mapping words to indices
        temporal_index_map: Array mapping vocab index -> base form index (identity mapping for non-temporal words)
        sample_ints: Pre-computed subsampling thresholds (uint32)
        cum_table: Cumulative distribution table for negative sampling
        use_subsampling: Whether to apply subsampling
        window: Context window size
        shrink_windows: Whether to randomly shrink windows
        alpha: Learning rate for this batch
        negative: Number of negative samples
        random_seed: Seed for random number generator
        compute_loss: Whether to compute and track loss (default True)
    
    Returns:
        Tuple of (batch_loss, examples_trained, words_processed, final_random_state)
    """
    cdef int vector_size = syn0.shape[1]
    cdef int num_sentences = len(sentences)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef unsigned long long cum_table_len = len(cum_table)
    
    # Get raw pointers
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef ITYPE_t *temporal_map_ptr = &temporal_index_map[0]
    cdef UITYPE_t *cum_table_ptr = &cum_table[0]
    
    # Fixed-size stack-allocated buffers (like Gensim)
    cdef UITYPE_t center_indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t context_indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t sentence_idx[MAX_WORDS_IN_BATCH + 1]
    cdef ITYPE_t reduced_windows[MAX_WORDS_IN_BATCH]
    
    # Work buffer needs to be dynamically allocated based on vector_size
    cdef np.ndarray[REAL_t, ndim=1] work = np.zeros(vector_size, dtype=np.float32)
    cdef REAL_t *work_ptr = <REAL_t *>np.PyArray_DATA(work)
    
    # State variables (local, not global)
    cdef unsigned long long next_random = random_seed
    cdef REAL_t running_loss = 0.0
    cdef long long total_examples = 0
    cdef long long total_words = 0
    
    # Batch processing state
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    cdef int word_idx
    cdef int sent_global_idx
    cdef int i
    
    # Process all sentences in this batch
    sentence_idx[0] = 0
    
    for sent_global_idx in range(num_sentences):
        sent = sentences[sent_global_idx]
        
        if sent is None or len(sent) == 0:
            continue
        
        # Index this sentence with temporal mapping
        for token in sent:
            if token not in word2idx:
                continue
            
            word_idx = word2idx[token]
            total_words += 1
            
            # Subsampling (use original index for subsampling decision)
            if use_subsampling:
                if sample_ints_ptr[word_idx] < (random_int32(&next_random) & 0xFFFFFFFF):
                    continue
            
            # Store center index (may be temporal variant)
            center_indexes[effective_words] = <UITYPE_t>word_idx
            
            # Store context index (mapped to base form via temporal_index_map)
            # For non-temporal words, this is identity mapping (index maps to itself)
            context_indexes[effective_words] = <UITYPE_t>temporal_map_ptr[word_idx]
            
            effective_words += 1
            
            # Safety check: don't exceed buffer size
            if effective_words >= MAX_WORDS_IN_BATCH:
                break
        
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words
        
        # Safety check: don't exceed buffer size
        if effective_words >= MAX_WORDS_IN_BATCH:
            break
    
    # Train on all words in this batch
    if effective_words > 0 and effective_sentences > 0:
        if shrink_windows:
            for i in range(effective_words):
                reduced_windows[i] = <ITYPE_t>(random_int32(&next_random) % window)
        else:
            memset(reduced_windows, 0, effective_words * cython.sizeof(ITYPE_t))
        
        with nogil:
            train_batch_sg_temporal(
                syn0_ptr, syn1neg_ptr, vector_size,
                center_indexes, context_indexes, sentence_idx, reduced_windows,
                effective_sentences, window, alpha,
                work_ptr, &next_random,
                cum_table_ptr, cum_table_len, negative,
                _compute_loss, &running_loss, &total_examples
            )
    
    return running_loss, total_examples, total_words, next_random


cdef void train_batch_sg_temporal(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *center_indexes,
    UITYPE_t *context_indexes,
    UITYPE_t *sentence_idx,
    ITYPE_t *reduced_windows,
    int num_sentences,
    int window,
    REAL_t alpha,
    REAL_t *work,
    unsigned long long *next_random,
    # Negative sampling parameters
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    # Loss computation
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """
    Train skip-gram on a batch with temporal mapping (nogil).
    
    center_indexes: Original word indices (may be temporal variants)
    context_indexes: Mapped indices (base forms for context)
    
    For each (center, context) pair:
    - center word uses center_indexes[i] (keeps temporal variant)
    - context word uses context_indexes[j] (mapped to base form)
    """
    cdef int sent_idx, i, j, k
    cdef int idx_start, idx_end
    
    for sent_idx in range(num_sentences):
        idx_start = sentence_idx[sent_idx]
        idx_end = sentence_idx[sent_idx + 1]
        
        for i in range(idx_start, idx_end):
            j = i - window + reduced_windows[i]
            if j < idx_start:
                j = idx_start
            k = i + window + 1 - reduced_windows[i]
            if k > idx_end:
                k = idx_end
            
            for j in range(j, k):
                if j == i:
                    continue
                
                # Train: center word (original) predicts context word (mapped base form)
                next_random[0] = train_sg_pair(
                    syn0, syn1neg, size,
                    center_indexes[i],   # center word (may be temporal variant)
                    context_indexes[j],  # context word (mapped to base form)
                    alpha, work, next_random[0],
                    cum_table, cum_table_len, negative,
                    _compute_loss, running_loss
                )
                examples_trained[0] += 1
