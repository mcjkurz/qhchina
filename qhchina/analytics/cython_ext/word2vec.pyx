"""
Fast Word2Vec training operations implemented in Cython.

This module provides optimized implementations of the core training
operations for Word2Vec with minimal Python/Cython boundary crossings.

Key optimizations:
- Direct BLAS function calls via scipy.linalg.cython_blas
- Vocabulary lookup happens inside Cython (with GIL, but no intermediate allocations)
- Entire epoch processed in single Cython call
- Pre-allocated fixed-size buffers (no per-sentence allocations)
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

DEF MAX_WORDS_IN_BATCH = 50000  # 50K words max per training chunk
DEF MAX_SENTENCES_IN_BATCH = 10000  # 10K sentences max
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6.0

# =============================================================================
# Global State (initialized once, used throughout training)
# =============================================================================

# Sigmoid lookup tables
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

# Cumulative table for negative sampling
cdef UITYPE_t *CUM_TABLE
cdef unsigned long long CUM_TABLE_LEN

# Training hyperparameters (set during init)
cdef int NEGATIVE
cdef int VECTOR_SIZE
cdef bint CBOW_MEAN

# RNG state
cdef unsigned long long NEXT_RANDOM = 1

# Global reference to prevent garbage collection
_cum_table_holder = None

# =============================================================================
# Fast LCG Random Number Generator
# =============================================================================

cdef inline unsigned long long random_int32(unsigned long long *next_random) noexcept nogil:
    """Fast LCG random number generator."""
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

def set_seed(unsigned long long seed):
    """Set the random number generator seed for reproducibility."""
    global NEXT_RANDOM
    NEXT_RANDOM = seed

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
# Initialization
# =============================================================================

def init_globals(
    noise_distribution,
    int vector_size,
    int negative=5,
    bint cbow_mean=True,
):
    """
    Initialize global variables for training.
    
    Call this once before training starts.
    """
    global CUM_TABLE, CUM_TABLE_LEN
    global NEGATIVE, VECTOR_SIZE, CBOW_MEAN
    global EXP_TABLE, LOG_TABLE, _cum_table_holder
    
    cdef int i
    cdef REAL_t x
    
    # Build sigmoid tables
    for i in range(EXP_TABLE_SIZE):
        x = (i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP
        EXP_TABLE[i] = <REAL_t>(1.0 / (1.0 + np.exp(-x)))
        LOG_TABLE[i] = <REAL_t>np.log(max(EXP_TABLE[i], 1e-10))
    
    # Build cumulative table for negative sampling
    cdef np.ndarray[UITYPE_t, ndim=1] cum_table_arr = np.zeros(len(noise_distribution), dtype=np.uint32)
    cdef double running = 0.0
    cdef int n = len(noise_distribution)
    cdef double domain = 2147483647.0  # 2^31 - 1
    
    for i in range(n):
        running += noise_distribution[i]
        cum_table_arr[i] = <UITYPE_t>min(running * domain, domain)
    
    # Store reference to prevent garbage collection
    _cum_table_holder = cum_table_arr
    CUM_TABLE = <UITYPE_t *>np.PyArray_DATA(cum_table_arr)
    CUM_TABLE_LEN = n
    
    NEGATIVE = negative
    VECTOR_SIZE = vector_size
    CBOW_MEAN = cbow_mean

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
    
    for d in range(NEGATIVE + 1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(CUM_TABLE, (next_random >> 16) % CUM_TABLE[CUM_TABLE_LEN - 1], 0, CUM_TABLE_LEN)
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
        
        # Accumulate loss only if requested (Gensim-style conditional)
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
    
    for d in range(NEGATIVE + 1):
        if d == 0:
            target_index = center_index
            label = ONEF
        else:
            target_index = bisect_left(CUM_TABLE, (next_random >> 16) % CUM_TABLE[CUM_TABLE_LEN - 1], 0, CUM_TABLE_LEN)
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
        
        # Accumulate loss only if requested (Gensim-style conditional)
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
# Training chunk processor (nogil inner loop)
# =============================================================================

cdef void train_chunk_sg(
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
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a chunk of indexed sentences (nogil)."""
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
                    alpha, work, next_random[0], _compute_loss, running_loss
                )
                examples_trained[0] += 1


cdef void train_chunk_cbow(
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
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train CBOW on a chunk of indexed sentences (nogil)."""
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
                cbow_mean, _compute_loss, running_loss
            )
            examples_trained[0] += 1


# =============================================================================
# Main Training Functions - Full Epoch in One Call
# =============================================================================

def train_epoch(
    np.float32_t[:, :] syn0,
    np.float32_t[:, :] syn1neg,
    list sentences,
    dict word2idx,
    np.uint32_t[:] sample_ints,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float start_alpha,
    float end_alpha,
    long long total_words_expected,
    object progress_callback,
    int callback_every_n_sentences,
    bint sg,
    bint compute_loss=True,
):
    """
    Train Word2Vec (Skip-gram or CBOW) for one full epoch.
    
    Processes ALL sentences in a single call, minimizing Python/Cython crossings.
    Vocabulary lookup happens inside Cython.
    
    Learning rate linearly interpolates from start_alpha to end_alpha based on
    word processing progress within this epoch.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size)
        syn1neg: Output word vectors (vocab_size x vector_size)
        sentences: List of tokenized sentences (list of list of str)
        word2idx: Dictionary mapping words to indices
        sample_ints: Pre-computed subsampling thresholds (uint32)
        use_subsampling: Whether to apply subsampling
        window: Context window size
        shrink_windows: Whether to randomly shrink windows
        start_alpha: Learning rate at the start of this epoch
        end_alpha: Learning rate at the end of this epoch
        total_words_expected: Expected in-vocab words for this epoch (for LR decay)
        progress_callback: Function(words_processed, loss) called periodically
        callback_every_n_sentences: How often to call progress callback
        sg: If True, use Skip-gram; if False, use CBOW
        compute_loss: Whether to compute and track loss (default True)
    
    Returns:
        Tuple of (total_loss, total_examples_trained, total_words_processed)
    """
    cdef int vector_size = syn0.shape[1]
    cdef int num_sentences = len(sentences)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef int _sg = 1 if sg else 0
    
    # Get raw pointers to weight matrices
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    
    # Pre-allocate all buffers (fixed size, no per-batch allocation)
    cdef np.ndarray[UITYPE_t, ndim=1] indexes = np.zeros(MAX_WORDS_IN_BATCH, dtype=np.uint32)
    cdef np.ndarray[UITYPE_t, ndim=1] sentence_idx = np.zeros(MAX_SENTENCES_IN_BATCH + 1, dtype=np.uint32)
    cdef np.ndarray[ITYPE_t, ndim=1] reduced_windows = np.zeros(MAX_WORDS_IN_BATCH, dtype=np.int32)
    cdef np.ndarray[REAL_t, ndim=1] work = np.zeros(vector_size, dtype=np.float32)
    # CBOW-specific buffers (allocated but only used if sg=0)
    cdef np.ndarray[REAL_t, ndim=1] neu1 = np.zeros(vector_size, dtype=np.float32)
    cdef np.ndarray[UITYPE_t, ndim=1] context_buffer = np.zeros(2 * window + 1, dtype=np.uint32)
    
    cdef UITYPE_t *indexes_ptr = <UITYPE_t *>np.PyArray_DATA(indexes)
    cdef UITYPE_t *sentence_idx_ptr = <UITYPE_t *>np.PyArray_DATA(sentence_idx)
    cdef ITYPE_t *reduced_windows_ptr = <ITYPE_t *>np.PyArray_DATA(reduced_windows)
    cdef REAL_t *work_ptr = <REAL_t *>np.PyArray_DATA(work)
    cdef REAL_t *neu1_ptr = <REAL_t *>np.PyArray_DATA(neu1)
    cdef UITYPE_t *context_buffer_ptr = <UITYPE_t *>np.PyArray_DATA(context_buffer)
    
    # State variables
    cdef unsigned long long next_random = NEXT_RANDOM
    cdef REAL_t running_loss = 0.0
    cdef REAL_t chunk_loss = 0.0
    cdef long long total_examples = 0
    cdef long long chunk_examples = 0
    cdef long long total_words = 0
    cdef REAL_t alpha = start_alpha
    cdef REAL_t progress
    
    # Chunk processing state
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    cdef int sentences_since_callback = 0
    cdef int word_idx
    cdef int sent_global_idx
    cdef int i
    
    # Process all sentences
    sentence_idx_ptr[0] = 0
    
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
            
            indexes_ptr[effective_words] = <UITYPE_t>word_idx
            effective_words += 1
            
            # Check if buffer is getting full
            if effective_words >= MAX_WORDS_IN_BATCH - 1000:
                break
        
        effective_sentences += 1
        sentence_idx_ptr[effective_sentences] = effective_words
        sentences_since_callback += 1
        
        # Check if we should train the current chunk
        should_train = (
            effective_words >= MAX_WORDS_IN_BATCH - 1000 or
            effective_sentences >= MAX_SENTENCES_IN_BATCH - 1 or
            sentences_since_callback >= callback_every_n_sentences
        )
        
        if should_train and effective_words > 0:
            # Compute reduced windows
            if shrink_windows:
                for i in range(effective_words):
                    reduced_windows_ptr[i] = <ITYPE_t>(random_int32(&next_random) % window)
            else:
                memset(reduced_windows_ptr, 0, effective_words * cython.sizeof(ITYPE_t))
            
            # Compute learning rate: linear interpolation from start_alpha to end_alpha
            if total_words_expected > 0:
                progress = <REAL_t>total_words / <REAL_t>total_words_expected
                if progress > 1.0:
                    progress = 1.0
                alpha = start_alpha + (end_alpha - start_alpha) * progress
            
            # Train on this chunk (RELEASE GIL)
            chunk_loss = 0.0
            chunk_examples = 0
            
            with nogil:
                if _sg == 1:
                    train_chunk_sg(
                        syn0_ptr, syn1neg_ptr, vector_size,
                        indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                        effective_sentences, window, alpha,
                        work_ptr, &next_random, _compute_loss, &chunk_loss, &chunk_examples
                    )
                else:
                    train_chunk_cbow(
                        syn0_ptr, syn1neg_ptr, vector_size,
                        indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                        effective_sentences, window, alpha,
                        neu1_ptr, work_ptr, context_buffer_ptr,
                        &next_random, CBOW_MEAN, _compute_loss, &chunk_loss, &chunk_examples
                    )
            
            running_loss += chunk_loss
            total_examples += chunk_examples
            
            # Progress callback (if provided)
            if progress_callback is not None and sentences_since_callback >= callback_every_n_sentences:
                try:
                    progress_callback(total_words, total_examples, running_loss, alpha)
                except:
                    pass
                sentences_since_callback = 0
            
            # Always reset buffer after training to avoid re-training same data
            effective_words = 0
            effective_sentences = 0
            sentence_idx_ptr[0] = 0
    
    # Train any remaining data
    if effective_words > 0 and effective_sentences > 0:
        if shrink_windows:
            for i in range(effective_words):
                reduced_windows_ptr[i] = <ITYPE_t>(random_int32(&next_random) % window)
        else:
            memset(reduced_windows_ptr, 0, effective_words * cython.sizeof(ITYPE_t))
        
        # Compute learning rate: linear interpolation from start_alpha to end_alpha
        if total_words_expected > 0:
            progress = <REAL_t>total_words / <REAL_t>total_words_expected
            if progress > 1.0:
                progress = 1.0
            alpha = start_alpha + (end_alpha - start_alpha) * progress
        
        chunk_loss = 0.0
        chunk_examples = 0
        
        with nogil:
            if _sg == 1:
                train_chunk_sg(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                    effective_sentences, window, alpha,
                    work_ptr, &next_random, _compute_loss, &chunk_loss, &chunk_examples
                )
            else:
                train_chunk_cbow(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                    effective_sentences, window, alpha,
                    neu1_ptr, work_ptr, context_buffer_ptr,
                    &next_random, CBOW_MEAN, _compute_loss, &chunk_loss, &chunk_examples
                )
        
        running_loss += chunk_loss
        total_examples += chunk_examples
    
    # Final callback
    if progress_callback is not None:
        try:
            progress_callback(total_words, total_examples, running_loss, alpha)
        except:
            pass
    
    # Update global random state
    global NEXT_RANDOM
    NEXT_RANDOM = next_random
    
    return running_loss, total_examples, total_words


# =============================================================================
# Temporal Referencing Variants (for TempRefWord2Vec)
# =============================================================================

def train_epoch_temporal(
    np.float32_t[:, :] syn0,
    np.float32_t[:, :] syn1neg,
    list sentences,
    dict word2idx,
    np.int32_t[:] temporal_index_map,
    np.uint32_t[:] sample_ints,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float start_alpha,
    float end_alpha,
    long long total_words_expected,
    object progress_callback,
    int callback_every_n_sentences,
    bint sg,
    bint compute_loss=True,
):
    """
    Train Word2Vec (Skip-gram or CBOW) with temporal mapping for one full epoch.
    
    Context words are mapped to base forms, center words keep temporal variant.
    Learning rate linearly interpolates from start_alpha to end_alpha.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size)
        syn1neg: Output word vectors (vocab_size x vector_size)
        sentences: List of tokenized sentences (list of list of str)
        word2idx: Dictionary mapping words to indices
        temporal_index_map: Array mapping word indices to base form indices
        sample_ints: Pre-computed subsampling thresholds (uint32)
        use_subsampling: Whether to apply subsampling
        window: Context window size
        shrink_windows: Whether to randomly shrink windows
        start_alpha: Learning rate at the start of this epoch
        end_alpha: Learning rate at the end of this epoch
        total_words_expected: Expected in-vocab words for this epoch (for LR decay)
        progress_callback: Function(words_processed, loss) called periodically
        callback_every_n_sentences: How often to call progress callback
        sg: If True, use Skip-gram; if False, use CBOW
        compute_loss: Whether to compute and track loss (default True)
    
    Returns:
        Tuple of (total_loss, total_examples_trained, total_words_processed)
    """
    cdef int vector_size = syn0.shape[1]
    cdef int num_sentences = len(sentences)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef int _sg = 1 if sg else 0
    
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef ITYPE_t *temporal_map_ptr = &temporal_index_map[0]
    
    cdef np.ndarray[UITYPE_t, ndim=1] indexes = np.zeros(MAX_WORDS_IN_BATCH, dtype=np.uint32)
    cdef np.ndarray[UITYPE_t, ndim=1] mapped_indexes = np.zeros(MAX_WORDS_IN_BATCH, dtype=np.uint32)
    cdef np.ndarray[UITYPE_t, ndim=1] sentence_idx = np.zeros(MAX_SENTENCES_IN_BATCH + 1, dtype=np.uint32)
    cdef np.ndarray[ITYPE_t, ndim=1] reduced_windows = np.zeros(MAX_WORDS_IN_BATCH, dtype=np.int32)
    cdef np.ndarray[REAL_t, ndim=1] work = np.zeros(vector_size, dtype=np.float32)
    # CBOW-specific buffers (allocated but only used if sg=0)
    cdef np.ndarray[REAL_t, ndim=1] neu1 = np.zeros(vector_size, dtype=np.float32)
    cdef np.ndarray[UITYPE_t, ndim=1] context_buffer = np.zeros(2 * window + 1, dtype=np.uint32)
    
    cdef UITYPE_t *indexes_ptr = <UITYPE_t *>np.PyArray_DATA(indexes)
    cdef UITYPE_t *mapped_indexes_ptr = <UITYPE_t *>np.PyArray_DATA(mapped_indexes)
    cdef UITYPE_t *sentence_idx_ptr = <UITYPE_t *>np.PyArray_DATA(sentence_idx)
    cdef ITYPE_t *reduced_windows_ptr = <ITYPE_t *>np.PyArray_DATA(reduced_windows)
    cdef REAL_t *work_ptr = <REAL_t *>np.PyArray_DATA(work)
    cdef REAL_t *neu1_ptr = <REAL_t *>np.PyArray_DATA(neu1)
    cdef UITYPE_t *context_buffer_ptr = <UITYPE_t *>np.PyArray_DATA(context_buffer)
    
    cdef unsigned long long next_random = NEXT_RANDOM
    cdef REAL_t running_loss = 0.0
    cdef REAL_t chunk_loss = 0.0
    cdef long long total_examples = 0
    cdef long long chunk_examples = 0
    cdef long long total_words = 0
    cdef REAL_t alpha = start_alpha
    cdef REAL_t progress
    
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    cdef int sentences_since_callback = 0
    cdef int word_idx, mapped_idx
    cdef int sent_global_idx
    cdef int i
    
    sentence_idx_ptr[0] = 0
    
    for sent_global_idx in range(num_sentences):
        sent = sentences[sent_global_idx]
        
        if sent is None or len(sent) == 0:
            continue
        
        for token in sent:
            if token not in word2idx:
                continue
            
            word_idx = word2idx[token]
            mapped_idx = temporal_map_ptr[word_idx]
            
            if mapped_idx < 0:
                continue
            
            total_words += 1
            
            if use_subsampling:
                if sample_ints_ptr[word_idx] < (random_int32(&next_random) & 0xFFFFFFFF):
                    continue
            
            indexes_ptr[effective_words] = <UITYPE_t>word_idx
            mapped_indexes_ptr[effective_words] = <UITYPE_t>mapped_idx
            effective_words += 1
            
            if effective_words >= MAX_WORDS_IN_BATCH - 1000:
                break
        
        effective_sentences += 1
        sentence_idx_ptr[effective_sentences] = effective_words
        sentences_since_callback += 1
        
        should_train = (
            effective_words >= MAX_WORDS_IN_BATCH - 1000 or
            effective_sentences >= MAX_SENTENCES_IN_BATCH - 1 or
            sentences_since_callback >= callback_every_n_sentences
        )
        
        if should_train and effective_words > 0:
            if shrink_windows:
                for i in range(effective_words):
                    reduced_windows_ptr[i] = <ITYPE_t>(random_int32(&next_random) % window)
            else:
                memset(reduced_windows_ptr, 0, effective_words * cython.sizeof(ITYPE_t))
            
            # Compute learning rate: linear interpolation from start_alpha to end_alpha
            if total_words_expected > 0:
                progress = <REAL_t>total_words / <REAL_t>total_words_expected
                if progress > 1.0:
                    progress = 1.0
                alpha = start_alpha + (end_alpha - start_alpha) * progress
            
            chunk_loss = 0.0
            chunk_examples = 0
            
            with nogil:
                if _sg == 1:
                    train_chunk_sg_temporal(
                        syn0_ptr, syn1neg_ptr, vector_size,
                        indexes_ptr, mapped_indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                        effective_sentences, window, alpha,
                        work_ptr, &next_random, _compute_loss, &chunk_loss, &chunk_examples
                    )
                else:
                    train_chunk_cbow_temporal(
                        syn0_ptr, syn1neg_ptr, vector_size,
                        indexes_ptr, mapped_indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                        effective_sentences, window, alpha,
                        neu1_ptr, work_ptr, context_buffer_ptr,
                        &next_random, CBOW_MEAN, _compute_loss, &chunk_loss, &chunk_examples
                    )
            
            running_loss += chunk_loss
            total_examples += chunk_examples
            
            if progress_callback is not None and sentences_since_callback >= callback_every_n_sentences:
                try:
                    progress_callback(total_words, total_examples, running_loss, alpha)
                except:
                    pass
                sentences_since_callback = 0
            
            # Always reset buffer after training
            effective_words = 0
            effective_sentences = 0
            sentence_idx_ptr[0] = 0
    
    # Train remaining
    if effective_words > 0 and effective_sentences > 0:
        if shrink_windows:
            for i in range(effective_words):
                reduced_windows_ptr[i] = <ITYPE_t>(random_int32(&next_random) % window)
        else:
            memset(reduced_windows_ptr, 0, effective_words * cython.sizeof(ITYPE_t))
        
        # Compute learning rate: linear interpolation from start_alpha to end_alpha
        if total_words_expected > 0:
            progress = <REAL_t>total_words / <REAL_t>total_words_expected
            if progress > 1.0:
                progress = 1.0
            alpha = start_alpha + (end_alpha - start_alpha) * progress
        
        chunk_loss = 0.0
        chunk_examples = 0
        
        with nogil:
            if _sg == 1:
                train_chunk_sg_temporal(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes_ptr, mapped_indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                    effective_sentences, window, alpha,
                    work_ptr, &next_random, _compute_loss, &chunk_loss, &chunk_examples
                )
            else:
                train_chunk_cbow_temporal(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes_ptr, mapped_indexes_ptr, sentence_idx_ptr, reduced_windows_ptr,
                    effective_sentences, window, alpha,
                    neu1_ptr, work_ptr, context_buffer_ptr,
                    &next_random, CBOW_MEAN, _compute_loss, &chunk_loss, &chunk_examples
                )
        
        running_loss += chunk_loss
        total_examples += chunk_examples
    
    if progress_callback is not None:
        try:
            progress_callback(total_words, total_examples, running_loss, alpha)
        except:
            pass
    
    global NEXT_RANDOM
    NEXT_RANDOM = next_random
    
    return running_loss, total_examples, total_words


cdef void train_chunk_sg_temporal(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *indexes,
    UITYPE_t *mapped_indexes,
    UITYPE_t *sentence_idx,
    ITYPE_t *reduced_windows,
    int num_sentences,
    int window,
    REAL_t alpha,
    REAL_t *work,
    unsigned long long *next_random,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram temporal on a chunk (nogil)."""
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
                
                # Center word: original index, Context word: mapped to base
                next_random[0] = train_sg_pair(
                    syn0, syn1neg, size,
                    indexes[i], mapped_indexes[j],
                    alpha, work, next_random[0], _compute_loss, running_loss
                )
                examples_trained[0] += 1


cdef void train_chunk_cbow_temporal(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *indexes,
    UITYPE_t *mapped_indexes,
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
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train CBOW temporal on a chunk (nogil)."""
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
            
            # Collect MAPPED context indices
            ctx_count = 0
            for m in range(j, k):
                if m == i:
                    continue
                context_buffer[ctx_count] = mapped_indexes[m]
                ctx_count += 1
            
            if ctx_count == 0:
                continue
            
            # Center word uses original index
            next_random[0] = train_cbow_pair(
                syn0, syn1neg, size,
                context_buffer, ctx_count, indexes[i],
                alpha, neu1, work, next_random[0],
                cbow_mean, _compute_loss, running_loss
            )
            examples_trained[0] += 1
