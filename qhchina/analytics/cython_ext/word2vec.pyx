"""
Fast Word2Vec training operations implemented in Cython.

Training loop design and BLAS integration inspired by Gensim's word2vec_inner.pyx.

Key optimizations:
- Raw BLAS function pointers for zero-overhead calls
- Vocabulary lookup inside Cython (avoids intermediate allocations)
- Fixed-size stack-allocated buffers for maximum performance
- GIL released during actual training computations
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

# Import PyCapsule at module level for extracting raw BLAS pointers
from cpython.pycapsule cimport PyCapsule_GetPointer

# Import scipy BLAS module for extracting raw function pointers
import scipy.linalg.blas as fblas

# =============================================================================
# BLAS Function Pointer Types and Extraction (like Gensim)
# =============================================================================

# Function pointer types for BLAS operations
ctypedef float (*sdot_ptr)(const int *N, const float *X, const int *incX,
                           const float *Y, const int *incY) noexcept nogil
ctypedef void (*saxpy_ptr)(const int *N, const float *alpha, const float *X,
                           const int *incX, float *Y, const int *incY) noexcept nogil
ctypedef void (*sscal_ptr)(const int *N, const float *alpha, float *X,
                           const int *incX) noexcept nogil
ctypedef double (*dsdot_ptr)(const int *N, const float *X, const int *incX,
                             const float *Y, const int *incY) noexcept nogil

# Raw BLAS function pointers - extracted at module load for zero-overhead calls
cdef sdot_ptr our_sdot
cdef saxpy_ptr our_saxpy
cdef sscal_ptr our_sscal

# For detecting whether sdot returns float or double
cdef dsdot_ptr dsdot_ptr_global

# Wrapper functions for when sdot returns different types
cdef float our_sdot_double(const int *N, const float *X, const int *incX,
                           const float *Y, const int *incY) noexcept nogil:
    """Wrapper when BLAS sdot returns double."""
    return <float>dsdot_ptr_global(N, X, incX, Y, incY)

cdef float our_sdot_float(const int *N, const float *X, const int *incX,
                          const float *Y, const int *incY) noexcept nogil:
    """Direct call when BLAS sdot returns float."""
    return (<sdot_ptr>dsdot_ptr_global)(N, X, incX, Y, incY)

# Fallback implementations when BLAS is not available
cdef float our_sdot_noblas(const int *N, const float *X, const int *incX,
                           const float *Y, const int *incY) noexcept nogil:
    """Pure Cython dot product fallback."""
    cdef int i
    cdef float result = 0.0
    for i in range(N[0]):
        result += X[i * incX[0]] * Y[i * incY[0]]
    return result

cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X,
                           const int *incX, float *Y, const int *incY) noexcept nogil:
    """Pure Cython saxpy fallback."""
    cdef int i
    for i in range(N[0]):
        Y[i * incY[0]] += alpha[0] * X[i * incX[0]]

cdef void our_sscal_noblas(const int *N, const float *alpha, float *X,
                           const int *incX) noexcept nogil:
    """Pure Cython sscal fallback."""
    cdef int i
    for i in range(N[0]):
        X[i * incX[0]] *= alpha[0]

# Define C types
ctypedef np.float32_t REAL_t
ctypedef np.int32_t ITYPE_t
ctypedef np.uint32_t UITYPE_t

# Constants
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6.0

# Maximum words per batch - fixed at compile time (10240)
# This determines the size of stack-allocated buffers for training.
# Batches exceeding this limit are safely truncated.
DEF MAX_WORDS_IN_BATCH = 10240

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

def _init_blas():
    """Extract raw BLAS function pointers from scipy for zero-overhead calls.
    
    BLAS provides optimized vector operations (dot product, saxpy) that are
    significantly faster than pure Python/Cython loops.
    
    Returns: 0=sdot returns double, 1=sdot returns float, 2=no BLAS (fallback)
    """
    global our_sdot, our_saxpy, our_sscal, dsdot_ptr_global
    
    cdef float x[1]
    cdef float y[1]
    cdef int size = 1
    cdef double d_res
    cdef float *p_res
    cdef float expected = 0.1
    
    x[0] = 10.0
    y[0] = 0.01
    
    # Try to extract raw function pointers from scipy BLAS
    try:
        # Extract raw C function pointers using PyCapsule
        # scipy exposes these via _cpointer attribute
        dsdot_ptr_global = <dsdot_ptr>PyCapsule_GetPointer(fblas.sdot._cpointer, NULL)
        our_saxpy = <saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer, NULL)
        our_sscal = <sscal_ptr>PyCapsule_GetPointer(fblas.sscal._cpointer, NULL)
        
        # Test whether sdot returns float or double
        d_res = dsdot_ptr_global(&size, x, &ONE, y, &ONE)
        p_res = <float *>&d_res
        
        if abs(d_res - expected) < 0.0001:
            # sdot returns double, need wrapper
            our_sdot = our_sdot_double
            return 0
        elif abs(p_res[0] - expected) < 0.0001:
            # sdot returns float, can use directly
            our_sdot = our_sdot_float
            return 1
        else:
            # Unexpected behavior, fall back to pure Cython
            our_sdot = our_sdot_noblas
            our_saxpy = our_saxpy_noblas
            our_sscal = our_sscal_noblas
            return 2
    except:
        # BLAS not available, use pure Cython fallback
        our_sdot = our_sdot_noblas
        our_saxpy = our_saxpy_noblas
        our_sscal = our_sscal_noblas
        return 2

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

# Initialize BLAS and tables at module import
FAST_VERSION = _init_blas()
_init_tables()


# =============================================================================
# Fast LCG Random Number Generator
# =============================================================================

cdef inline unsigned long long random_int32(unsigned long long *next_random) noexcept nogil:
    """Linear Congruential Generator for fast random numbers inside nogil blocks.
    
    LCG parameters (multiplier=25214903917, increment=11, modulus=2^48) chosen for
    good statistical properties and fast computation.
    """
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
    """Binary search on cumulative table to find word index for negative sampling."""
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
    const UITYPE_t center_index,
    const UITYPE_t context_index,
    const REAL_t alpha,
    REAL_t *work,
    unsigned long long next_random,
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss
) noexcept nogil:
    """Train on a single (center, context) word pair with negative sampling.
    
    Updates:
      - syn1neg[center_index]: Output embedding of center word (positive target)
      - syn0[context_index]: Input embedding of context word
    
    Why syn0 holds CONTEXT embeddings (not center), despite theory suggesting otherwise:
    
    In skip-gram, each center word generates multiple (center, context) pairs from its
    window. If syn0 held center embeddings, syn0[center] would receive multiple correlated
    updates in rapid succession (one per context word in the same window). These updates
    all derive from the same local context, leading to "bursty" correlated gradients.
    
    By making syn0 the context matrix instead:
      - Each syn0[word] update comes from a DIFFERENT center word elsewhere in the corpus
      - Updates to the same syn0 row are spread across training, not clustered
      - The discarded syn1neg absorbs the bursty updates (multiple per window)
    
    This produces better-quality embeddings in syn0, which is what we return for queries.
    """
    cdef long long row1 = <long long>context_index * <long long>size
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d
    
    memset(work, 0, size * cython.sizeof(REAL_t))
    
    for d in range(negative + 1):
        if d == 0:
            target_index = center_index  # Positive sample = center word
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len - 1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == center_index:
                continue
            label = <REAL_t>0.0
        
        row2 = <long long>target_index * <long long>size
        f_dot = our_sdot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        
        if _compute_loss == 1:
            f_dot_for_loss = (f_dot if d == 0 else -f_dot)
            if f_dot_for_loss <= -MAX_EXP or f_dot_for_loss >= MAX_EXP:
                pass
            else:
                log_e_f_dot = LOG_TABLE[<int>((f_dot_for_loss + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                running_loss[0] = running_loss[0] - log_e_f_dot
        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
    
    our_saxpy(&size, &ONEF, work, &ONE, &syn0[row1], &ONE)
    
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
    """Train on a single CBOW example: predict center_index from context_indices."""
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, count, inv_count, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d, m
    
    # Build combined context vector
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(context_len):
        our_saxpy(&size, &ONEF, &syn0[<long long>context_indices[m] * <long long>size], &ONE, neu1, &ONE)
        count += ONEF
    
    if count < <REAL_t>0.5:
        return next_random
    
    inv_count = ONEF / count
    if cbow_mean:
        our_sscal(&size, &inv_count, neu1, &ONE)
    
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
        f_dot = our_sdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        
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
        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    
    if not cbow_mean:
        our_sscal(&size, &inv_count, work, &ONE)
    
    for m in range(context_len):
        our_saxpy(&size, &ONEF, work, &ONE, &syn0[<long long>context_indices[m] * <long long>size], &ONE)
    
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
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a batch of indexed sentences (nogil).
    
    For each word at position i, trains pairs with context words in window.
    indexes[i] = center word, indexes[j] = context word.
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
                
                # indexes[i] = center, indexes[j] = context
                next_random[0] = train_sg_pair(
                    syn0, syn1neg, size,
                    indexes[i], indexes[j],  # center, context
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
    """Train CBOW on a batch: for each center word, predict from surrounding context."""
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
    np.float32_t[:] work,
    np.float32_t[:] neu1,
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
    """Train Word2Vec on a batch of sentences.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size).
        syn1neg: Output word vectors for negative sampling.
        sentences: List of tokenized sentences.
        word2idx: Word to index mapping.
        sample_ints: Subsampling thresholds.
        cum_table: Cumulative table for negative sampling.
        work: Pre-allocated work buffer (vector_size,). Reused across batches.
        neu1: Pre-allocated CBOW context buffer (vector_size,). Reused across batches.
        use_subsampling: Apply subsampling of frequent words.
        window: Context window size.
        shrink_windows: Randomly reduce window size per word.
        alpha: Learning rate.
        sg: True for Skip-gram, False for CBOW.
        negative: Number of negative samples per positive.
        cbow_mean: Average context vectors (CBOW only).
        random_seed: LCG seed (fresh per batch from Python RNG).
        compute_loss: Track loss during training.
    
    Returns:
        (batch_loss, examples_trained, words_processed, final_random_state)
    """
    cdef int vector_size = syn0.shape[1]
    cdef int num_sentences = len(sentences)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef int _sg = 1 if sg else 0
    cdef unsigned long long cum_table_len = len(cum_table)
    
    # Get raw pointers
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef UITYPE_t *cum_table_ptr = &cum_table[0]
    cdef REAL_t *work_ptr = &work[0]
    cdef REAL_t *neu1_ptr = &neu1[0]
    
    # Stack-allocated buffers for batch data
    cdef UITYPE_t indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t sentence_idx[MAX_WORDS_IN_BATCH + 1]
    cdef ITYPE_t reduced_windows[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t context_buffer[50]  # 2 * max_window
    
    # State variables
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
            
            # Subsampling (matches Gensim exactly)
            if use_subsampling:
                if sample_ints_ptr[word_idx] < random_int32(&next_random):
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
    np.float32_t[:] work,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float alpha,
    int negative,
    unsigned long long random_seed,
    bint compute_loss=True,
):
    """Train Skip-gram with temporal mapping (for TempRefWord2Vec).
    
    Reverses the standard Skip-gram framing to align with negative sampling gradient flow:
    - CENTER words = base forms (mapped via temporal_index_map) -> syn1neg (noisy updates)
    - CONTEXT words = temporal variants (original tokens) -> syn0 (clean updates)
    
    This gives temporal variants high-quality embeddings in syn0 (W), so all queries
    use W only. Base forms in syn1neg provide a stable reference frame.
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size).
        syn1neg: Output word vectors (vocab_size x vector_size).
        sentences: List of tokenized sentences.
        word2idx: Word to index mapping.
        temporal_index_map: Maps vocab index -> base form index.
        sample_ints: Subsampling thresholds.
        cum_table: Cumulative table for negative sampling.
        work: Pre-allocated work buffer (vector_size,). Reused across batches.
        use_subsampling: Apply subsampling of frequent words.
        window: Context window size.
        shrink_windows: Randomly reduce window size per word.
        alpha: Learning rate.
        negative: Number of negative samples.
        random_seed: LCG seed.
        compute_loss: Track loss during training.
    
    Returns:
        (batch_loss, examples_trained, words_processed, final_random_state)
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
    cdef REAL_t *work_ptr = &work[0]
    
    # Stack-allocated buffers for batch data
    cdef UITYPE_t center_indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t context_indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t sentence_idx[MAX_WORDS_IN_BATCH + 1]
    cdef ITYPE_t reduced_windows[MAX_WORDS_IN_BATCH]
    
    # State variables
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
                if sample_ints_ptr[word_idx] < random_int32(&next_random):
                    continue
            
            # Store center index as BASE FORM (via temporal_index_map)
            # For temporal variants like "民_宋", this maps to base "民"
            # For regular words, identity mapping (index maps to itself)
            # Base forms go to syn1neg (noisy updates, stable reference)
            center_indexes[effective_words] = <UITYPE_t>temporal_map_ptr[word_idx]
            
            # Store context index as ORIGINAL token (temporal variant if applicable)
            # Temporal variants go to syn0 (clean updates, informative embeddings)
            context_indexes[effective_words] = <UITYPE_t>word_idx
            
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
    UITYPE_t *cum_table,
    unsigned long long cum_table_len,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a batch with temporal mapping (nogil).
    
    center_indexes[i] = base form (mapped via temporal_index_map) -> syn1neg (noisy)
    context_indexes[j] = original token (temporal variant) -> syn0 (clean)
    
    Result: temporal variants get high-quality syn0 embeddings, base forms get noisy syn1neg.
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
                
                # center_indexes[i] = base form -> syn1neg (noisy updates)
                # context_indexes[j] = temporal variant -> syn0 (clean updates)
                next_random[0] = train_sg_pair(
                    syn0, syn1neg, size,
                    center_indexes[i],   # center (base form)
                    context_indexes[j],  # context (temporal variant)
                    alpha, work, next_random[0],
                    cum_table, cum_table_len, negative,
                    _compute_loss, running_loss
                )
                examples_trained[0] += 1
