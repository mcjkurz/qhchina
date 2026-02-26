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
from libc.stdlib cimport malloc, free

# Import PyCapsule at module level for extracting raw BLAS pointers
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.dict cimport PyDict_GetItemWithError
from cpython.object cimport PyObject
from cpython.exc cimport PyErr_Occurred
from cpython.long cimport PyLong_AsLong

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
# Alias Method for O(1) Negative Sampling
# =============================================================================

cdef inline UITYPE_t alias_draw(
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    unsigned long long *next_random,
) noexcept nogil:
    """O(1) negative sample draw using the alias method.
    
    Uses two random numbers from the LCG:
    - First selects a column uniformly in [0, vocab_size)
    - Second compares against the alias probability threshold
    """
    cdef unsigned long long r1, r2
    cdef UITYPE_t col
    r1 = random_int32(next_random)
    col = <UITYPE_t>(r1 % vocab_size)
    r2 = random_int32(next_random)
    if <UITYPE_t>r2 < alias_prob[col]:
        return col
    else:
        return alias_index[col]


# =============================================================================
# Core Skip-gram Training (nogil)
# =============================================================================

cdef inline void train_sg_pair(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    const UITYPE_t center_index,
    const UITYPE_t context_index,
    const REAL_t alpha,
    REAL_t *work,
    unsigned long long *next_random,
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss
) noexcept nogil:
    """Train on a single (center, context) word pair with negative sampling.
    
    Standard skip-gram framing where center word predicts context:
      - syn0[center_index]: Input embedding of center word (row1)
      - syn1neg[context_index]: Output embedding of context word (positive target, d=0)
      - syn1neg[negative]: Output embeddings of negative samples (d>0)
    
    Uses O(1) alias method for drawing negative samples.
    """
    cdef long long row1 = <long long>center_index * <long long>size
    cdef long long row2
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d
    
    memset(work, 0, size * cython.sizeof(REAL_t))
    
    for d in range(negative + 1):
        if d == 0:
            target_index = context_index
            label = ONEF
        else:
            target_index = alias_draw(alias_prob, alias_index, vocab_size, next_random)
            if target_index == context_index:
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


# =============================================================================
# Core CBOW Training (nogil)
# =============================================================================

cdef inline void train_cbow_pair(
    REAL_t *syn0,
    REAL_t *syn1neg,
    const int size,
    UITYPE_t *context_indices,
    int context_len,
    const UITYPE_t center_index,
    const REAL_t alpha,
    REAL_t *neu1,
    REAL_t *work,
    unsigned long long *next_random,
    bint cbow_mean,
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss
) noexcept nogil:
    """Train on a single CBOW example: predict center_index from context_indices.
    
    Uses O(1) alias method for drawing negative samples.
    """
    cdef long long row2
    cdef REAL_t f, g, label, f_dot, f_dot_for_loss, count, inv_count, log_e_f_dot
    cdef UITYPE_t target_index
    cdef int d, m
    
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(context_len):
        our_saxpy(&size, &ONEF, &syn0[<long long>context_indices[m] * <long long>size], &ONE, neu1, &ONE)
        count += ONEF
    
    if count < <REAL_t>0.5:
        return
    
    inv_count = ONEF / count
    if cbow_mean:
        our_sscal(&size, &inv_count, neu1, &ONE)
    
    memset(work, 0, size * cython.sizeof(REAL_t))
    
    for d in range(negative + 1):
        if d == 0:
            target_index = center_index
            label = ONEF
        else:
            target_index = alias_draw(alias_prob, alias_index, vocab_size, next_random)
            if target_index == center_index:
                continue
            label = <REAL_t>0.0
        
        row2 = <long long>target_index * <long long>size
        f_dot = our_sdot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        
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
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    
    if not cbow_mean:
        our_sscal(&size, &inv_count, work, &ONE)
    
    for m in range(context_len):
        our_saxpy(&size, &ONEF, work, &ONE, &syn0[<long long>context_indices[m] * <long long>size], &ONE)


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
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a batch of indexed sentences (nogil).
    
    For each word at position i, trains pairs with context words in window.
    Center word (indexes[i]) in syn0, context word (indexes[j]) as positive target in syn1neg.
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
                
                train_sg_pair(
                    syn0, syn1neg, size,
                    indexes[i], indexes[j],
                    alpha, work, next_random,
                    alias_prob, alias_index, vocab_size, negative,
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
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    int negative,
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
            
            ctx_count = 0
            for m in range(j, k):
                if m == i:
                    continue
                context_buffer[ctx_count] = indexes[m]
                ctx_count += 1
            
            if ctx_count == 0:
                continue
            
            train_cbow_pair(
                syn0, syn1neg, size,
                context_buffer, ctx_count, indexes[i],
                alpha, neu1, work, next_random,
                cbow_mean,
                alias_prob, alias_index, vocab_size, negative,
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
    np.uint32_t[:] alias_prob,
    np.uint32_t[:] alias_index,
    np.float32_t[:] work,
    np.float32_t[:] neu1,
    np.uint32_t[:] context_buffer_arr,
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
        alias_prob: Alias method probability thresholds (uint32).
        alias_index: Alias method fallback indices (uint32).
        work: Pre-allocated work buffer (vector_size,). Reused across batches.
        neu1: Pre-allocated CBOW context buffer (vector_size,). Reused across batches.
        context_buffer_arr: Pre-allocated context index buffer (2*window,). CBOW only.
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
    cdef unsigned long long vocab_size = len(alias_prob)
    
    # Get raw pointers
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef UITYPE_t *alias_prob_ptr = &alias_prob[0]
    cdef UITYPE_t *alias_index_ptr = &alias_index[0]
    cdef REAL_t *work_ptr = &work[0]
    cdef REAL_t *neu1_ptr = &neu1[0]
    cdef UITYPE_t *context_buffer = &context_buffer_arr[0]
    
    # Stack-allocated buffers for batch data
    cdef UITYPE_t indexes[MAX_WORDS_IN_BATCH]
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
    cdef PyObject *result
    
    sentence_idx[0] = 0
    
    for sent_global_idx in range(num_sentences):
        sent = sentences[sent_global_idx]
        
        if sent is None or len(sent) == 0:
            continue
        
        for token in sent:
            result = PyDict_GetItemWithError(word2idx, token)
            if result == NULL:
                if PyErr_Occurred():
                    raise
                continue
            
            word_idx = <int>PyLong_AsLong(<object>result)
            total_words += 1
            
            if use_subsampling:
                if sample_ints_ptr[word_idx] < random_int32(&next_random):
                    continue
            
            indexes[effective_words] = <UITYPE_t>word_idx
            effective_words += 1
            
            if effective_words >= MAX_WORDS_IN_BATCH:
                break
        
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words
        
        if effective_words >= MAX_WORDS_IN_BATCH:
            break
    
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
                    alias_prob_ptr, alias_index_ptr, vocab_size, negative,
                    _compute_loss, &running_loss, &total_examples
                )
            else:
                train_batch_cbow(
                    syn0_ptr, syn1neg_ptr, vector_size,
                    indexes, sentence_idx, reduced_windows,
                    effective_sentences, window, alpha,
                    neu1_ptr, work_ptr, context_buffer,
                    &next_random, cbow_mean,
                    alias_prob_ptr, alias_index_ptr, vocab_size, negative,
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
    np.uint32_t[:] alias_prob,
    np.uint32_t[:] alias_index,
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
    
    Temporal referencing training where:
    - CENTER words = temporal variants (tagged words like "民_宋") -> syn0
    - CONTEXT words = base forms (untagged words like "民") -> syn1neg (positive targets)
    - Negative samples are drawn from the alias table distribution -> syn1neg
    
    Args:
        syn0: Input word vectors (vocab_size x vector_size).
        syn1neg: Output word vectors (vocab_size x vector_size).
        sentences: List of tokenized sentences.
        word2idx: Word to index mapping.
        temporal_index_map: Maps vocab index -> base form index.
        sample_ints: Subsampling thresholds.
        alias_prob: Alias method probability thresholds (uint32).
        alias_index: Alias method fallback indices (uint32).
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
    cdef unsigned long long vocab_size = len(alias_prob)
    
    # Get raw pointers
    cdef REAL_t *syn0_ptr = &syn0[0, 0]
    cdef REAL_t *syn1neg_ptr = &syn1neg[0, 0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef ITYPE_t *temporal_map_ptr = &temporal_index_map[0]
    cdef UITYPE_t *alias_prob_ptr = &alias_prob[0]
    cdef UITYPE_t *alias_index_ptr = &alias_index[0]
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
    cdef PyObject *result
    
    sentence_idx[0] = 0
    
    for sent_global_idx in range(num_sentences):
        sent = sentences[sent_global_idx]
        
        if sent is None or len(sent) == 0:
            continue
        
        for token in sent:
            result = PyDict_GetItemWithError(word2idx, token)
            if result == NULL:
                if PyErr_Occurred():
                    raise
                continue
            
            word_idx = <int>PyLong_AsLong(<object>result)
            total_words += 1
            
            if use_subsampling:
                if sample_ints_ptr[word_idx] < random_int32(&next_random):
                    continue
            
            center_indexes[effective_words] = <UITYPE_t>word_idx
            context_indexes[effective_words] = <UITYPE_t>temporal_map_ptr[word_idx]
            
            effective_words += 1
            
            if effective_words >= MAX_WORDS_IN_BATCH:
                break
        
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words
        
        if effective_words >= MAX_WORDS_IN_BATCH:
            break
    
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
                alias_prob_ptr, alias_index_ptr, vocab_size, negative,
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
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long vocab_size,
    int negative,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained
) noexcept nogil:
    """Train skip-gram on a batch with temporal mapping (nogil).
    
    center_indexes[i] = temporal variant (tagged) -> syn0
    context_indexes[j] = base form (untagged) -> syn1neg (positive target)
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
                
                train_sg_pair(
                    syn0, syn1neg, size,
                    center_indexes[i],
                    context_indexes[j],
                    alpha, work, next_random,
                    alias_prob, alias_index, vocab_size, negative,
                    _compute_loss, running_loss
                )
                examples_trained[0] += 1


# =============================================================================
# DynamicWord2Vec Training with Temporal Regularization
# =============================================================================

cdef inline void apply_temporal_regularization(
    REAL_t *U_all,
    int T,
    int t,
    int vocab_size,
    int vector_size,
    UITYPE_t word_idx,
    REAL_t temporal_lambda,
    REAL_t alpha,
) noexcept nogil:
    """
    Apply temporal L2 regularization to U[t, word_idx].

    Gradient computation:
    - Interior slices (0 < t < T-1): grad = λ * [2·U[t] - U[t-1] - U[t+1]]
    - Left boundary (t=0): grad = λ * [U[t] - U[t+1]]
    - Right boundary (t=T-1): grad = λ * [U[t] - U[t-1]]

    The gradient is multiplied by the learning rate alpha and subtracted from U[t].

    Args:
        U_all: Flat array of all U embeddings [T * vocab_size * vector_size]
        T: Number of time slices
        t: Current time slice index
        vocab_size: Vocabulary size
        vector_size: Embedding dimensionality
        word_idx: Word index being regularized
        temporal_lambda: Regularization strength
        alpha: Learning rate
    """
    cdef int d
    cdef long long offset_t = (<long long>t * vocab_size + word_idx) * vector_size
    cdef long long offset_prev, offset_next
    cdef REAL_t grad_val
    cdef REAL_t update_scale = temporal_lambda * alpha

    if T == 1:
        # Single time slice - no regularization
        return

    if t == 0:
        # Left boundary: grad = λ * [U[t] - U[t+1]]
        offset_next = (<long long>(t + 1) * vocab_size + word_idx) * vector_size
        for d in range(vector_size):
            grad_val = U_all[offset_t + d] - U_all[offset_next + d]
            U_all[offset_t + d] -= update_scale * grad_val
    elif t == T - 1:
        # Right boundary: grad = λ * [U[t] - U[t-1]]
        offset_prev = (<long long>(t - 1) * vocab_size + word_idx) * vector_size
        for d in range(vector_size):
            grad_val = U_all[offset_t + d] - U_all[offset_prev + d]
            U_all[offset_t + d] -= update_scale * grad_val
    else:
        # Interior: grad = λ * [2·U[t] - U[t-1] - U[t+1]]
        offset_prev = (<long long>(t - 1) * vocab_size + word_idx) * vector_size
        offset_next = (<long long>(t + 1) * vocab_size + word_idx) * vector_size
        for d in range(vector_size):
            grad_val = 2.0 * U_all[offset_t + d] - U_all[offset_prev + d] - U_all[offset_next + d]
            U_all[offset_t + d] -= update_scale * grad_val


cdef void train_batch_sg_dynamic(
    REAL_t *U_all,
    REAL_t *V_all,
    const int T,
    const int vocab_size,
    const int size,
    UITYPE_t *indexes,
    UITYPE_t *sentence_idx,
    ITYPE_t *time_indices,
    ITYPE_t *reduced_windows,
    int num_sentences,
    int window,
    REAL_t alpha,
    REAL_t *work,
    unsigned long long *next_random,
    UITYPE_t *alias_prob,
    UITYPE_t *alias_index,
    unsigned long long alias_vocab_size,
    int negative,
    REAL_t temporal_lambda,
    bint temporal_reg_V,
    const int _compute_loss,
    REAL_t *running_loss,
    long long *examples_trained,
    unsigned char *seen_U,
    UITYPE_t *touched_U_words,
    ITYPE_t *touched_U_times,
    int *n_touched_U,
    unsigned char *seen_V,
    UITYPE_t *touched_V_words,
    ITYPE_t *touched_V_times,
    int *n_touched_V,
) noexcept nogil:
    """
    Train skip-gram with temporal regularization on a batch (nogil).

    All skip-gram updates are performed first, then temporal regularization
    is applied once per unique (word, time_slice) pair. This prevents
    frequent words from receiving disproportionately more regularization.

    The seen/touched buffers are pre-allocated by the caller:
    - seen_U/seen_V: byte arrays of size T * vocab_size (dedup bitsets)
    - touched_U/V_words + touched_U/V_times: parallel arrays for unique pairs
    - n_touched_U/V: counters for how many unique pairs collected
    """
    cdef int sent_idx, i, j, k, t
    cdef int idx_start, idx_end, ctx_start, ctx_end
    cdef UITYPE_t center_idx, context_idx
    cdef REAL_t *U_ptr
    cdef REAL_t *V_ptr
    cdef long long seen_offset

    # --- Phase 1: Skip-gram updates + collect unique (word, t) pairs ---
    for sent_idx in range(num_sentences):
        idx_start = sentence_idx[sent_idx]
        idx_end = sentence_idx[sent_idx + 1]
        t = time_indices[sent_idx]

        U_ptr = U_all + (<long long>t * vocab_size * size)
        V_ptr = V_all + (<long long>t * vocab_size * size)
        seen_offset = <long long>t * vocab_size

        for i in range(idx_start, idx_end):
            center_idx = indexes[i]

            ctx_start = i - window + reduced_windows[i]
            if ctx_start < idx_start:
                ctx_start = idx_start
            ctx_end = i + window + 1 - reduced_windows[i]
            if ctx_end > idx_end:
                ctx_end = idx_end

            for j in range(ctx_start, ctx_end):
                if j == i:
                    continue

                context_idx = indexes[j]

                train_sg_pair(
                    U_ptr, V_ptr, size,
                    center_idx, context_idx,
                    alpha, work, next_random,
                    alias_prob, alias_index, alias_vocab_size, negative,
                    _compute_loss, running_loss
                )
                examples_trained[0] += 1

                # Track unique context words for V regularization
                if temporal_lambda > 0.0 and temporal_reg_V:
                    if seen_V[seen_offset + context_idx] == 0:
                        seen_V[seen_offset + context_idx] = 1
                        touched_V_words[n_touched_V[0]] = context_idx
                        touched_V_times[n_touched_V[0]] = <ITYPE_t>t
                        n_touched_V[0] += 1

            # Track unique center words for U regularization
            if temporal_lambda > 0.0:
                if seen_U[seen_offset + center_idx] == 0:
                    seen_U[seen_offset + center_idx] = 1
                    touched_U_words[n_touched_U[0]] = center_idx
                    touched_U_times[n_touched_U[0]] = <ITYPE_t>t
                    n_touched_U[0] += 1

    # --- Phase 2: Apply regularization once per unique (word, t) ---
    if temporal_lambda > 0.0:
        for k in range(n_touched_U[0]):
            apply_temporal_regularization(
                U_all, T, touched_U_times[k], vocab_size, size,
                touched_U_words[k], temporal_lambda, alpha
            )

        if temporal_reg_V:
            for k in range(n_touched_V[0]):
                apply_temporal_regularization(
                    V_all, T, touched_V_times[k], vocab_size, size,
                    touched_V_words[k], temporal_lambda, alpha
                )


def train_batch_dynamic(
    np.float32_t[:] U_flat,
    np.float32_t[:] V_flat,
    int T,
    int vocab_size,
    list sentences_with_time,
    dict word2idx,
    np.uint32_t[:] sample_ints,
    np.uint32_t[:] alias_prob,
    np.uint32_t[:] alias_index,
    np.float32_t[:] work,
    bint use_subsampling,
    int window,
    bint shrink_windows,
    float alpha,
    int negative,
    float temporal_lambda,
    bint temporal_reg_V,
    unsigned long long random_seed,
    bint compute_loss=True,
):
    """
    Train DynamicWord2Vec on a batch of sentences with temporal information.

    Temporal regularization is applied once per unique (word, time_slice) pair
    per batch, not per token occurrence, to avoid frequency-dependent
    regularization strength.

    Args:
        U_flat: Flattened U array [T * vocab_size * vector_size]
        V_flat: Flattened V array [T * vocab_size * vector_size]
        T: Number of time slices
        vocab_size: Vocabulary size
        sentences_with_time: List of (sentence, time_idx) tuples
        word2idx: Word to index mapping
        sample_ints: Subsampling thresholds
        alias_prob: Alias method probability thresholds
        alias_index: Alias method fallback indices
        work: Pre-allocated work buffer
        use_subsampling: Apply subsampling of frequent words
        window: Context window size
        shrink_windows: Randomly reduce window size per word
        alpha: Learning rate
        negative: Number of negative samples
        temporal_lambda: Regularization strength
        temporal_reg_V: Apply regularization to V embeddings
        random_seed: LCG seed
        compute_loss: Track loss during training

    Returns:
        (batch_loss, examples_trained, words_processed, final_random_state)
    """
    cdef int vector_size = len(U_flat) // (T * vocab_size)
    cdef int num_sentences = len(sentences_with_time)
    cdef int _compute_loss = 1 if compute_loss else 0
    cdef unsigned long long alias_vocab_size = len(alias_prob)

    # Get raw pointers
    cdef REAL_t *U_ptr = &U_flat[0]
    cdef REAL_t *V_ptr = &V_flat[0]
    cdef UITYPE_t *sample_ints_ptr = &sample_ints[0]
    cdef UITYPE_t *alias_prob_ptr = &alias_prob[0]
    cdef UITYPE_t *alias_index_ptr = &alias_index[0]
    cdef REAL_t *work_ptr = &work[0]

    # Stack-allocated buffers for batch data
    cdef UITYPE_t indexes[MAX_WORDS_IN_BATCH]
    cdef UITYPE_t sentence_idx[MAX_WORDS_IN_BATCH + 1]
    cdef ITYPE_t time_indices[MAX_WORDS_IN_BATCH]
    cdef ITYPE_t reduced_windows[MAX_WORDS_IN_BATCH]

    # State variables
    cdef unsigned long long next_random = random_seed
    cdef REAL_t running_loss = 0.0
    cdef long long total_examples = 0
    cdef long long total_words = 0

    # Batch processing state
    cdef int effective_words = 0
    cdef int effective_sentences = 0
    cdef int word_idx, time_idx
    cdef int sent_global_idx
    cdef int i
    cdef PyObject *result

    # Regularization dedup buffers (heap-allocated, vocab_size is runtime)
    cdef long long seen_size = <long long>T * vocab_size
    cdef unsigned char *seen_U = NULL
    cdef UITYPE_t *touched_U_words = NULL
    cdef ITYPE_t *touched_U_times = NULL
    cdef int n_touched_U = 0
    cdef unsigned char *seen_V = NULL
    cdef UITYPE_t *touched_V_words = NULL
    cdef ITYPE_t *touched_V_times = NULL
    cdef int n_touched_V = 0

    sentence_idx[0] = 0

    for sent_global_idx in range(num_sentences):
        sent_tuple = sentences_with_time[sent_global_idx]
        sent = sent_tuple[0]
        time_idx = sent_tuple[1]

        if sent is None or len(sent) == 0:
            continue

        # Store time index for this sentence
        time_indices[effective_sentences] = <ITYPE_t>time_idx

        for token in sent:
            result = PyDict_GetItemWithError(word2idx, token)
            if result == NULL:
                if PyErr_Occurred():
                    raise
                continue

            word_idx = <int>PyLong_AsLong(<object>result)
            total_words += 1

            if use_subsampling:
                if sample_ints_ptr[word_idx] < random_int32(&next_random):
                    continue

            indexes[effective_words] = <UITYPE_t>word_idx
            effective_words += 1

            if effective_words >= MAX_WORDS_IN_BATCH:
                break

        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words >= MAX_WORDS_IN_BATCH:
            break

    if effective_words > 0 and effective_sentences > 0:
        if shrink_windows:
            for i in range(effective_words):
                reduced_windows[i] = <ITYPE_t>(random_int32(&next_random) % window)
        else:
            memset(reduced_windows, 0, effective_words * cython.sizeof(ITYPE_t))

        # Allocate dedup buffers for regularization
        if temporal_lambda > 0.0:
            seen_U = <unsigned char *>malloc(seen_size)
            touched_U_words = <UITYPE_t *>malloc(MAX_WORDS_IN_BATCH * sizeof(UITYPE_t))
            touched_U_times = <ITYPE_t *>malloc(MAX_WORDS_IN_BATCH * sizeof(ITYPE_t))
            if seen_U == NULL or touched_U_words == NULL or touched_U_times == NULL:
                free(seen_U); free(touched_U_words); free(touched_U_times)
                raise MemoryError("Failed to allocate regularization buffers for U")
            memset(seen_U, 0, seen_size)

            if temporal_reg_V:
                seen_V = <unsigned char *>malloc(seen_size)
                touched_V_words = <UITYPE_t *>malloc(MAX_WORDS_IN_BATCH * sizeof(UITYPE_t))
                touched_V_times = <ITYPE_t *>malloc(MAX_WORDS_IN_BATCH * sizeof(ITYPE_t))
                if seen_V == NULL or touched_V_words == NULL or touched_V_times == NULL:
                    free(seen_U); free(touched_U_words); free(touched_U_times)
                    free(seen_V); free(touched_V_words); free(touched_V_times)
                    raise MemoryError("Failed to allocate regularization buffers for V")
                memset(seen_V, 0, seen_size)

        try:
            with nogil:
                train_batch_sg_dynamic(
                    U_ptr, V_ptr, T, vocab_size, vector_size,
                    indexes, sentence_idx, time_indices, reduced_windows,
                    effective_sentences, window, alpha,
                    work_ptr, &next_random,
                    alias_prob_ptr, alias_index_ptr, alias_vocab_size, negative,
                    temporal_lambda, temporal_reg_V,
                    _compute_loss, &running_loss, &total_examples,
                    seen_U, touched_U_words, touched_U_times, &n_touched_U,
                    seen_V, touched_V_words, touched_V_times, &n_touched_V,
                )
        finally:
            free(seen_U)
            free(touched_U_words)
            free(touched_U_times)
            free(seen_V)
            free(touched_V_words)
            free(touched_V_times)

    return running_loss, total_examples, total_words, next_random
