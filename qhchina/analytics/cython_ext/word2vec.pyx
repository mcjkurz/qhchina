"""
Fast Word2Vec training operations implemented in Cython, inspired heavily by gensim.models.word2vec
------------------------------------------------------
This module provides optimized implementations of the core training
operations for Word2Vec, including:

1. Training Skip-gram examples with negative sampling
2. Training CBOW examples with negative sampling
3. Efficient sigmoid and vector operations

These functions are designed to be called from the Python Word2Vec implementation
to accelerate the most computationally intensive parts of the training process.

Key optimizations:
- Alias method for O(1) negative sampling (vs. O(n) for linear search)
- Reusable buffers to eliminate memory allocations in the training loop
- BLAS operations for efficient vector math
- Precomputed tables for fast sigmoid calculations
- Xorshift128+ PRNG for fast random number generation
"""

# Compiler directives for maximum performance
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, fmax, fmin, sqrt
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time
from cython cimport floating
from builtins import print as py_print

# Import BLAS functions for optimized linear algebra
from scipy.linalg.cython_blas cimport sdot, ddot  # Dot product
from scipy.linalg.cython_blas cimport saxpy, daxpy  # Vector addition
from scipy.linalg.cython_blas cimport sscal, dscal  # Vector scaling

# Define C types for NumPy arrays
ctypedef fused real_t:
    np.float32_t
    np.float64_t
    
ctypedef np.int32_t ITYPE_t    # for word indices

# for grad clipping
DEF DEFAULT_MAX_GRAD = 1.0  # Can be overridden by gradient_clip parameter

# Constants for BLAS
cdef int ONE = 1
cdef float ONEF = 1.0
cdef double ONED = 1.0
cdef float ZEROF = 0.0  # Add constant for zeroing vectors
cdef double ZEROD = 0.0  # Add constant for zeroing vectors

# Global variables for shared resources
# These will be initialized once per training session
cdef np.float32_t[:] SIGMOID_TABLE_FLOAT32
cdef np.float32_t[:] LOG_SIGMOID_TABLE_FLOAT32
cdef np.float32_t[:] NOISE_DISTRIBUTION_FLOAT32

cdef np.float64_t[:] SIGMOID_TABLE_FLOAT64
cdef np.float64_t[:] LOG_SIGMOID_TABLE_FLOAT64
cdef np.float64_t[:] NOISE_DISTRIBUTION_FLOAT64

cdef float SIGMOID_SCALE
cdef int SIGMOID_OFFSET
cdef int NOISE_DISTRIBUTION_SIZE  # Store size separately to avoid .shape in nogil
cdef float NOISE_DISTRIBUTION_SUM
cdef float GRADIENT_CLIP
cdef float MAX_EXP = 6.0
cdef int NEGATIVE  # Number of negative samples
cdef float LEARNING_RATE
cdef bint CBOW_MEAN
cdef bint USING_DOUBLE_PRECISION  # Flag to indicate which set of tables to use
cdef int VECTOR_SIZE  # Dimensionality of the word vectors

# Alias method for efficient sampling
cdef np.int32_t[:] alias
cdef np.float32_t[:] prob

# Reusable buffers to reduce memory allocations
cdef np.ndarray _reusable_input_grad
cdef np.ndarray _reusable_output_grads
cdef np.ndarray _reusable_output_mask
cdef np.ndarray _reusable_center_grad
cdef np.ndarray _reusable_context_grads
cdef np.ndarray _reusable_neg_grads
cdef np.ndarray _reusable_context_mask
cdef np.ndarray _reusable_neg_mask
cdef np.ndarray _reusable_combined_input  # Add a new reusable buffer for combined input vector

# Define Xorshift128+ state structure
cdef struct xorshift128plus_state:
    unsigned long long s0
    unsigned long long s1

# Initialize global RNG state
cdef xorshift128plus_state RNG_STATE

# Initialize the RNG state with a seed
cdef void seed_xorshift128plus(unsigned long long seed) noexcept:
    cdef unsigned long long z = seed
    # Use splitmix64 algorithm to initialize state from seed
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL
    RNG_STATE.s0 = z ^ (z >> 31)
    
    z = (seed + 1)
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL
    RNG_STATE.s1 = z ^ (z >> 31)

# Fast Xorshift128+ random number generation (returns double in range [0,1))
cdef inline double xorshift128plus_random() noexcept:
    cdef unsigned long long s1 = RNG_STATE.s0
    cdef unsigned long long s0 = RNG_STATE.s1
    RNG_STATE.s0 = s0
    s1 ^= s1 << 23
    RNG_STATE.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)
    return (RNG_STATE.s1 + s0) / 18446744073709551616.0  # Divide by 2^64

# Initialize RNG with current time
seed_xorshift128plus(<unsigned long long>time(NULL))

# Python-accessible seeding function
def set_seed(unsigned long long seed):
    """
    Set the random number generator seed for reproducibility
    
    Args:
        seed: Unsigned 64-bit integer seed value
    """
    seed_xorshift128plus(seed)
    # Also seed the C standard library RNG as a fallback
    srand(<unsigned int>seed)

# Initialization function for global variables
def init_globals(
    sigmoid_table,  # Can be either float32 or float64
    log_sigmoid_table,  # Can be either float32 or float64
    float max_exp,  # Maximum exp value for sigmoid calculations
    noise_distribution,  # Can be either float32 or float64
    int vector_size,
    float gradient_clip=DEFAULT_MAX_GRAD,
    int negative=5,
    float learning_rate=0.025,
    bint cbow_mean=True,
    bint use_double_precision=False
):
    """
    Initialize global variables for shared resources.
    Call this once before training.
    
    Args:
        sigmoid_table: Precomputed sigmoid values (float32 or float64)
        log_sigmoid_table: Precomputed log sigmoid values (float32 or float64)
        max_exp: Maximum exp value for sigmoid lookup range [-max_exp, max_exp]
        noise_distribution: Distribution for negative sampling (float32 or float64)
        vector_size: Dimensionality of word vectors
        gradient_clip: Maximum absolute value for gradients
        negative: Number of negative samples
        learning_rate: Current learning rate
        cbow_mean: Whether to use mean or sum for context vectors
        use_double_precision: Whether to use double precision (float64) calculations
    """
    global SIGMOID_TABLE_FLOAT32, LOG_SIGMOID_TABLE_FLOAT32, NOISE_DISTRIBUTION_FLOAT32
    global SIGMOID_TABLE_FLOAT64, LOG_SIGMOID_TABLE_FLOAT64, NOISE_DISTRIBUTION_FLOAT64
    global SIGMOID_SCALE, SIGMOID_OFFSET, MAX_EXP
    global NOISE_DISTRIBUTION_SIZE, NOISE_DISTRIBUTION_SUM
    global GRADIENT_CLIP, NEGATIVE, LEARNING_RATE, CBOW_MEAN, USING_DOUBLE_PRECISION
    global VECTOR_SIZE
    global _reusable_input_grad, _reusable_output_grads, _reusable_output_mask
    global _reusable_center_grad, _reusable_context_grads, _reusable_neg_grads
    global _reusable_context_mask, _reusable_neg_mask
    
    # Set the precision based on the provided parameter instead of inferring from dtype
    USING_DOUBLE_PRECISION = use_double_precision
    
    # Copy data to the appropriate global variables based on precision
    if USING_DOUBLE_PRECISION:
        SIGMOID_TABLE_FLOAT64 = sigmoid_table
        LOG_SIGMOID_TABLE_FLOAT64 = log_sigmoid_table
        NOISE_DISTRIBUTION_FLOAT64 = noise_distribution
    else:
        SIGMOID_TABLE_FLOAT32 = sigmoid_table
        LOG_SIGMOID_TABLE_FLOAT32 = log_sigmoid_table
        NOISE_DISTRIBUTION_FLOAT32 = noise_distribution
    
    # Calculate sigmoid scale and offset based on max_exp and table size
    # Get the table size from the sigmoid table
    cdef int exp_table_size = len(sigmoid_table)
    MAX_EXP = max_exp
    SIGMOID_SCALE = exp_table_size / (2 * MAX_EXP)
    SIGMOID_OFFSET = exp_table_size / 2
    
    NOISE_DISTRIBUTION_SIZE = noise_distribution.shape[0]
    
    # Calculate the sum of noise distribution
    cdef float sum_probs = 0.0
    cdef int i
    
    # Calculate sum based on precision
    if USING_DOUBLE_PRECISION:
        for i in range(NOISE_DISTRIBUTION_SIZE):
            sum_probs += NOISE_DISTRIBUTION_FLOAT64[i]
    else:
        for i in range(NOISE_DISTRIBUTION_SIZE):
            sum_probs += NOISE_DISTRIBUTION_FLOAT32[i]
    
    NOISE_DISTRIBUTION_SUM = sum_probs
    
    GRADIENT_CLIP = gradient_clip
    NEGATIVE = negative
    LEARNING_RATE = learning_rate
    CBOW_MEAN = cbow_mean
    VECTOR_SIZE = vector_size
    dtype = np.float64 if USING_DOUBLE_PRECISION else np.float32

    # Initialize alias method for sampling
    init_alias(np.asarray(noise_distribution, dtype=dtype))
    
    # Initialize reusable buffers for single example training functions
    # Use the appropriate dtype for the reusable buffers
    _reusable_input_grad = np.zeros(VECTOR_SIZE, dtype=dtype)
    _reusable_output_grads = np.zeros((NOISE_DISTRIBUTION_SIZE, VECTOR_SIZE), dtype=dtype)
    _reusable_output_mask = np.zeros(NOISE_DISTRIBUTION_SIZE, dtype=np.int8)
    _reusable_center_grad = np.zeros(VECTOR_SIZE, dtype=dtype)
    _reusable_context_grads = np.zeros((NOISE_DISTRIBUTION_SIZE, VECTOR_SIZE), dtype=dtype)
    _reusable_neg_grads = np.zeros((NOISE_DISTRIBUTION_SIZE, VECTOR_SIZE), dtype=dtype)
    _reusable_context_mask = np.zeros(NOISE_DISTRIBUTION_SIZE, dtype=np.int8)
    _reusable_neg_mask = np.zeros(NOISE_DISTRIBUTION_SIZE, dtype=np.int8)
    _reusable_combined_input = np.zeros(VECTOR_SIZE, dtype=dtype)  # Initialize the combined input buffer

# Update learning rate from Python code
def update_learning_rate(float new_learning_rate):
    """Update the global learning rate value"""
    global LEARNING_RATE
    LEARNING_RATE = new_learning_rate

# Initialize alias method for efficient sampling
def init_alias(noise_distribution):
    """
    Initialize alias method for efficient sampling from discrete probability distribution.
    This implementation uses O(n) setup time and O(1) sampling time.
    
    Args:
        noise_distribution: Probability distribution for negative sampling (float32 or float64)
    """
    global alias, prob
    cdef int n = noise_distribution.shape[0]
    alias = np.zeros(n, dtype=np.int32)
    prob = np.zeros(n, dtype=np.float32)
    
    # Scaled probabilities for alias method - always use float32 for this
    cdef np.ndarray[np.float32_t, ndim=1] q = np.zeros(n, dtype=np.float32)
    cdef float sum_probs = 0.0
    cdef int i
    
    # Compute sum of probabilities
    for i in range(n):
        sum_probs += noise_distribution[i]
    
    # Scale probabilities to have mean = 1.0
    for i in range(n):
        q[i] = noise_distribution[i] * n / sum_probs
    
    # Create lists for small and large probabilities
    cdef list small = []
    cdef list large = []
    cdef int s, l
    
    # Initial partition between small and large probabilities
    for i in range(n):
        if q[i] < 1.0:
            small.append(i)
        else:
            large.append(i)
    
    # Generate probability and alias tables
    while small and large:
        s = small.pop()
        l = large.pop()
        
        prob[s] = q[s]  # Probability of drawing s directly
        alias[s] = l    # Alias for s when not drawn directly
        
        # Adjust probability of l
        q[l] = (q[l] + q[s]) - 1.0
        
        # Reclassify l based on new probability
        if q[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
    
    # Handle remaining elements (due to numerical precision)
    while large:
        l = large.pop()
        prob[l] = 1.0
    
    while small:
        s = small.pop()
        prob[s] = 1.0

# Efficient sampling using the alias method
cdef inline int alias_sample() noexcept:
    """
    Sample from the noise distribution in O(1) time using alias method.
    
    Returns:
        Sampled index from the noise distribution
    """
    # Select a bucket uniformly
    cdef int i = <int>(xorshift128plus_random() * NOISE_DISTRIBUTION_SIZE)
    
    # Flip weighted coin to decide whether to return bucket or its alias
    if xorshift128plus_random() < prob[i]:
        return i
    else:
        return alias[i]

# Wrapper functions for BLAS operations that automatically handle type selection
cdef inline void our_axpy(real_t *src, real_t *dst, real_t alpha, int size) noexcept:
    """
    Wrapper for BLAS axpy operation that automatically selects saxpy or daxpy based on real_t type.
    dst += alpha * src
    """
    cdef int inc = 1
    if real_t is np.float32_t:
        saxpy(&size, &alpha, src, &inc, dst, &inc)
    else:
        daxpy(&size, &alpha, src, &inc, dst, &inc)

cdef inline real_t our_dot(real_t *vec1, real_t *vec2, int size) noexcept:
    """
    Wrapper for BLAS dot operation that automatically selects sdot or ddot based on real_t type.
    Returns dot product of vec1 and vec2.
    """
    cdef int inc = 1
    if real_t is np.float32_t:
        return sdot(&size, vec1, &inc, vec2, &inc)
    else:
        return ddot(&size, vec1, &inc, vec2, &inc)

cdef inline void our_scal(real_t *vec, real_t alpha, int size) noexcept:
    """
    Wrapper for BLAS scal operation that automatically selects sscal or dscal based on real_t type.
    vec *= alpha
    """
    cdef int inc = 1
    if real_t is np.float32_t:
        sscal(&size, &alpha, vec, &inc)
    else:
        dscal(&size, &alpha, vec, &inc)

# Add a new function to efficiently zero a vector
cdef inline void our_zero(real_t *vec, int size) noexcept:
    """Zero out a vector using BLAS scal with zero factor"""
    if real_t is np.float32_t:
        sscal(&size, &ZEROF, vec, &ONE)
    else:
        dscal(&size, &ZEROD, vec, &ONE)

# Fast sigmoid implementation using table lookup
cdef inline real_t fast_sigmoid(real_t x) noexcept:
    """Fast sigmoid computation using precomputed lookup table"""
    # Handle extreme values with better approximations
    if x <= -MAX_EXP:
        return 0.0  # For very negative values, sigmoid is effectively 0
    elif x >= MAX_EXP:
        return 1.0  # For very positive values, sigmoid is effectively 1
    
    cdef int idx = <int>(x * SIGMOID_SCALE + SIGMOID_OFFSET)
    
    # Clamp index to valid range
    if idx < 0:
        idx = 0
    
    # Use the appropriate lookup table based on precision
    if real_t is np.float32_t:
        if idx >= SIGMOID_TABLE_FLOAT32.shape[0]:
            idx = SIGMOID_TABLE_FLOAT32.shape[0] - 1
        return SIGMOID_TABLE_FLOAT32[idx]
    else:  # float64
        if idx >= SIGMOID_TABLE_FLOAT64.shape[0]:
            idx = SIGMOID_TABLE_FLOAT64.shape[0] - 1
        return SIGMOID_TABLE_FLOAT64[idx]

# Fast log sigmoid using table lookup
cdef inline real_t fast_log_sigmoid(real_t x) noexcept:
    """Fast log sigmoid computation using precomputed lookup table"""
    # Handle extreme values with better approximations
    if x <= -MAX_EXP:
        return x  # For very negative values, log(sigmoid(x)) ≈ x
    elif x >= MAX_EXP:
        return 0.0  # For very positive values, log(sigmoid(x)) ≈ 0
    
    cdef int idx = <int>(x * SIGMOID_SCALE + SIGMOID_OFFSET)
    
    # Clamp index to valid range
    if idx < 0:
        idx = 0
    
    # Use the appropriate lookup table based on precision
    if real_t is np.float32_t:
        if idx >= LOG_SIGMOID_TABLE_FLOAT32.shape[0]:
            idx = LOG_SIGMOID_TABLE_FLOAT32.shape[0] - 1
        return LOG_SIGMOID_TABLE_FLOAT32[idx]
    else:  # float64
        if idx >= LOG_SIGMOID_TABLE_FLOAT64.shape[0]:
            idx = LOG_SIGMOID_TABLE_FLOAT64.shape[0] - 1
        return LOG_SIGMOID_TABLE_FLOAT64[idx]

# Function to sample from a discrete probability distribution
cdef inline int sample_from_distribution() noexcept:
    """
    Sample an index from the global noise distribution using the alias method.
    
    Returns:
        Sampled index
    """
    # Use the alias method for O(1) sampling
    return alias_sample()

# Clip gradient norm to prevent explosion while preserving direction
cdef inline void clip_gradient_norm(real_t* grad_vector, int size, real_t max_norm) noexcept:
    """
    Clip gradient by L2 norm to prevent explosion while preserving direction
    
    Args:
        grad_vector: Gradient vector to clip in-place
        size: Dimension of the vector
        max_norm: Maximum allowed L2 norm
    """
    # Declare all variables at the beginning
    cdef real_t norm_squared, norm, scale
    
    # Calculate L2 norm of gradient
    norm_squared = our_dot(grad_vector, grad_vector, size)
    
    # Skip if gradient is zero or very small
    if norm_squared <= 1e-12:
        return
    
    norm = sqrt(norm_squared)
    
    # Only clip if norm exceeds threshold
    if norm > max_norm:
        # Scale down the entire gradient vector to have norm = max_norm
        scale = max_norm / norm
        our_scal(grad_vector, scale, size)

# Single-example skipgram training function
cdef real_t train_skipgram_single(
    real_t[:, :] W,           # Input word embeddings
    real_t[:, :] W_prime,     # Output word embeddings
    ITYPE_t input_idx,        # Center word index
    ITYPE_t output_idx        # Context word index
) noexcept:
    """
    Train a single Skip-gram example with negative sampling.
    
    Args:
        W: Input word vectors (vocabulary_size x vector_size)
        W_prime: Output word vectors (vocabulary_size x vector_size)
        input_idx: Index of center word
        output_idx: Index of context word
        
    Returns:
        Loss for this example
    """
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int j, k, neg_idx
    cdef real_t score, prediction, gradient, neg_loss, loss = 0.0
    cdef real_t pos_loss = 0.0
    
    # Use reusable buffers instead of allocating new memory
    cdef real_t[:] input_grad = _reusable_input_grad
    cdef real_t[:, :] output_grads = _reusable_output_grads
    cdef np.int8_t[:] output_mask = _reusable_output_mask
    
    # Zero out the buffers with BLAS - only the portion we'll actually use
    our_zero(&input_grad[0], vector_size)
    
    for j in range(vocab_size):
        output_mask[j] = 0
        for k in range(vector_size):
            output_grads[j, k] = 0.0
    
    # === POSITIVE EXAMPLE ===
    # Compute dot product
    score = our_dot(&W[input_idx, 0], &W_prime[output_idx, 0], vector_size)
    
    # Apply sigmoid 
    prediction = fast_sigmoid(score)
    
    # Compute gradient for positive example (target = 1)
    gradient = prediction - 1.0
    
    # Accumulate gradients for positive example
    our_axpy(&W_prime[output_idx, 0], &input_grad[0], gradient, vector_size)
    
    # Accumulate gradients for output vector
    our_axpy(&W[input_idx, 0], &output_grads[output_idx, 0], gradient, vector_size)
    output_mask[output_idx] = 1  # Mark this index for update
    
    # Compute positive loss
    pos_loss = -fast_log_sigmoid(score)
    loss -= fast_log_sigmoid(score)
    
    # === NEGATIVE EXAMPLES ===
    for k in range(NEGATIVE):
        # Sample from noise distribution using alias method (O(1) time)
        neg_idx = sample_from_distribution()
        
        # Resample if we get the positive target
        while neg_idx == output_idx:
            neg_idx = sample_from_distribution()
            
        # Compute score
        score = our_dot(&W[input_idx, 0], &W_prime[neg_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for negative example (target = 0)
        gradient = prediction
        
        # Accumulate gradients for negative examples
        our_axpy(&W_prime[neg_idx, 0], &input_grad[0], gradient, vector_size)
        
        # Accumulate gradients for negative output vector
        our_axpy(&W[input_idx, 0], &output_grads[neg_idx, 0], gradient, vector_size)
        output_mask[neg_idx] = 1  # Mark this index for update
            
        # Calculate negative loss
        neg_loss = -fast_log_sigmoid(-score)
        loss -= fast_log_sigmoid(-score)
    
    # Apply all accumulated gradients at the end
    
    # Clip accumulated gradients by norm before applying
    clip_gradient_norm(&input_grad[0], vector_size, GRADIENT_CLIP)
    
    # Update input word vector
    our_axpy(&input_grad[0], &W[input_idx, 0], -LEARNING_RATE, vector_size)
    
    # Update output word vectors (both positive and negative)
    for j in range(vocab_size):
        if output_mask[j] == 1:  # Only update vectors that were used
            # Clip accumulated output gradients by norm before applying
            clip_gradient_norm(&output_grads[j, 0], vector_size, GRADIENT_CLIP)
            our_axpy(&output_grads[j, 0], &W_prime[j, 0], -LEARNING_RATE, vector_size)
    
    return loss

# Batch training for skipgram model
def train_skipgram_batch(
    real_t[:, :] W,           # Input word embeddings
    real_t[:, :] W_prime,     # Output word embeddings
    ITYPE_t[:] input_indices,  # Center word indices
    ITYPE_t[:] output_indices  # Context word indices
):
    """
    Train a batch of Skip-gram examples with negative sampling.
    
    Args:
        W: Input word vectors (vocabulary_size x vector_size)
        W_prime: Output word vectors (vocabulary_size x vector_size)
        input_indices: Indices of center words
        output_indices: Indices of context words
        
    Returns:
        Total loss for the batch
    """
    cdef int batch_size = input_indices.shape[0]
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int i, j, k, neg_idx
    cdef real_t total_loss = 0.0
    cdef real_t score, prediction, gradient
    cdef ITYPE_t in_idx, out_idx
    cdef real_t neg_lr = -LEARNING_RATE
    
    # Pre-allocate arrays for each example's processing
    cdef np.ndarray[real_t, ndim=1] input_grad = np.zeros(vector_size, dtype=W.base.dtype)
    
    # Generate all negative samples at once for efficiency
    cdef np.ndarray[ITYPE_t, ndim=2] neg_indices = np.zeros((batch_size, NEGATIVE), dtype=np.int32)
    generate_negative_samples(neg_indices, output_indices, batch_size, NEGATIVE)
    
    # Process each example (positive + its negative samples) one at a time
    for i in range(batch_size):
        # Get input and output indices for this example
        in_idx = input_indices[i]
        out_idx = output_indices[i]
        
        # Reset the input gradient buffer using BLAS
        our_zero(&input_grad[0], vector_size)
        
        # === POSITIVE EXAMPLE ===
        # Compute dot product
        score = our_dot(&W[in_idx, 0], &W_prime[out_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for positive example (target = 1)
        gradient = prediction - 1.0
        
        # Accumulate gradients for input vector
        our_axpy(&W_prime[out_idx, 0], &input_grad[0], gradient, vector_size)
        
        # Update output vector directly
        for j in range(vector_size):
            W_prime[out_idx, j] += neg_lr * gradient * W[in_idx, j]
        
        # Compute loss for positive example
        total_loss -= fast_log_sigmoid(score)
        
        # === NEGATIVE EXAMPLES for this positive example ===
        for k in range(NEGATIVE):
            neg_idx = neg_indices[i, k]
            
            # Skip duplicates (if a negative sample equals the positive)
            if neg_idx == out_idx:
                continue
            
            # Compute score
            score = our_dot(&W[in_idx, 0], &W_prime[neg_idx, 0], vector_size)
            
            # Apply sigmoid
            prediction = fast_sigmoid(score)
            
            # Compute gradient (target = 0)
            gradient = prediction
            
            # Accumulate gradients for input vector
            our_axpy(&W_prime[neg_idx, 0], &input_grad[0], gradient, vector_size)
            
            # Update negative output vector directly
            for j in range(vector_size):
                W_prime[neg_idx, j] += neg_lr * gradient * W[in_idx, j]
            
            # Compute loss for negative example
            total_loss -= fast_log_sigmoid(-score)
        
        # Update input word vector after processing all negatives
        # Clip accumulated input gradients by norm before applying
        clip_gradient_norm(&input_grad[0], vector_size, GRADIENT_CLIP)
        our_axpy(&input_grad[0], &W[in_idx, 0], neg_lr, vector_size)
    
    return total_loss

# Single-example CBOW training function
cdef real_t train_cbow_single(
    real_t[:, :] W,           # Input word embeddings
    real_t[:, :] W_prime,     # Output word embeddings
    ITYPE_t[:] context_indices,  # Context word indices
    ITYPE_t center_idx        # Center word index
) noexcept:
    """
    Train a single CBOW example with negative sampling.
    
    Args:
        W: Input word vectors (vocabulary_size x vector_size)
        W_prime: Output word vectors (vocabulary_size x vector_size)
        context_indices: Indices of context words
        center_idx: Index of center word
        
    Returns:
        Loss for this example
    """
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int j, k, c, neg_idx
    cdef int context_size = context_indices.shape[0]
    cdef ITYPE_t ctx_idx
    cdef real_t score, prediction, gradient, neg_loss, loss = 0.0
    cdef real_t pos_loss = 0.0  # Track positive sample loss
    cdef real_t scale_factor, input_gradient_scale
    
    # Return zero loss if no context words
    if context_size == 0:
        return 0.0
    
    # Use reusable buffers to avoid allocations
    cdef real_t[:] center_grad = _reusable_center_grad  # For positive center word
    cdef real_t[:, :] context_grads = _reusable_context_grads
    cdef real_t[:, :] neg_grads = _reusable_neg_grads
    cdef np.int8_t[:] context_mask = _reusable_context_mask
    cdef np.int8_t[:] neg_mask = _reusable_neg_mask
    cdef real_t[:] combined_input = _reusable_combined_input  # Use the global reusable buffer
    
    # Reset the buffers using BLAS
    our_zero(&center_grad[0], vector_size)
    our_zero(&combined_input[0], vector_size)
        
    for j in range(vocab_size):
        context_mask[j] = 0
        neg_mask[j] = 0
        for k in range(vector_size):
            context_grads[j, k] = 0.0
            neg_grads[j, k] = 0.0
    
    # Combine context vectors using BLAS for efficiency
    for c in range(context_size):
        ctx_idx = context_indices[c]
        # Add context vector to combined input using BLAS
        our_axpy(&W[ctx_idx, 0], &combined_input[0], 1.0, vector_size)
        # Mark context words for update
        if ctx_idx < context_mask.shape[0]:
            context_mask[ctx_idx] = 1
            
    # Apply mean if required - use BLAS for scaling
    if CBOW_MEAN and context_size > 1:
        scale_factor = 1.0 / context_size
        our_scal(&combined_input[0], scale_factor, vector_size)
    
    # === POSITIVE EXAMPLE ===
    # Compute dot product
    score = our_dot(&combined_input[0], &W_prime[center_idx, 0], vector_size)
    
    # Apply sigmoid
    prediction = fast_sigmoid(score)
    
    # Compute gradient for positive example (target = 1)
    gradient = prediction - 1.0
    
    # Accumulate gradient for center word
    our_axpy(&combined_input[0], &center_grad[0], gradient, vector_size)
    
    # Accumulate gradients for context words
    for c in range(context_size):
        ctx_idx = context_indices[c]
        
        # Compute gradient scale
        input_gradient_scale = gradient
        if CBOW_MEAN and context_size > 1:
            input_gradient_scale /= context_size
        
        # Accumulate gradient for this context word
        our_axpy(&W_prime[center_idx, 0], &context_grads[ctx_idx, 0], input_gradient_scale, vector_size)
    
    # Compute positive loss
    pos_loss = -fast_log_sigmoid(score)
    loss -= fast_log_sigmoid(score)
    
    # === NEGATIVE EXAMPLES ===
    for k in range(NEGATIVE):
        # Sample from noise distribution using the alias method (O(1) time)
        neg_idx = sample_from_distribution()
        
        # Resample if we get the positive target
        while neg_idx == center_idx:
            neg_idx = sample_from_distribution()
            
        # Compute score
        score = our_dot(&combined_input[0], &W_prime[neg_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for negative example (target = 0)
        gradient = prediction
        
        # Accumulate gradient for negative word
        our_axpy(&combined_input[0], &neg_grads[neg_idx, 0], gradient, vector_size)
        neg_mask[neg_idx] = 1  # Mark for update
        
        # Accumulate gradients for context words
        for c in range(context_size):
            ctx_idx = context_indices[c]
            
            # Compute gradient scale
            input_gradient_scale = gradient
            if CBOW_MEAN and context_size > 1:
                input_gradient_scale /= context_size
            
            # Accumulate gradient for this context word
            our_axpy(&W_prime[neg_idx, 0], &context_grads[ctx_idx, 0], input_gradient_scale, vector_size)
                
        # Calculate negative loss
        neg_loss = -fast_log_sigmoid(-score)
        loss -= fast_log_sigmoid(-score)
    
    # Apply all accumulated gradients at the end
    
    # Clip accumulated gradients by norm before applying
    clip_gradient_norm(&center_grad[0], vector_size, GRADIENT_CLIP)
    
    # Update center word vector
    our_axpy(&center_grad[0], &W_prime[center_idx, 0], -LEARNING_RATE, vector_size)
    
    # Update negative sample vectors
    for j in range(vocab_size):
        if neg_mask[j] == 1: # Only update vectors that were used
            clip_gradient_norm(&neg_grads[j, 0], vector_size, GRADIENT_CLIP)
            our_axpy(&neg_grads[j, 0], &W_prime[j, 0], -LEARNING_RATE, vector_size)
    
    # Update context word vectors
    for j in range(vocab_size):
        if context_mask[j] == 1: # Only update context words
            clip_gradient_norm(&context_grads[j, 0], vector_size, GRADIENT_CLIP)
            our_axpy(&context_grads[j, 0], &W[j, 0], -LEARNING_RATE, vector_size)
        
    return loss

# Batch training for CBOW model
def train_cbow_batch(
    real_t[:, :] W,              # Input word embeddings
    real_t[:, :] W_prime,        # Output word vectors (vocabulary_size x vector_size)
    list context_indices_list,    # List of lists of context word indices
    ITYPE_t[:] center_indices    # Center word indices
):
    """
    Train a batch of CBOW examples with negative sampling.
    
    Args:
        W: Input word vectors (vocabulary_size x vector_size)
        W_prime: Output word vectors (vocabulary_size x vector_size)
        context_indices_list: List of lists of context word indices
        center_indices: Indices of center words
        
    Returns:
        Total loss for the batch
    """
    cdef int batch_size = center_indices.shape[0]
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int i, j, k, c, neg_idx
    cdef ITYPE_t center_idx, ctx_idx
    cdef real_t total_loss = 0.0
    cdef real_t score, prediction, gradient, scale_factor, input_gradient_scale
    cdef real_t neg_lr = -LEARNING_RATE
    
    # Pre-allocate buffers for processing one example at a time
    cdef np.ndarray[real_t, ndim=1] combined_input = np.zeros(vector_size, dtype=W.base.dtype)
    cdef np.ndarray[real_t, ndim=1] context_grad = np.zeros(vector_size, dtype=W.base.dtype)
    
    # Generate all negative samples at once for efficiency
    cdef np.ndarray[ITYPE_t, ndim=2] neg_indices = np.zeros((batch_size, NEGATIVE), dtype=np.int32)
    generate_negative_samples(neg_indices, center_indices, batch_size, NEGATIVE)
    
    # Process each example (positive + its negative samples) one at a time
    for i in range(batch_size):
        # Get current example's context indices and center word
        context_indices = context_indices_list[i]
        center_idx = center_indices[i]
        
        # Skip examples with no context
        if not context_indices:
            continue
            
        # Reset the combined input vector using BLAS
        our_zero(&combined_input[0], vector_size)
        
        # Combine context vectors
        context_size = len(context_indices)
        for j in range(context_size):
            ctx_idx = context_indices[j]
            # Add context vector to combined input using BLAS
            our_axpy(&W[ctx_idx, 0], &combined_input[0], 1.0, vector_size)
        
        # Apply mean if required
        if CBOW_MEAN and context_size > 1:
            scale_factor = 1.0 / context_size
            our_scal(&combined_input[0], scale_factor, vector_size)
        
        # === POSITIVE EXAMPLE (center word) ===
        # Compute dot product
        score = our_dot(&combined_input[0], &W_prime[center_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for positive example (target = 1)
        gradient = prediction - 1.0
        
        # Update center word vector directly (for positive)
        for j in range(vector_size):
            W_prime[center_idx, j] += neg_lr * gradient * combined_input[j]
            
        # Calculate context word gradients for positive sample
        # Reset context_grad buffer using BLAS
        our_zero(&context_grad[0], vector_size)
            
        # Compute context gradient scaling factor
        input_gradient_scale = gradient
        if CBOW_MEAN and context_size > 1:
            input_gradient_scale /= context_size
            
        # Calculate context gradient for positive example
        our_axpy(&W_prime[center_idx, 0], &context_grad[0], input_gradient_scale, vector_size)
        
        # Compute loss for positive example
        total_loss -= fast_log_sigmoid(score)
        
        # === NEGATIVE EXAMPLES for this center word ===
        for k in range(NEGATIVE):
            neg_idx = neg_indices[i, k]
            
            # Skip if negative equals the positive (rare, but possible)
            if neg_idx == center_idx:
                continue
            
            # Compute score
            score = our_dot(&combined_input[0], &W_prime[neg_idx, 0], vector_size)
            
            # Apply sigmoid
            prediction = fast_sigmoid(score)
            
            # Compute gradient for negative example (target = 0)
            gradient = prediction
            
            # Update negative center word vector directly
            for j in range(vector_size):
                W_prime[neg_idx, j] += neg_lr * gradient * combined_input[j]
            
            # Calculate context word gradients for negative sample
            # Compute context gradient scaling factor
            input_gradient_scale = gradient
            if CBOW_MEAN and context_size > 1:
                input_gradient_scale /= context_size
                
            # Add to context_grad for each negative
            our_axpy(&W_prime[neg_idx, 0], &context_grad[0], input_gradient_scale, vector_size)
            
            # Compute loss for negative example
            total_loss -= fast_log_sigmoid(-score)
        
        # After processing all examples, update all context word vectors
        # Clip accumulated context gradients by norm before applying
        clip_gradient_norm(&context_grad[0], vector_size, GRADIENT_CLIP)
        for j in range(context_size):
            ctx_idx = context_indices[j]
            our_axpy(&context_grad[0], &W[ctx_idx, 0], neg_lr, vector_size)
    
    return total_loss

# Function to generate negative samples for an entire batch at once
cdef void generate_negative_samples(
    ITYPE_t[:, :] neg_indices,
    ITYPE_t[:] targets,
    int batch_size,
    int n_samples
) noexcept:
    """
    Generate negative samples for an entire batch at once using the alias method.
    Avoids positive targets and ensures unique indices when possible.
    
    Args:
        neg_indices: Pre-allocated array to store negative samples, shape (batch_size, n_samples)
        targets: Target indices to avoid in negative sampling
        batch_size: Number of examples in the batch
        n_samples: Number of negative samples per example
    """
    cdef int i, j
    cdef ITYPE_t neg_idx, target_idx
    
    for i in range(batch_size):
        target_idx = targets[i]
        
        for j in range(n_samples):
            # Sample from noise distribution using alias method (O(1) time)
            neg_idx = alias_sample()
            
            # Resample if we get the positive target
            while neg_idx == target_idx:
                neg_idx = alias_sample()
            
            # Store the negative sample
            neg_indices[i, j] = neg_idx 