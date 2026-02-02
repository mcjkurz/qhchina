"""
Fast Word2Vec batch training operations implemented in Cython.

This module provides optimized implementations of the core batch training
operations for Word2Vec, including:

1. Batch training Skip-gram examples with negative sampling
2. Batch training CBOW examples with negative sampling
3. Efficient sigmoid and vector operations

These functions are designed to be called from the Python Word2Vec implementation
to accelerate the most computationally intensive parts of the training process.

Key optimizations:
- Alias method for O(1) negative sampling (vs. O(n) for linear search)
- BLAS operations for efficient vector math (via scipy.linalg.cython_blas)
- Precomputed tables for fast sigmoid calculations
- Xorshift128+ PRNG for fast random number generation
- Per-batch array allocation using NumPy's optimized memory pool
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
    # Declare all variables at the top of the function
    cdef int batch_size = input_indices.shape[0]
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int i, j, k, neg_idx
    cdef real_t total_loss = 0.0
    cdef real_t score, prediction, gradient
    cdef ITYPE_t in_idx, out_idx
    cdef real_t neg_lr = -LEARNING_RATE
    # Variables for optimized gradient clipping
    cdef real_t input_norm, grad_norm, clip_scale, effective_lr
    
    # Allocate fresh arrays each batch (NumPy allocation is fast and well-optimized)
    cdef np.ndarray[real_t, ndim=1] input_grad = np.zeros(vector_size, dtype=W.base.dtype)
    
    # Generate all negative samples at once for efficiency
    cdef np.ndarray[ITYPE_t, ndim=2] neg_indices = np.zeros((batch_size, NEGATIVE), dtype=np.int32)
    generate_negative_samples(neg_indices, output_indices, batch_size, NEGATIVE)
    
    # Process each example (positive + its negative samples) one at a time
    for i in range(batch_size):
        # Get input and output indices for this example
        in_idx = input_indices[i]
        out_idx = output_indices[i]
        
        # Reset gradient buffer using BLAS
        our_zero(&input_grad[0], vector_size)
        
        # Compute input vector norm once per example for efficient gradient clipping
        # All negative samples use the same input vector, so we can reuse this
        input_norm = sqrt(our_dot(&W[in_idx, 0], &W[in_idx, 0], vector_size))
        
        # === POSITIVE EXAMPLE ===
        # Compute dot product
        score = our_dot(&W[in_idx, 0], &W_prime[out_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for positive example (target = 1)
        gradient = prediction - 1.0
        
        # Accumulate gradients for input vector
        our_axpy(&W_prime[out_idx, 0], &input_grad[0], gradient, vector_size)
        
        # Apply clipped gradient to positive output vector (unified scalar method)
        # For positive: gradient = prediction - 1.0 ∈ [-1, 0], so |gradient| = -gradient
        # grad_norm = |gradient| * input_norm = -gradient * input_norm
        grad_norm = -gradient * input_norm
        if grad_norm > GRADIENT_CLIP and grad_norm > 1e-12:
            clip_scale = GRADIENT_CLIP / grad_norm
            effective_lr = neg_lr * gradient * clip_scale
        else:
            effective_lr = neg_lr * gradient
        our_axpy(&W[in_idx, 0], &W_prime[out_idx, 0], effective_lr, vector_size)
        
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
            
            # Optimized gradient clipping: compute clip scale directly from input_norm
            # The gradient vector is gradient * W[in_idx], with norm = gradient * input_norm
            # Instead of: zero buffer, copy, clip, apply (4 BLAS calls)
            # We compute the effective learning rate and apply directly (1 BLAS call)
            grad_norm = gradient * input_norm
            if grad_norm > GRADIENT_CLIP and grad_norm > 1e-12:
                # Scale down: effective_gradient = gradient * (GRADIENT_CLIP / grad_norm)
                clip_scale = GRADIENT_CLIP / grad_norm
                effective_lr = neg_lr * gradient * clip_scale
            else:
                effective_lr = neg_lr * gradient
            
            # Apply gradient directly to negative output vector (single BLAS call)
            our_axpy(&W[in_idx, 0], &W_prime[neg_idx, 0], effective_lr, vector_size)
            
            # Compute loss for negative example
            total_loss -= fast_log_sigmoid(-score)
        
        # Clip and apply accumulated input gradient
        clip_gradient_norm(&input_grad[0], vector_size, GRADIENT_CLIP)
        our_axpy(&input_grad[0], &W[in_idx, 0], neg_lr, vector_size)
    
    return total_loss

# Batch training for CBOW model using flat arrays (optimized, no Python list overhead)
def train_cbow_batch(
    real_t[:, :] W,              # Input word embeddings
    real_t[:, :] W_prime,        # Output word vectors (vocabulary_size x vector_size)
    ITYPE_t[:] context_flat,     # Flat array of all context indices
    ITYPE_t[:] context_offsets,  # Start offset for each example's context
    ITYPE_t[:] context_lengths,  # Length of context for each example
    ITYPE_t[:] center_indices    # Center word indices
):
    """
    Train a batch of CBOW examples with negative sampling using flat arrays.
    
    This is the optimized interface that avoids Python list construction overhead.
    Context indices are stored in a flat array with offsets and lengths for each example.
    
    Args:
        W: Input word vectors (vocabulary_size x vector_size)
        W_prime: Output word vectors (vocabulary_size x vector_size)
        context_flat: Flat array containing all context indices concatenated
        context_offsets: Start offset in context_flat for each example
        context_lengths: Number of context words for each example
        center_indices: Indices of center words
        
    Returns:
        Total loss for the batch
    """
    cdef int batch_size = center_indices.shape[0]
    cdef int vector_size = W.shape[1]
    cdef int vocab_size = W.shape[0]
    cdef int i, j, k, neg_idx, offset, context_size
    cdef ITYPE_t center_idx, ctx_idx
    cdef real_t total_loss = 0.0
    cdef real_t score, prediction, gradient, scale_factor, input_gradient_scale
    cdef real_t neg_lr = -LEARNING_RATE
    # Variables for optimized gradient clipping
    cdef real_t combined_input_norm, grad_norm, clip_scale, effective_lr
    
    # Allocate fresh arrays each batch (NumPy allocation is fast and well-optimized)
    cdef np.ndarray[real_t, ndim=1] combined_input = np.zeros(vector_size, dtype=W.base.dtype)
    cdef np.ndarray[real_t, ndim=1] context_grad = np.zeros(vector_size, dtype=W.base.dtype)
    
    # Generate all negative samples at once for efficiency
    cdef np.ndarray[ITYPE_t, ndim=2] neg_indices = np.zeros((batch_size, NEGATIVE), dtype=np.int32)
    generate_negative_samples(neg_indices, center_indices, batch_size, NEGATIVE)
    
    # Process each example (positive + its negative samples) one at a time
    for i in range(batch_size):
        # Get current example's context info and center word
        center_idx = center_indices[i]
        offset = context_offsets[i]
        context_size = context_lengths[i]
        
        # Skip examples with no context
        if context_size == 0:
            continue
            
        # Reset the combined input vector using BLAS
        our_zero(&combined_input[0], vector_size)
        
        # Combine context vectors from flat array
        for j in range(context_size):
            ctx_idx = context_flat[offset + j]
            # Add context vector to combined input using BLAS
            our_axpy(&W[ctx_idx, 0], &combined_input[0], 1.0, vector_size)
        
        # Apply mean if required
        if CBOW_MEAN and context_size > 1:
            scale_factor = 1.0 / context_size
            our_scal(&combined_input[0], scale_factor, vector_size)
        
        # Compute combined_input norm once per example for efficient gradient clipping
        # All negative samples use the same combined_input, so we can reuse this
        combined_input_norm = sqrt(our_dot(&combined_input[0], &combined_input[0], vector_size))
        
        # === POSITIVE EXAMPLE (center word) ===
        # Reset context gradient accumulator using BLAS
        our_zero(&context_grad[0], vector_size)
        
        # Compute dot product
        score = our_dot(&combined_input[0], &W_prime[center_idx, 0], vector_size)
        
        # Apply sigmoid
        prediction = fast_sigmoid(score)
        
        # Compute gradient for positive example (target = 1)
        gradient = prediction - 1.0
        
        # Apply clipped gradient to positive center word (unified scalar method)
        # For positive: gradient = prediction - 1.0 ∈ [-1, 0], so |gradient| = -gradient
        # grad_norm = |gradient| * combined_input_norm = -gradient * combined_input_norm
        grad_norm = -gradient * combined_input_norm
        if grad_norm > GRADIENT_CLIP and grad_norm > 1e-12:
            clip_scale = GRADIENT_CLIP / grad_norm
            effective_lr = neg_lr * gradient * clip_scale
        else:
            effective_lr = neg_lr * gradient
        our_axpy(&combined_input[0], &W_prime[center_idx, 0], effective_lr, vector_size)
            
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
            
            # Optimized gradient clipping: compute clip scale directly from combined_input_norm
            # The gradient vector is gradient * combined_input, with norm = gradient * combined_input_norm
            # Instead of: zero buffer, copy, clip, apply (4 BLAS calls)
            # We compute the effective learning rate and apply directly (1 BLAS call)
            grad_norm = gradient * combined_input_norm
            if grad_norm > GRADIENT_CLIP and grad_norm > 1e-12:
                # Scale down: effective_gradient = gradient * (GRADIENT_CLIP / grad_norm)
                clip_scale = GRADIENT_CLIP / grad_norm
                effective_lr = neg_lr * gradient * clip_scale
            else:
                effective_lr = neg_lr * gradient
            
            # Apply gradient directly to negative word vector (single BLAS call)
            our_axpy(&combined_input[0], &W_prime[neg_idx, 0], effective_lr, vector_size)
            
            # Compute context gradient scaling factor
            input_gradient_scale = gradient
            if CBOW_MEAN and context_size > 1:
                input_gradient_scale /= context_size
                
            # Add to context_grad for each negative
            our_axpy(&W_prime[neg_idx, 0], &context_grad[0], input_gradient_scale, vector_size)
            
            # Compute loss for negative example
            total_loss -= fast_log_sigmoid(-score)
        
        # Clip accumulated context gradients by norm before applying
        clip_gradient_norm(&context_grad[0], vector_size, GRADIENT_CLIP)
        for j in range(context_size):
            ctx_idx = context_flat[offset + j]
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


# =============================================================================
# Fast Example Generation Functions
# =============================================================================

def generate_skipgram_examples(
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling
):
    """
    Generate Skip-gram training examples from pre-indexed sentences.
    
    This function performs the heavy lifting of example generation in Cython,
    including subsampling and dynamic window sizing.
    
    Args:
        indexed_sentences: List of numpy int32 arrays, each containing word indices
                          for a sentence (-1 for OOV words)
        discard_probs: Array of discard probabilities indexed by word ID
        window: Maximum context window size
        shrink_windows: If True, randomly shrink window for each center word
        use_subsampling: If True, apply subsampling based on discard_probs
    
    Returns:
        Tuple of (input_indices, output_indices) as numpy int32 arrays
    """
    cdef int num_sentences = len(indexed_sentences)
    cdef int sent_idx, pos, context_pos, sentence_len
    cdef int center_idx, context_idx, dynamic_window, start, end
    cdef np.ndarray[ITYPE_t, ndim=1] sentence
    cdef double rand_val
    
    # First pass: estimate number of examples for pre-allocation
    # Use a generous estimate: sum of (sentence_len * 2 * window)
    cdef long estimated_examples = 0
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        estimated_examples += len(sentence) * 2 * window
    
    # Pre-allocate output arrays (we'll trim at the end)
    cdef np.ndarray[ITYPE_t, ndim=1] input_indices = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] output_indices = np.empty(estimated_examples, dtype=np.int32)
    cdef long example_count = 0
    
    # Process each sentence
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        
        if sentence_len == 0:
            continue
        
        # Process each position in the sentence
        for pos in range(sentence_len):
            center_idx = sentence[pos]
            
            # Skip OOV words (marked as -1)
            if center_idx < 0:
                continue
            
            # Apply subsampling to center word
            if use_subsampling:
                rand_val = xorshift128plus_random()
                if rand_val < discard_probs[center_idx]:
                    continue
            
            # Determine window size
            if shrink_windows:
                dynamic_window = 1 + <int>(xorshift128plus_random() * window)
            else:
                dynamic_window = window
            
            # Calculate context boundaries
            start = pos - dynamic_window
            if start < 0:
                start = 0
            end = pos + dynamic_window + 1
            if end > sentence_len:
                end = sentence_len
            
            # Generate examples for each context position
            for context_pos in range(start, end):
                if context_pos == pos:
                    continue
                
                context_idx = sentence[context_pos]
                
                # Skip OOV context words
                if context_idx < 0:
                    continue
                
                # Apply subsampling to context word
                if use_subsampling:
                    rand_val = xorshift128plus_random()
                    if rand_val < discard_probs[context_idx]:
                        continue
                
                # Store the example
                input_indices[example_count] = center_idx
                output_indices[example_count] = context_idx
                example_count += 1
    
    # Trim arrays to actual size
    return input_indices[:example_count].copy(), output_indices[:example_count].copy()


def generate_cbow_examples(
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling
):
    """
    Generate CBOW training examples from pre-indexed sentences.
    
    For CBOW, each example has variable-length context. We return:
    - context_flat: Flat array of all context indices
    - context_offsets: Start offset for each example's context in context_flat
    - context_lengths: Length of context for each example
    - center_indices: Center word index for each example
    
    Args:
        indexed_sentences: List of numpy int32 arrays, each containing word indices
                          for a sentence (-1 for OOV words)
        discard_probs: Array of discard probabilities indexed by word ID
        window: Maximum context window size
        shrink_windows: If True, randomly shrink window for each center word
        use_subsampling: If True, apply subsampling based on discard_probs
    
    Returns:
        Tuple of (context_flat, context_offsets, context_lengths, center_indices)
        all as numpy int32 arrays
    """
    cdef int num_sentences = len(indexed_sentences)
    cdef int sent_idx, pos, context_pos, sentence_len, i
    cdef int center_idx, context_idx, dynamic_window, start, end
    cdef np.ndarray[ITYPE_t, ndim=1] sentence
    cdef double rand_val
    cdef int ctx_count
    
    # First pass: estimate sizes for pre-allocation
    cdef long estimated_examples = 0
    cdef long estimated_context = 0
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        estimated_examples += sentence_len
        estimated_context += sentence_len * 2 * window
    
    # Pre-allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] context_flat = np.empty(estimated_context, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] context_offsets = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] context_lengths = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] center_indices = np.empty(estimated_examples, dtype=np.int32)
    
    cdef long example_count = 0
    cdef long context_total = 0
    
    # Temporary buffer for context indices (max possible size = 2 * window)
    cdef np.ndarray[ITYPE_t, ndim=1] context_buffer = np.empty(2 * window, dtype=np.int32)
    
    # Process each sentence
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        
        if sentence_len == 0:
            continue
        
        # Process each position in the sentence
        for pos in range(sentence_len):
            center_idx = sentence[pos]
            
            # Skip OOV words
            if center_idx < 0:
                continue
            
            # Apply subsampling to center word
            if use_subsampling:
                rand_val = xorshift128plus_random()
                if rand_val < discard_probs[center_idx]:
                    continue
            
            # Determine window size
            if shrink_windows:
                dynamic_window = 1 + <int>(xorshift128plus_random() * window)
            else:
                dynamic_window = window
            
            # Calculate context boundaries
            start = pos - dynamic_window
            if start < 0:
                start = 0
            end = pos + dynamic_window + 1
            if end > sentence_len:
                end = sentence_len
            
            # Collect context indices
            ctx_count = 0
            for context_pos in range(start, end):
                if context_pos == pos:
                    continue
                
                context_idx = sentence[context_pos]
                
                # Skip OOV context words
                if context_idx < 0:
                    continue
                
                # Apply subsampling to context word
                if use_subsampling:
                    rand_val = xorshift128plus_random()
                    if rand_val < discard_probs[context_idx]:
                        continue
                
                context_buffer[ctx_count] = context_idx
                ctx_count += 1
            
            # Only create example if we have at least one context word
            if ctx_count > 0:
                # Store example metadata
                context_offsets[example_count] = context_total
                context_lengths[example_count] = ctx_count
                center_indices[example_count] = center_idx
                
                # Copy context indices to flat array
                for i in range(ctx_count):
                    context_flat[context_total + i] = context_buffer[i]
                
                context_total += ctx_count
                example_count += 1
    
    # Trim arrays to actual size and return
    return (
        context_flat[:context_total].copy(),
        context_offsets[:example_count].copy(),
        context_lengths[:example_count].copy(),
        center_indices[:example_count].copy()
    )


def train_skipgram_from_indexed(
    real_t[:, :] W,
    real_t[:, :] W_prime,
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    int batch_size
):
    """
    Generate Skip-gram examples and train in batches, all in Cython.
    
    This combines example generation and training to minimize Python overhead.
    
    Args:
        W: Input word vectors
        W_prime: Output word vectors  
        indexed_sentences: List of numpy int32 arrays with word indices
        discard_probs: Subsampling discard probabilities
        window: Maximum context window size
        shrink_windows: Whether to use dynamic window sizing
        use_subsampling: Whether to apply subsampling
        batch_size: Number of examples per training batch
    
    Returns:
        Tuple of (total_loss, total_examples)
    """
    # Generate all examples first
    input_indices, output_indices = generate_skipgram_examples(
        indexed_sentences, discard_probs, window, shrink_windows, use_subsampling
    )
    
    cdef long total_examples = len(input_indices)
    cdef real_t total_loss = 0.0
    cdef long start_idx, end_idx
    cdef int current_batch_size
    
    # Train in batches
    start_idx = 0
    while start_idx < total_examples:
        end_idx = start_idx + batch_size
        if end_idx > total_examples:
            end_idx = total_examples
        
        current_batch_size = end_idx - start_idx
        
        # Extract batch
        batch_input = input_indices[start_idx:end_idx]
        batch_output = output_indices[start_idx:end_idx]
        
        # Train batch
        total_loss += train_skipgram_batch(W, W_prime, batch_input, batch_output)
        
        start_idx = end_idx
    
    return total_loss, total_examples


def train_cbow_from_indexed(
    real_t[:, :] W,
    real_t[:, :] W_prime,
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    int batch_size
):
    """
    Generate CBOW examples and train in batches, all in Cython.
    
    Uses the optimized flat array interface to avoid Python list construction overhead.
    
    Args:
        W: Input word vectors
        W_prime: Output word vectors  
        indexed_sentences: List of numpy int32 arrays with word indices
        discard_probs: Subsampling discard probabilities
        window: Maximum context window size
        shrink_windows: Whether to use dynamic window sizing
        use_subsampling: Whether to apply subsampling
        batch_size: Number of examples per training batch
    
    Returns:
        Tuple of (total_loss, total_examples)
    """
    # Generate all examples using flat array format
    context_flat, context_offsets, context_lengths, center_indices = generate_cbow_examples(
        indexed_sentences, discard_probs, window, shrink_windows, use_subsampling
    )
    
    cdef long total_examples = len(center_indices)
    cdef real_t total_loss = 0.0
    cdef long start_idx, end_idx
    cdef int current_batch_size
    
    # Train in batches using the optimized flat array interface
    start_idx = 0
    while start_idx < total_examples:
        end_idx = start_idx + batch_size
        if end_idx > total_examples:
            end_idx = total_examples
        
        current_batch_size = end_idx - start_idx
        
        # Extract batch slices (no Python list construction needed)
        batch_offsets = context_offsets[start_idx:end_idx]
        batch_lengths = context_lengths[start_idx:end_idx]
        batch_center = center_indices[start_idx:end_idx]
        
        # Train batch
        total_loss += train_cbow_batch(
            W, W_prime,
            context_flat,  # Pass entire flat array (offsets handle indexing)
            batch_offsets,
            batch_lengths,
            batch_center
        )
        
        start_idx = end_idx
    
    return total_loss, total_examples


# =============================================================================
# Temporal Referencing Functions for TempRefWord2Vec
# =============================================================================

def generate_skipgram_examples_temporal(
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    ITYPE_t[:] temporal_index_map
):
    """
    Generate Skip-gram training examples with temporal index mapping.
    
    For TempRefWord2Vec, context words (outputs) should use their base form.
    The temporal_index_map converts temporal variant indices to base word indices.
    Input (center) words keep their temporal variant form.
    
    Args:
        indexed_sentences: List of numpy int32 arrays, each containing word indices
        discard_probs: Array of discard probabilities indexed by word ID
        window: Maximum context window size
        shrink_windows: If True, randomly shrink window for each center word
        use_subsampling: If True, apply subsampling based on discard_probs
        temporal_index_map: Array mapping word indices to their base form indices.
                           For non-temporal words, map[i] = i.
                           For temporal variants, map[i] = base_word_index.
    
    Returns:
        Tuple of (input_indices, output_indices) as numpy int32 arrays
    """
    cdef int num_sentences = len(indexed_sentences)
    cdef int sent_idx, pos, context_pos, sentence_len
    cdef int center_idx, context_idx, mapped_context_idx, dynamic_window, start, end
    cdef np.ndarray[ITYPE_t, ndim=1] sentence
    cdef double rand_val
    
    # First pass: estimate number of examples for pre-allocation
    cdef long estimated_examples = 0
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        estimated_examples += len(sentence) * 2 * window
    
    # Pre-allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] input_indices = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] output_indices = np.empty(estimated_examples, dtype=np.int32)
    cdef long example_count = 0
    
    # Process each sentence
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        
        if sentence_len == 0:
            continue
        
        # Process each position in the sentence
        for pos in range(sentence_len):
            center_idx = sentence[pos]
            
            # Skip OOV words (marked as -1)
            if center_idx < 0:
                continue
            
            # Apply subsampling to center word
            if use_subsampling:
                rand_val = xorshift128plus_random()
                if rand_val < discard_probs[center_idx]:
                    continue
            
            # Determine window size
            if shrink_windows:
                dynamic_window = 1 + <int>(xorshift128plus_random() * window)
            else:
                dynamic_window = window
            
            # Calculate context boundaries
            start = pos - dynamic_window
            if start < 0:
                start = 0
            end = pos + dynamic_window + 1
            if end > sentence_len:
                end = sentence_len
            
            # Generate examples for each context position
            for context_pos in range(start, end):
                if context_pos == pos:
                    continue
                
                context_idx = sentence[context_pos]
                
                # Skip OOV context words
                if context_idx < 0:
                    continue
                
                # Apply subsampling to context word
                if use_subsampling:
                    rand_val = xorshift128plus_random()
                    if rand_val < discard_probs[context_idx]:
                        continue
                
                # Apply temporal mapping to context word (convert to base form)
                mapped_context_idx = temporal_index_map[context_idx]
                
                # Skip if mapping results in invalid index
                if mapped_context_idx < 0:
                    continue
                
                # Store the example (center word unchanged, context mapped to base)
                input_indices[example_count] = center_idx
                output_indices[example_count] = mapped_context_idx
                example_count += 1
    
    # Trim arrays to actual size
    return input_indices[:example_count].copy(), output_indices[:example_count].copy()


def generate_cbow_examples_temporal(
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    ITYPE_t[:] temporal_index_map
):
    """
    Generate CBOW training examples with temporal index mapping.
    
    For TempRefWord2Vec CBOW, context words (inputs) should use their base form.
    The temporal_index_map converts temporal variant indices to base word indices.
    Center (output) words keep their temporal variant form.
    
    Args:
        indexed_sentences: List of numpy int32 arrays, each containing word indices
        discard_probs: Array of discard probabilities indexed by word ID
        window: Maximum context window size
        shrink_windows: If True, randomly shrink window for each center word
        use_subsampling: If True, apply subsampling based on discard_probs
        temporal_index_map: Array mapping word indices to their base form indices.
    
    Returns:
        Tuple of (context_flat, context_offsets, context_lengths, center_indices)
    """
    cdef int num_sentences = len(indexed_sentences)
    cdef int sent_idx, pos, context_pos, sentence_len, i
    cdef int center_idx, context_idx, mapped_context_idx, dynamic_window, start, end
    cdef np.ndarray[ITYPE_t, ndim=1] sentence
    cdef double rand_val
    cdef int ctx_count
    
    # First pass: estimate sizes for pre-allocation
    cdef long estimated_examples = 0
    cdef long estimated_context = 0
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        estimated_examples += sentence_len
        estimated_context += sentence_len * 2 * window
    
    # Pre-allocate output arrays
    cdef np.ndarray[ITYPE_t, ndim=1] context_flat = np.empty(estimated_context, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] context_offsets = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] context_lengths = np.empty(estimated_examples, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] center_indices = np.empty(estimated_examples, dtype=np.int32)
    
    cdef long example_count = 0
    cdef long context_total = 0
    
    # Temporary buffer for context indices (max possible size = 2 * window)
    cdef np.ndarray[ITYPE_t, ndim=1] context_buffer = np.empty(2 * window, dtype=np.int32)
    
    # Process each sentence
    for sent_idx in range(num_sentences):
        sentence = indexed_sentences[sent_idx]
        sentence_len = len(sentence)
        
        if sentence_len == 0:
            continue
        
        # Process each position in the sentence
        for pos in range(sentence_len):
            center_idx = sentence[pos]
            
            # Skip OOV words
            if center_idx < 0:
                continue
            
            # Apply subsampling to center word
            if use_subsampling:
                rand_val = xorshift128plus_random()
                if rand_val < discard_probs[center_idx]:
                    continue
            
            # Determine window size
            if shrink_windows:
                dynamic_window = 1 + <int>(xorshift128plus_random() * window)
            else:
                dynamic_window = window
            
            # Calculate context boundaries
            start = pos - dynamic_window
            if start < 0:
                start = 0
            end = pos + dynamic_window + 1
            if end > sentence_len:
                end = sentence_len
            
            # Collect context indices with temporal mapping
            ctx_count = 0
            for context_pos in range(start, end):
                if context_pos == pos:
                    continue
                
                context_idx = sentence[context_pos]
                
                # Skip OOV context words
                if context_idx < 0:
                    continue
                
                # Apply subsampling to context word
                if use_subsampling:
                    rand_val = xorshift128plus_random()
                    if rand_val < discard_probs[context_idx]:
                        continue
                
                # Apply temporal mapping to context word (convert to base form)
                mapped_context_idx = temporal_index_map[context_idx]
                
                # Skip if mapping results in invalid index
                if mapped_context_idx < 0:
                    continue
                
                context_buffer[ctx_count] = mapped_context_idx
                ctx_count += 1
            
            # Only create example if we have at least one context word
            if ctx_count > 0:
                # Store example metadata (center word unchanged)
                context_offsets[example_count] = context_total
                context_lengths[example_count] = ctx_count
                center_indices[example_count] = center_idx
                
                # Copy mapped context indices to flat array
                for i in range(ctx_count):
                    context_flat[context_total + i] = context_buffer[i]
                
                context_total += ctx_count
                example_count += 1
    
    # Trim arrays to actual size and return
    return (
        context_flat[:context_total].copy(),
        context_offsets[:example_count].copy(),
        context_lengths[:example_count].copy(),
        center_indices[:example_count].copy()
    )


def train_skipgram_from_indexed_temporal(
    real_t[:, :] W,
    real_t[:, :] W_prime,
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    int batch_size,
    ITYPE_t[:] temporal_index_map
):
    """
    Generate Skip-gram examples with temporal mapping and train in batches.
    
    For TempRefWord2Vec: context words (outputs) are mapped to base forms,
    while center words (inputs) retain their temporal variant form.
    
    Args:
        W: Input word vectors
        W_prime: Output word vectors  
        indexed_sentences: List of numpy int32 arrays with word indices
        discard_probs: Subsampling discard probabilities
        window: Maximum context window size
        shrink_windows: Whether to use dynamic window sizing
        use_subsampling: Whether to apply subsampling
        batch_size: Number of examples per training batch
        temporal_index_map: Array mapping temporal variants to base word indices
    
    Returns:
        Tuple of (total_loss, total_examples)
    """
    # Generate all examples with temporal mapping
    input_indices, output_indices = generate_skipgram_examples_temporal(
        indexed_sentences, discard_probs, window, shrink_windows, 
        use_subsampling, temporal_index_map
    )
    
    cdef long total_examples = len(input_indices)
    cdef real_t total_loss = 0.0
    cdef long start_idx, end_idx
    cdef int current_batch_size
    
    # Train in batches
    start_idx = 0
    while start_idx < total_examples:
        end_idx = start_idx + batch_size
        if end_idx > total_examples:
            end_idx = total_examples
        
        current_batch_size = end_idx - start_idx
        
        # Extract batch
        batch_input = input_indices[start_idx:end_idx]
        batch_output = output_indices[start_idx:end_idx]
        
        # Train batch
        total_loss += train_skipgram_batch(W, W_prime, batch_input, batch_output)
        
        start_idx = end_idx
    
    return total_loss, total_examples


def train_cbow_from_indexed_temporal(
    real_t[:, :] W,
    real_t[:, :] W_prime,
    list indexed_sentences,
    np.float32_t[:] discard_probs,
    int window,
    bint shrink_windows,
    bint use_subsampling,
    int batch_size,
    ITYPE_t[:] temporal_index_map
):
    """
    Generate CBOW examples with temporal mapping and train in batches.
    
    For TempRefWord2Vec CBOW: context words (inputs) are mapped to base forms,
    while center words (outputs) retain their temporal variant form.
    
    Args:
        W: Input word vectors
        W_prime: Output word vectors  
        indexed_sentences: List of numpy int32 arrays with word indices
        discard_probs: Subsampling discard probabilities
        window: Maximum context window size
        shrink_windows: Whether to use dynamic window sizing
        use_subsampling: Whether to apply subsampling
        batch_size: Number of examples per training batch
        temporal_index_map: Array mapping temporal variants to base word indices
    
    Returns:
        Tuple of (total_loss, total_examples)
    """
    # Generate all examples with temporal mapping
    context_flat, context_offsets, context_lengths, center_indices = generate_cbow_examples_temporal(
        indexed_sentences, discard_probs, window, shrink_windows,
        use_subsampling, temporal_index_map
    )
    
    cdef long total_examples = len(center_indices)
    cdef real_t total_loss = 0.0
    cdef long start_idx, end_idx
    cdef int current_batch_size
    
    # Train in batches using the optimized flat array interface
    start_idx = 0
    while start_idx < total_examples:
        end_idx = start_idx + batch_size
        if end_idx > total_examples:
            end_idx = total_examples
        
        current_batch_size = end_idx - start_idx
        
        # Extract batch slices
        batch_offsets = context_offsets[start_idx:end_idx]
        batch_lengths = context_lengths[start_idx:end_idx]
        batch_center = center_indices[start_idx:end_idx]
        
        # Train batch
        total_loss += train_cbow_batch(
            W, W_prime,
            context_flat,
            batch_offsets,
            batch_lengths,
            batch_center
        )
        
        start_idx = end_idx
    
    return total_loss, total_examples