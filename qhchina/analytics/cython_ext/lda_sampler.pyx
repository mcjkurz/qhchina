# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Thread-safe LDA Gibbs sampler implementation.

All state (probability buffers, normalizers, RNG) is passed as function parameters
rather than using module-level globals. This allows safe concurrent use from multiple threads.
"""
import numpy as np
cimport numpy as np
from libc.math cimport log, exp
from libc.time cimport time

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

# Define Xorshift128+ state structure
cdef struct xorshift128plus_state:
    unsigned long long s0
    unsigned long long s1

# Initialize RNG state from a seed (pure function, no global state)
cdef void seed_xorshift128plus(xorshift128plus_state* state, unsigned long long seed):
    cdef unsigned long long z = seed
    # Use splitmix64 algorithm to initialize state from seed
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL
    state.s0 = z ^ (z >> 31)
    
    z = (seed + 1)
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL
    state.s1 = z ^ (z >> 31)

# Fast Xorshift128+ random number generation (returns double in range [0,1))
cdef inline double xorshift128plus_random(xorshift128plus_state* state) noexcept nogil:
    cdef unsigned long long s1 = state.s0
    cdef unsigned long long s0 = state.s1
    state.s0 = s0
    s1 ^= s1 << 23
    state.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)
    return (state.s1 + s0) / 18446744073709551616.0  # Divide by 2^64

# Function to sample from a multinomial distribution using binary search
cdef int _sample_multinomial_binary(double* p_cumsum, int length, 
                                    xorshift128plus_state* rng_state) noexcept nogil:
    """Sample from a discrete probability distribution using binary search.
    
    Args:
        p_cumsum: Pointer to array containing cumulative probabilities
        length: Length of the array
        rng_state: Pointer to the RNG state
        
    Returns:
        Sampled index
    """
    cdef double r = xorshift128plus_random(rng_state)
    cdef int left = 0
    cdef int right = length - 1
    cdef int mid
    
    # Binary search on the cumulative probability array
    while left < right:
        mid = (left + right) // 2
        if r <= p_cumsum[mid]:
            right = mid
        else:
            left = mid + 1
    
    return left

# Helper function to normalize probabilities to cumulative distribution
cdef inline void normalize_to_cumsum(double* probs, int length, double p_sum) noexcept nogil:
    """Convert probability array to cumulative probability distribution.
    
    Args:
        probs: Pointer to probability array (modified in-place)
        length: Length of the array
        p_sum: Sum of all probabilities
    """
    cdef int k
    
    # Guard against zero or negative p_sum (should not happen with valid data)
    if p_sum <= 0.0:
        # Fallback to uniform distribution
        for k in range(length - 1):
            probs[k] = (k + 1.0) / length
        probs[length - 1] = 1.0
        return
    
    # Normalize first element
    probs[0] /= p_sum
    
    # Convert to cumulative probabilities
    for k in range(1, length - 1):
        probs[k] = probs[k-1] + (probs[k] / p_sum)
    probs[length - 1] = 1.0

cdef inline int sample_topic(INT_t[:, ::1] n_wt, 
                             INT_t[:, ::1] n_dt, 
                             INT_t[::1] n_t, 
                             INT_t[:, ::1] z,
                             int d, int i, int w, 
                             DOUBLE_t[::1] alpha, double beta, 
                             int n_topics,
                             double* prob_buffer,
                             double* topic_normalizers,
                             double vocab_size_beta,
                             xorshift128plus_state* rng_state) noexcept nogil:
    """
    Thread-safe topic sampling for LDA Gibbs sampler.
    
    Args:
        n_wt: Word-topic count matrix (vocab_size, n_topics)
        n_dt: Document-topic count matrix (n_docs, n_topics)
        n_t: Topic count vector (n_topics)
        z: Topic assignments (n_docs, max_doc_length)
        d: Document ID
        i: Position in document
        w: Word ID
        alpha: Dirichlet prior for document-topic distributions (array)
        beta: Dirichlet prior for topic-word distributions
        n_topics: Number of topics
        prob_buffer: Pre-allocated probability buffer (size n_topics)
        topic_normalizers: Pre-computed topic normalizers (size n_topics)
        vocab_size_beta: Pre-computed vocab_size * beta
        rng_state: Pointer to thread-local RNG state
        
    Returns:
        Sampled topic ID
    """
    cdef int old_topic = z[d, i]
    cdef double p_sum = 0.0
    cdef int k

    # Decrease counts for current topic assignment
    n_wt[w, old_topic] -= 1
    n_dt[d, old_topic] -= 1
    n_t[old_topic] -= 1
    
    # Update the normalizer for the old topic
    topic_normalizers[old_topic] = 1.0 / (n_t[old_topic] + vocab_size_beta)
    
    # Calculate probability for each topic directly into the buffer
    for k in range(n_topics):
        prob_buffer[k] = (n_wt[w, k] + beta) * (n_dt[d, k] + alpha[k]) * topic_normalizers[k]
        p_sum += prob_buffer[k]
    
    # Convert to cumulative probabilities using helper function
    normalize_to_cumsum(prob_buffer, n_topics, p_sum)
    
    # Use binary search for sampling
    cdef int new_topic = _sample_multinomial_binary(prob_buffer, n_topics, rng_state)
    
    # Update counts for new topic assignment
    n_wt[w, new_topic] += 1
    n_dt[d, new_topic] += 1
    n_t[new_topic] += 1
    
    # Update the normalizer for the new topic
    topic_normalizers[new_topic] = 1.0 / (n_t[new_topic] + vocab_size_beta)
    
    return new_topic
    
def run_iteration(INT_t[:, ::1] n_wt,
                 INT_t[:, ::1] n_dt,
                 INT_t[::1] n_t,
                 INT_t[:, ::1] z,
                 INT_t[:, ::1] docs_tokens,
                 INT_t[::1] doc_lengths,
                 DOUBLE_t[::1] alpha, double beta,
                 int n_topics, int vocab_size,
                 seed=None):
    """
    Run a full iteration of Gibbs sampling over all documents and words.
    
    This function is thread-safe - all state is local to the function call.
    
    Args:
        n_wt: Word-topic count matrix (vocab_size, n_topics)
        n_dt: Document-topic count matrix (n_docs, n_topics)
        n_t: Topic count vector (n_topics)
        z: Topic assignments (n_docs, max_doc_length)
        docs_tokens: 2D array with shape (n_docs, max_doc_length), padded with -1
        doc_lengths: Array of actual document lengths
        alpha: Dirichlet prior for document-topic distributions
        beta: Dirichlet prior for topic-word distributions
        n_topics: Number of topics
        vocab_size: Size of vocabulary
        seed: Random seed (if None, uses current time)
        
    Returns:
        Updated topic assignments z
    """
    cdef int d, i, w, doc_len, num_docs, k
    cdef xorshift128plus_state rng_state
    cdef double vocab_size_beta
    
    # Initialize RNG state
    cdef unsigned long long seed_value
    if seed is None:
        seed_value = <unsigned long long>time(NULL)
    else:
        seed_value = <unsigned long long>seed
    seed_xorshift128plus(&rng_state, seed_value)
    
    # Allocate thread-local buffers
    cdef double[::1] prob_buffer = np.empty(n_topics, dtype=np.float64)
    cdef double[::1] topic_normalizers = np.empty(n_topics, dtype=np.float64)
    
    # Compute vocab_size_beta once for the entire run
    vocab_size_beta = vocab_size * beta
    
    # Pre-compute topic normalizers
    for k in range(n_topics):
        topic_normalizers[k] = 1.0 / (n_t[k] + vocab_size_beta)
    
    num_docs = docs_tokens.shape[0]
    
    for d in range(num_docs):
        doc_len = doc_lengths[d]
        
        for i in range(doc_len):
            w = docs_tokens[d, i]
            z[d, i] = sample_topic(n_wt, n_dt, n_t, z, d, i, w, alpha, beta, n_topics,
                                   &prob_buffer[0], &topic_normalizers[0], 
                                   vocab_size_beta, &rng_state)
    
    return z.base if z.base is not None else np.asarray(z) 

def calculate_perplexity(
    DOUBLE_t[:, ::1] phi,     # Topic-word distributions (n_topics, vocab_size)
    DOUBLE_t[:, ::1] theta,   # Document-topic distributions (n_docs, n_topics) 
    INT_t[:, ::1] docs_tokens, # Document tokens as word IDs (2D array, padded with -1)
    INT_t[::1] doc_lengths    # Actual document lengths
):
    """
    Optimized Cython implementation for perplexity calculation.
    
    This function is thread-safe - it uses no global state.
    
    Args:
        phi: Topic-word distributions (n_topics, vocab_size)
        theta: Document-topic distributions (n_docs, n_topics)
        docs_tokens: 2D array with shape (n_docs, max_doc_length), padded with -1
        doc_lengths: Array of actual document lengths
        
    Returns:
        Perplexity value (lower is better)
    """
    cdef int n_docs = docs_tokens.shape[0]
    cdef int n_topics = phi.shape[0]
    cdef int d, w, k, doc_len, i
    cdef double log_likelihood = 0.0
    cdef double word_prob
    cdef long total_tokens = 0
    
    # For each document
    for d in range(n_docs):
        doc_len = doc_lengths[d]
        
        if doc_len == 0:
            continue
            
        # For each word in document
        for i in range(doc_len):
            w = docs_tokens[d, i]
            # Calculate P(word|doc) = sum_k P(word|topic_k) * P(topic_k|doc)
            word_prob = 0.0
            for k in range(n_topics):
                word_prob += phi[k, w] * theta[d, k]
            
            # Prevent log(0) errors with a small epsilon
            if word_prob > 0:
                log_likelihood += log(word_prob)
            else:
                log_likelihood += log(1e-10)  # Small epsilon value
        
        total_tokens += doc_len
    
    # If no tokens processed, return infinity
    if total_tokens == 0:
        return float('inf')
    
    # Perplexity = exp(-log_likelihood / total_tokens)
    return exp(-log_likelihood / total_tokens)

def run_inference(
    INT_t[:, ::1] n_wt,           # Word-topic count matrix (vocab_size, n_topics)
    INT_t[::1] n_t,               # Topic count vector (n_topics)
    INT_t[::1] new_doc_tokens,    # New document tokens as word IDs
    DOUBLE_t[::1] alpha,          # Dirichlet prior for document-topic distributions
    double beta,                  # Dirichlet prior for topic-word distributions
    int n_topics,                 # Number of topics
    int vocab_size,               # Size of vocabulary
    int inference_iterations,     # Number of sampling iterations
    seed=None                     # Random seed (if None, uses current time)
):
    """
    Thread-safe inference of topic distribution for a new document.
    
    Args:
        n_wt: Word-topic count matrix from trained model (vocab_size, n_topics)
        n_t: Topic count vector from trained model (n_topics)
        new_doc_tokens: New document tokens as word IDs (already filtered to be in vocabulary)
        alpha: Dirichlet prior for document-topic distributions (array)
        beta: Dirichlet prior for topic-word distributions
        n_topics: Number of topics
        vocab_size: Size of vocabulary
        inference_iterations: Number of sampling iterations for inference
        seed: Random seed (if None, uses current time)
        
    Returns:
        Document-topic distribution (numpy array of length n_topics)
    """
    cdef int doc_len = new_doc_tokens.shape[0]
    cdef int i, w, k, iteration
    cdef double p_sum, alpha_sum
    cdef int old_topic, new_topic
    cdef INT_t[::1] z_doc, n_dt_doc
    cdef DOUBLE_t[::1] theta_doc
    cdef xorshift128plus_state rng_state
    cdef double vocab_size_beta
    
    # If document is empty, return uniform distribution
    if doc_len == 0:
        return np.ones(n_topics, dtype=np.float64) / n_topics
    
    # Initialize RNG state
    cdef unsigned long long seed_value
    if seed is None:
        seed_value = <unsigned long long>time(NULL)
    else:
        seed_value = <unsigned long long>seed
    seed_xorshift128plus(&rng_state, seed_value)
    
    # Allocate thread-local buffers
    cdef double[::1] prob_buffer = np.empty(n_topics, dtype=np.float64)
    cdef double[::1] topic_normalizers = np.empty(n_topics, dtype=np.float64)
    
    # Compute vocab_size_beta once
    vocab_size_beta = vocab_size * beta
    
    # Pre-compute topic normalizers from the trained model
    for k in range(n_topics):
        topic_normalizers[k] = 1.0 / (n_t[k] + vocab_size_beta)
    
    # Initialize topic assignments randomly for the new document
    z_doc_arr = np.empty(doc_len, dtype=np.int32)
    z_doc = z_doc_arr
    for i in range(doc_len):
        z_doc[i] = <int>(xorshift128plus_random(&rng_state) * n_topics)
    
    # Initialize document-topic counts for the new document
    n_dt_doc_arr = np.zeros(n_topics, dtype=np.int32)
    n_dt_doc = n_dt_doc_arr
    for i in range(doc_len):
        n_dt_doc[z_doc[i]] += 1
    
    # Precompute alpha sum (used for final theta calculation)
    alpha_sum = 0.0
    for k in range(n_topics):
        alpha_sum += alpha[k]
    
    # Run Gibbs sampling for inference iterations
    for iteration in range(inference_iterations):
        for i in range(doc_len):
            w = new_doc_tokens[i]
            old_topic = z_doc[i]
            
            # Remove current topic assignment
            n_dt_doc[old_topic] -= 1
            
            # Calculate probabilities for each topic
            p_sum = 0.0
            for k in range(n_topics):
                # Use the trained model's word-topic counts (n_wt) and topic counts (n_t)
                # but only the new document's document-topic counts (n_dt_doc)
                prob_buffer[k] = (n_wt[w, k] + beta) * (n_dt_doc[k] + alpha[k]) * topic_normalizers[k]
                p_sum += prob_buffer[k]
            
            # Convert to cumulative probabilities using helper function
            normalize_to_cumsum(&prob_buffer[0], n_topics, p_sum)
            
            # Sample new topic using binary search
            new_topic = _sample_multinomial_binary(&prob_buffer[0], n_topics, &rng_state)
            
            # Update assignment
            z_doc[i] = new_topic
            n_dt_doc[new_topic] += 1
    
    theta_doc_arr = np.empty(n_topics, dtype=np.float64)
    theta_doc = theta_doc_arr
    for k in range(n_topics):
        theta_doc[k] = (n_dt_doc[k] + alpha[k]) / (doc_len + alpha_sum)
    
    return theta_doc_arr 
