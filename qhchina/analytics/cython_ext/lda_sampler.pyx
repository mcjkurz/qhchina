import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport log
from libc.time cimport time

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

# Define global buffers for reuse
cdef double[:] PROB_BUFFER = np.zeros(100, dtype=np.float64)  # Initial size, will be resized if needed
cdef double[::1] TOPIC_NORMALIZERS  # Buffer for topic normalizers (C-contiguous)
cdef double VOCAB_SIZE_BETA = 0.0  # Global to store vocab_size * beta

# Define Xorshift128+ state structure
cdef struct xorshift128plus_state:
    unsigned long long s0
    unsigned long long s1

# Initialize global RNG state
cdef xorshift128plus_state RNG_STATE

# Initialize the RNG state with a seed
cdef void seed_xorshift128plus(unsigned long long seed) nogil:
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
cdef inline double xorshift128plus_random() nogil:
    cdef unsigned long long s1 = RNG_STATE.s0
    cdef unsigned long long s0 = RNG_STATE.s1
    RNG_STATE.s0 = s0
    s1 ^= s1 << 23
    RNG_STATE.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)
    return (RNG_STATE.s1 + s0) / 18446744073709551616.0  # Divide by 2^64

# Function to sample from a multinomial distribution using linear search
cdef int _sample_multinomial_linear(double* p_cumsum, int length) nogil:
    """Sample from a discrete probability distribution using linear search.
    
    This is a simple implementation of multinomial sampling using
    cumulative probabilities and linear search.
    
    Args:
        p_cumsum: Pointer to array containing cumulative probabilities
        length: Length of the array
        
    Returns:
        Sampled index
    """
    # Use our fast xorshift128+ RNG instead of rand()
    cdef double r = xorshift128plus_random()
    cdef int k
    
    # Linear search on the cumulative probability array
    for k in range(length):
        if r <= p_cumsum[k]:
            return k
    
    # In case of numerical issues, return the last topic
    return length - 1

def sample_topic(np.ndarray[INT_t, ndim=2] n_wt, 
                np.ndarray[INT_t, ndim=2] n_dt, 
                np.ndarray[INT_t, ndim=1] n_t, 
                np.ndarray[INT_t, ndim=2] z,
                int d, int i, int w, 
                double alpha, double beta, 
                int n_topics, int vocab_size) nogil:
    """
    Optimized Cython implementation of topic sampling for LDA Gibbs sampler.
    
    Args:
        n_wt: Word-topic count matrix (vocab_size, n_topics)
        n_dt: Document-topic count matrix (n_docs, n_topics)
        n_t: Topic count vector (n_topics)
        z: Topic assignments (n_docs, max_doc_length)
        d: Document ID
        i: Position in document
        w: Word ID
        alpha: Dirichlet prior for document-topic distributions
        beta: Dirichlet prior for topic-word distributions
        n_topics: Number of topics
        vocab_size: Size of vocabulary
        
    Returns:
        Sampled topic ID
    """
    global PROB_BUFFER, TOPIC_NORMALIZERS

    cdef int old_topic = z[d, i]
    cdef double p_sum = 0.0
    cdef int k

    # Decrease counts for current topic assignment
    n_wt[w, old_topic] -= 1
    n_dt[d, old_topic] -= 1
    n_t[old_topic] -= 1
    
    # Update the normalizer for the old topic
    TOPIC_NORMALIZERS[old_topic] = 1.0 / (n_t[old_topic] + VOCAB_SIZE_BETA)
    
    # Calculate probability for each topic directly into the buffer
    for k in range(n_topics):
        PROB_BUFFER[k] = (n_wt[w, k] + beta) * (n_dt[d, k] + alpha) * TOPIC_NORMALIZERS[k]
        p_sum += PROB_BUFFER[k]
    
    # Convert to cumulative probabilities for linear search
    PROB_BUFFER[0] /= p_sum
    for k in range(1, n_topics):
        PROB_BUFFER[k] = PROB_BUFFER[k-1] + (PROB_BUFFER[k] / p_sum)
    
    cdef int new_topic = _sample_multinomial_linear(&PROB_BUFFER[0], n_topics)
    
    # Update counts for new topic assignment
    n_wt[w, new_topic] += 1
    n_dt[d, new_topic] += 1
    n_t[new_topic] += 1
    
    # Update the normalizer for the new topic
    TOPIC_NORMALIZERS[new_topic] = 1.0 / (n_t[new_topic] + VOCAB_SIZE_BETA)
    
    return new_topic
    
def run_iteration(np.ndarray[INT_t, ndim=2] n_wt,
                 np.ndarray[INT_t, ndim=2] n_dt,
                 np.ndarray[INT_t, ndim=1] n_t,
                 np.ndarray[INT_t, ndim=2] z,
                 list docs_tokens,
                 double alpha, double beta,
                 int n_topics, int vocab_size):
    """
    Run a full iteration of Gibbs sampling over all documents and words.
    
    This is highly optimized by combining the iteration loop with the sampling
    logic in Cython.
    """
    global PROB_BUFFER, TOPIC_NORMALIZERS, VOCAB_SIZE_BETA
    cdef int d, i, w, doc_len, num_docs, k
    cdef list doc
    
    # Ensure RNG is seeded (only needs to be done once, but cheap to repeat)
    seed_xorshift128plus(<unsigned long long>time(NULL))
    
    # Compute vocab_size_beta once for the entire run
    VOCAB_SIZE_BETA = vocab_size * beta
    
    # Ensure buffer is exactly equal to n_topics (the exact size needed)
    if PROB_BUFFER.shape[0] != n_topics:
        PROB_BUFFER = np.empty(n_topics, dtype=np.float64)
    
    # Initialize or resize the topic normalizers array
    TOPIC_NORMALIZERS = np.empty(n_topics, dtype=np.float64)
    
    # Pre-compute topic normalizers
    for k in range(n_topics):
        TOPIC_NORMALIZERS[k] = 1.0 / (n_t[k] + VOCAB_SIZE_BETA)
    
    num_docs = len(docs_tokens)
    
    for d in range(num_docs):
        doc = docs_tokens[d]
        doc_len = len(doc)
        
        for i in range(doc_len):
            w = doc[i]
            z[d, i] = sample_topic(n_wt, n_dt, n_t, z, d, i, w, alpha, beta, n_topics, vocab_size)
    
    return z 