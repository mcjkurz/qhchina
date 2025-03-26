import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

# Define a global probability buffer to reuse
cdef:
    double[:] PROB_BUFFER = np.zeros(100, dtype=np.float64)  # Initial size, will be resized if needed

# Function to sample from a multinomial distribution using linear search
cdef int _sample_multinomial_linear(double[:] p_cumsum, int length) nogil:
    """Sample from a discrete probability distribution using linear search.
    
    This is a simple implementation of multinomial sampling using
    cumulative probabilities and linear search.
    """
    cdef double r = <double>rand() / RAND_MAX
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
    global PROB_BUFFER
    
    cdef int old_topic = z[d, i]
    cdef double p_sum = 0.0
    cdef int k
    # Precompute vocab_size * beta for efficiency
    cdef double vocab_size_beta = vocab_size * beta
    
    # Decrease counts for current topic assignment
    n_wt[w, old_topic] -= 1
    n_dt[d, old_topic] -= 1
    n_t[old_topic] -= 1
    
    # Calculate probability for each topic directly into the buffer
    for k in range(n_topics):
        PROB_BUFFER[k] = (n_wt[w, k] + beta) / (n_t[k] + vocab_size_beta) * (n_dt[d, k] + alpha)
        p_sum += PROB_BUFFER[k]
    
    # Convert to cumulative probabilities for linear search
    PROB_BUFFER[0] /= p_sum
    for k in range(1, n_topics):
        PROB_BUFFER[k] = PROB_BUFFER[k-1] + (PROB_BUFFER[k] / p_sum)
    
    # Sample new topic using linear search (faster for small n_topics)
    cdef int new_topic = _sample_multinomial_linear(PROB_BUFFER, n_topics)
    
    # Update counts for new topic assignment
    n_wt[w, new_topic] += 1
    n_dt[d, new_topic] += 1
    n_t[new_topic] += 1
    
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
    global PROB_BUFFER
    cdef int d, i, w, doc_len, num_docs
    cdef list doc
    
    # Ensure buffer is exactly equal to n_topics (the exact size needed)
    if PROB_BUFFER.shape[0] != n_topics:
        PROB_BUFFER = np.zeros(n_topics, dtype=np.float64)
    
    num_docs = len(docs_tokens)
    
    for d in range(num_docs):
        doc = docs_tokens[d]
        doc_len = len(doc)
        
        for i in range(doc_len):
            w = doc[i]
            z[d, i] = sample_topic(n_wt, n_dt, n_t, z, d, i, w, alpha, beta, n_topics, vocab_size)
    
    return z 