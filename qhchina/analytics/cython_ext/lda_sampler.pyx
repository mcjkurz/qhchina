import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log

# Define C-level types for better performance
ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE_t

# Function to sample from a multinomial distribution
cdef int _sample_multinomial(double[:] p, int length) nogil:
    """Sample from a discrete probability distribution.
    
    This is a fast implementation of multinomial sampling using
    cumulative probabilities and binary search.
    """
    cdef double r = <double>rand() / RAND_MAX
    cdef double cumsum = 0
    cdef int i
    
    for i in range(length):
        cumsum += p[i]
        if r < cumsum:
            return i
    
    # In case of numerical issues, return the last index
    return length - 1

def sample_topic(np.ndarray[INT_t, ndim=2] n_wt, 
                np.ndarray[INT_t, ndim=2] n_dt, 
                np.ndarray[INT_t, ndim=1] n_t, 
                np.ndarray[INT_t, ndim=2] z,
                int d, int i, int w, 
                double alpha, double beta, 
                int n_topics, int vocab_size):
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
    cdef int old_topic = z[d, i]
    cdef np.ndarray[DOUBLE_t, ndim=1] p = np.zeros(n_topics, dtype=np.float64)
    cdef double p_sum = 0.0
    cdef int k
    
    # Decrease counts for current topic assignment
    n_wt[w, old_topic] -= 1
    n_dt[d, old_topic] -= 1
    n_t[old_topic] -= 1
    
    # Calculate probability for each topic
    for k in range(n_topics):
        p[k] = (n_wt[w, k] + beta) / (n_t[k] + vocab_size * beta) * (n_dt[d, k] + alpha)
        p_sum += p[k]
    
    # Normalize probabilities
    for k in range(n_topics):
        p[k] /= p_sum
    
    # Sample new topic
    cdef int new_topic = _sample_multinomial(p, n_topics)
    
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
    cdef int d, i, w, doc_len
    cdef list doc
    
    for d in range(len(docs_tokens)):
        doc = docs_tokens[d]
        doc_len = len(doc)
        
        for i in range(doc_len):
            w = doc[i]
            z[d, i] = sample_topic(n_wt, n_dt, n_t, z, d, i, w, alpha, beta, n_topics, vocab_size)
    
    return z 