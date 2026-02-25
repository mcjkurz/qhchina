# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Batch Fisher's exact test for 2x2 contingency tables.

Computes two-sided, less, or greater p-values for N tables at once,
entirely in C with the GIL released. Uses lgamma for log-factorial
computation to handle arbitrarily large table margins.
"""
import numpy as np
cimport numpy as np
from libc.math cimport lgamma, exp
from libc.stdint cimport int64_t


cdef inline double _log_factorial(int64_t n) noexcept nogil:
    return lgamma(<double>(n + 1))


cdef inline double _margin_constant(int64_t R1, int64_t R2,
                                    int64_t C1, int64_t N) noexcept nogil:
    """Log-PMF terms that depend only on margins (constant across all k)."""
    return (_log_factorial(R1) + _log_factorial(R2) +
            _log_factorial(C1) + _log_factorial(N - C1) -
            _log_factorial(N))


cdef inline double _log_k_terms(int64_t k, int64_t R1,
                                int64_t C1, int64_t R2) noexcept nogil:
    """Log-PMF terms that vary with k."""
    return (-_log_factorial(k) - _log_factorial(R1 - k) -
            _log_factorial(C1 - k) - _log_factorial(R2 - C1 + k))


cdef double _fisher_twosided(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """Two-sided Fisher's exact test p-value (sum-of-small-probabilities method)."""
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_min, k_max, k
    cdef double mc, log_p_obs, log_pk, pval

    if N == 0:
        return 1.0

    k_min = max(0, C1 - R2)
    k_max = min(R1, C1)

    if k_min == k_max:
        return 1.0

    mc = _margin_constant(R1, R2, C1, N)
    log_p_obs = mc + _log_k_terms(a, R1, C1, R2)

    # Tolerance accounts for catastrophic cancellation in lgamma differences.
    # log-PMF sums lgamma values of order N*log(N); for N~10^7 the absolute
    # error is ~1.6e8 * eps_machine ≈ 1.6e-7, so 1e-6 is a safe bound.
    pval = 0.0
    for k in range(k_min, k_max + 1):
        log_pk = mc + _log_k_terms(k, R1, C1, R2)
        if log_pk <= log_p_obs + 1e-6:
            pval = pval + exp(log_pk)

    if pval > 1.0:
        pval = 1.0
    return pval


cdef double _fisher_less(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """One-sided Fisher p-value: P(X <= a)."""
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_min, k
    cdef double mc, pval, log_pk

    if N == 0:
        return 1.0

    k_min = max(0, C1 - R2)
    mc = _margin_constant(R1, R2, C1, N)

    pval = 0.0
    for k in range(k_min, a + 1):
        log_pk = mc + _log_k_terms(k, R1, C1, R2)
        pval = pval + exp(log_pk)

    if pval > 1.0:
        pval = 1.0
    return pval


cdef double _fisher_greater(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """One-sided Fisher p-value: P(X >= a)."""
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_max, k
    cdef double mc, pval, log_pk

    if N == 0:
        return 1.0

    k_max = min(R1, C1)
    mc = _margin_constant(R1, R2, C1, N)

    pval = 0.0
    for k in range(a, k_max + 1):
        log_pk = mc + _log_k_terms(k, R1, C1, R2)
        pval = pval + exp(log_pk)

    if pval > 1.0:
        pval = 1.0
    return pval


def batch_fisher_exact(np.int64_t[::1] a_arr, np.int64_t[::1] b_arr,
                       np.int64_t[::1] c_arr, np.int64_t[::1] d_arr,
                       str alternative="two-sided"):
    """
    Vectorised Fisher's exact test for N independent 2x2 tables.

    Parameters
    ----------
    a_arr, b_arr, c_arr, d_arr : 1-D int64 arrays (contiguous)
        Cell values for each table: [[a, b], [c, d]].
    alternative : {'two-sided', 'less', 'greater'}

    Returns
    -------
    numpy.ndarray of float64 p-values, length N.
    """
    cdef Py_ssize_t n = a_arr.shape[0]
    if b_arr.shape[0] != n or c_arr.shape[0] != n or d_arr.shape[0] != n:
        raise ValueError("All input arrays must have the same length")

    cdef int mode
    if alternative == "two-sided":
        mode = 0
    elif alternative == "less":
        mode = 1
    elif alternative == "greater":
        mode = 2
    else:
        raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater', got '{alternative}'")

    pvals_np = np.empty(n, dtype=np.float64)
    cdef double[::1] pvals = pvals_np
    cdef Py_ssize_t i

    with nogil:
        for i in range(n):
            if mode == 0:
                pvals[i] = _fisher_twosided(a_arr[i], b_arr[i],
                                            c_arr[i], d_arr[i])
            elif mode == 1:
                pvals[i] = _fisher_less(a_arr[i], b_arr[i],
                                        c_arr[i], d_arr[i])
            else:
                pvals[i] = _fisher_greater(a_arr[i], b_arr[i],
                                           c_arr[i], d_arr[i])

    return pvals_np
