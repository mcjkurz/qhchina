# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True
"""
Batch Fisher's exact test for 2x2 contingency tables.

Computes two-sided, less, or greater p-values for N tables,
entirely in C with the GIL released. Uses lgamma for log-factorial
computation to handle arbitrarily large table margins.

Uses early-termination and mode-gap skipping so that per-table cost
scales with the *effective* support (where probability mass lives)
rather than the full combinatorial support.
"""
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport lgamma, exp, log, INFINITY
from libc.math cimport llround
from libc.stdint cimport int64_t

cdef inline long long _maxn():
    """Find the largest N for which lgamma-based log-factorial is accurate."""
    cdef long long lo = 1
    cdef long long n = 2
    cdef double hi = INFINITY
    while lo < n:
        if abs(lgamma(<double>n + 1.0) - lgamma(<double>n) - log(<double>n)) >= 1.0:
            hi = <double>n
        else:
            lo = n
        n = (lo + <long long>min(hi, <double>(lo * 3))) // 2
    return lo

cdef long long MAXN = _maxn()

cdef double NAN_VAL = float('nan')


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
    """
    Sum of log-factorials of the four cells.

    log P(k) = margin_constant - _log_k_terms(k),
    so larger _log_k_terms => smaller probability.
    """
    return (_log_factorial(k) + _log_factorial(R1 - k) +
            _log_factorial(C1 - k) + _log_factorial(R2 - C1 + k))


cdef inline int _validate(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """Return 0 if valid, nonzero if invalid."""
    if a < 0 or b < 0 or c < 0 or d < 0:
        return 1
    if a + b + c + d > MAXN:
        return 2
    return 0


cdef double _fisher_twosided(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """
    Two-sided Fisher's exact test (sum-of-small-probabilities).

    Works in ratio space: accumulates sum of P(k)/P(a_obs) for all k
    where P(k) <= P(a_obs), then multiplies by P(a_obs) at the end.

    Uses llround(R1*C1/N) as the mode estimate to determine which side
    the observed value falls on, and skips the mode-gap region on the
    opposite tail.
    """
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_min, k_max, k, mode_k
    cdef double pa, pi, mc, sl, sr, st_new, pval, pa_tol

    if _validate(a, b, c, d) != 0:
        return NAN_VAL

    if N == 0:
        return 1.0

    k_min = max(0, C1 - R2)
    k_max = min(R1, C1)

    if a < k_min or a > k_max:
        return NAN_VAL

    if k_min == k_max:
        return 1.0

    pa = _log_k_terms(a, R1, C1, R2)
    mode_k = llround(<double>R1 * <double>C1 / <double>N)
    if mode_k < k_min:
        mode_k = k_min
    elif mode_k > k_max:
        mode_k = k_max

    # Tolerance for the "skip higher-prob terms" comparison on the opposite
    # tail.  lgamma-based log_k_terms can differ by ~1e-13 for mirror-
    # symmetric k values, so we need a small guard to avoid excluding the
    # symmetric counterpart of the observed value.
    pa_tol = pa - 1e-8

    sl = 0.0
    sr = 0.0

    if <double>R1 * <double>C1 < <double>a * <double>N:
        # a is right of the expected value (a > E[a] = R1*C1/N).
        # Opposite side (left): walk from min(a-1, mode) down to k_min.
        # Skip k where P(k) > P(a), i.e. where pi << pa.
        for k in range(min(a - 1, mode_k), k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            if pi < pa_tol:
                continue
            st_new = sl + exp(pa - pi)
            if st_new == sl:
                break
            sl = st_new

        # Same side (right): walk from a+1 up to k_max.
        # P decreases monotonically away from observed, early-terminate.
        for k in range(a + 1, k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            st_new = sr + exp(pa - pi)
            if st_new == sr:
                break
            sr = st_new

    else:
        # a is left of or at the expected value.
        # Same side (left): walk from a-1 down to k_min.
        for k in range(a - 1, k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            st_new = sl + exp(pa - pi)
            if st_new == sl:
                break
            sl = st_new

        # Opposite side (right): walk from max(a+1, mode) up to k_max.
        # Skip k where P(k) > P(a).
        for k in range(max(a + 1, mode_k), k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            if pi < pa_tol:
                continue
            st_new = sr + exp(pa - pi)
            if st_new == sr:
                break
            sr = st_new

    mc = _margin_constant(R1, R2, C1, N)
    pval = (sl + 1.0 + sr) * exp(mc - pa)

    if pval > 1.0:
        pval = 1.0
    if pval < 0.0:
        pval = 0.0
    return pval


cdef double _fisher_less(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """
    One-sided Fisher p-value: P(X <= a).

    Sums the shorter tail with early termination; uses complement
    when a is far from k_min.
    """
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_min, k_max, k
    cdef double pa, pi, mc, s, s_new, pval

    if _validate(a, b, c, d) != 0:
        return NAN_VAL

    if N == 0:
        return 1.0

    k_min = max(0, C1 - R2)
    k_max = min(R1, C1)

    if a < k_min or a > k_max:
        return NAN_VAL

    if a >= k_max:
        return 1.0

    pa = _log_k_terms(a, R1, C1, R2)
    mc = _margin_constant(R1, R2, C1, N)

    if a - k_min <= k_max - a:
        # Sum left tail directly: P(a) + P(a-1) + ... + P(k_min)
        s = 1.0
        for k in range(a - 1, k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = s * exp(mc - pa)
    else:
        # Sum right tail and complement: 1 - [P(a+1) + ... + P(k_max)]
        s = 0.0
        for k in range(a + 1, k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = 1.0 - s * exp(mc - pa)
        if pval < 0.0:
            pval = 0.0

    if pval > 1.0:
        pval = 1.0
    return pval


cdef double _fisher_greater(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """
    One-sided Fisher p-value: P(X >= a).

    Sums the shorter tail with early termination; uses complement
    when a is far from k_max.
    """
    cdef int64_t R1 = a + b
    cdef int64_t R2 = c + d
    cdef int64_t C1 = a + c
    cdef int64_t N = R1 + R2
    cdef int64_t k_min, k_max, k
    cdef double pa, pi, mc, s, s_new, pval

    if _validate(a, b, c, d) != 0:
        return NAN_VAL

    if N == 0:
        return 1.0

    k_min = max(0, C1 - R2)
    k_max = min(R1, C1)

    if a < k_min or a > k_max:
        return NAN_VAL

    if a <= k_min:
        return 1.0

    pa = _log_k_terms(a, R1, C1, R2)
    mc = _margin_constant(R1, R2, C1, N)

    if k_max - a <= a - k_min:
        # Sum right tail directly: P(a) + P(a+1) + ... + P(k_max)
        s = 1.0
        for k in range(a + 1, k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = s * exp(mc - pa)
    else:
        # Sum left tail and complement: 1 - [P(k_min) + ... + P(a-1)]
        s = 0.0
        for k in range(a - 1, k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = 1.0 - s * exp(mc - pa)
        if pval < 0.0:
            pval = 0.0

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
        All values must be non-negative.  The grand total (a+b+c+d)
        must not exceed the lgamma accuracy limit (~10^15).
    alternative : {'two-sided', 'less', 'greater'}

    Returns
    -------
    numpy.ndarray of float64 p-values, length N.

    Raises
    ------
    ValueError
        If input arrays differ in length, alternative is invalid,
        any cell is negative, or any table has infeasible margins.
    OverflowError
        If any table's grand total exceeds the lgamma accuracy limit.
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

    cdef Py_ssize_t i
    cdef int64_t ai, bi, ci, di
    cdef int64_t k_min_i, k_max_i

    for i in range(n):
        ai = a_arr[i]; bi = b_arr[i]; ci = c_arr[i]; di = d_arr[i]
        if ai < 0 or bi < 0 or ci < 0 or di < 0:
            raise ValueError(
                f"Negative cell value in table {i}: ({ai}, {bi}, {ci}, {di})")
        if ai + bi + ci + di > MAXN:
            raise OverflowError(
                f"Grand total of table {i} ({ai + bi + ci + di}) exceeds "
                f"lgamma accuracy limit ({MAXN})")
        k_min_i = max(0, (ai + ci) - (ci + di))
        k_max_i = min(ai + bi, ai + ci)
        if ai < k_min_i or ai > k_max_i:
            raise ValueError(
                f"Cell 'a' in table {i} is outside feasible range: "
                f"a={ai}, range=[{k_min_i}, {k_max_i}]")

    pvals_np = np.empty(n, dtype=np.float64)
    cdef double[::1] pvals = pvals_np

    for i in prange(n, nogil=True):
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
