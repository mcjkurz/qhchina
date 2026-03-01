# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True
"""
Batch statistical tests for 2x2 contingency tables.

Provides vectorised Fisher's exact test, Pearson's chi-squared test,
and log-likelihood ratio (G2) test, all operating on N independent
2x2 tables entirely in C with the GIL released.

Fisher's exact test uses lgamma for log-factorial computation to handle
arbitrarily large table margins, with early-termination and mode-gap
skipping so that per-table cost scales with the *effective* support.

Chi-squared and G2 use scipy's chdtrc (chi-squared survival function)
via the Cython-callable C interface for nogil p-value computation.
"""
import numpy as np
cimport numpy as np
from libc.math cimport lgamma, exp, log, fabs, INFINITY
from libc.math cimport llround
from libc.stdint cimport int64_t
from scipy.special.cython_special cimport chdtrc


# ===========================================================================
# Fisher's exact test internals
# ===========================================================================

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
        for k in range(min(a - 1, mode_k), k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            if pi < pa_tol:
                continue
            st_new = sl + exp(pa - pi)
            if st_new == sl:
                break
            sl = st_new

        for k in range(a + 1, k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            st_new = sr + exp(pa - pi)
            if st_new == sr:
                break
            sr = st_new

    else:
        for k in range(a - 1, k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            st_new = sl + exp(pa - pi)
            if st_new == sl:
                break
            sl = st_new

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
        s = 1.0
        for k in range(a - 1, k_min - 1, -1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = s * exp(mc - pa)
    else:
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
        s = 1.0
        for k in range(a + 1, k_max + 1):
            pi = _log_k_terms(k, R1, C1, R2)
            s_new = s + exp(pa - pi)
            if s_new == s:
                break
            s = s_new
        pval = s * exp(mc - pa)
    else:
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


# ===========================================================================
# Chi-squared test internals
# ===========================================================================

cdef inline double _chi2_stat(int64_t a, int64_t b, int64_t c, int64_t d,
                              int correction) noexcept nogil:
    """
    Pearson's chi-squared statistic for a single 2x2 table.

    With Yates' continuity correction when correction=1.
    Returns NaN if any expected value is zero.
    """
    cdef double N = <double>(a + b + c + d)
    cdef double R1 = <double>(a + b)
    cdef double R2 = <double>(c + d)
    cdef double C1 = <double>(a + c)
    cdef double C2 = <double>(b + d)
    cdef double E_a, E_b, E_c, E_d
    cdef double diff, yates, chi2

    if N == 0.0:
        return NAN_VAL

    E_a = R1 * C1 / N
    E_b = R1 * C2 / N
    E_c = R2 * C1 / N
    E_d = R2 * C2 / N

    if E_a == 0.0 or E_b == 0.0 or E_c == 0.0 or E_d == 0.0:
        return NAN_VAL

    yates = 0.5 if correction else 0.0

    chi2 = 0.0
    diff = fabs(<double>a - E_a) - yates
    if diff > 0.0:
        chi2 += diff * diff / E_a
    diff = fabs(<double>b - E_b) - yates
    if diff > 0.0:
        chi2 += diff * diff / E_b
    diff = fabs(<double>c - E_c) - yates
    if diff > 0.0:
        chi2 += diff * diff / E_c
    diff = fabs(<double>d - E_d) - yates
    if diff > 0.0:
        chi2 += diff * diff / E_d

    return chi2


# ===========================================================================
# Log-likelihood ratio (G2) internals
# ===========================================================================

cdef inline double _g2_stat(int64_t a, int64_t b, int64_t c, int64_t d) noexcept nogil:
    """
    Log-likelihood ratio (G2) statistic for a single 2x2 table.

    G2 = 2 * sum(O_ij * ln(O_ij / E_ij)) for cells where O_ij > 0.
    Returns NaN if N is zero.
    """
    cdef double N = <double>(a + b + c + d)
    cdef double R1 = <double>(a + b)
    cdef double R2 = <double>(c + d)
    cdef double C1 = <double>(a + c)
    cdef double C2 = <double>(b + d)
    cdef double E_a, E_b, E_c, E_d
    cdef double g2

    if N == 0.0:
        return NAN_VAL

    E_a = R1 * C1 / N
    E_b = R1 * C2 / N
    E_c = R2 * C1 / N
    E_d = R2 * C2 / N

    g2 = 0.0
    if a > 0 and E_a > 0.0:
        g2 += <double>a * log(<double>a / E_a)
    if b > 0 and E_b > 0.0:
        g2 += <double>b * log(<double>b / E_b)
    if c > 0 and E_c > 0.0:
        g2 += <double>c * log(<double>c / E_c)
    if d > 0 and E_d > 0.0:
        g2 += <double>d * log(<double>d / E_d)

    return 2.0 * g2


# ===========================================================================
# Public batch functions
# ===========================================================================

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


def batch_chi2(np.int64_t[::1] a_arr, np.int64_t[::1] b_arr,
               np.int64_t[::1] c_arr, np.int64_t[::1] d_arr,
               bint correction=False):
    """
    Vectorised Pearson's chi-squared test for N independent 2x2 tables.

    Parameters
    ----------
    a_arr, b_arr, c_arr, d_arr : 1-D int64 arrays (contiguous)
        Cell values for each table: [[a, b], [c, d]].
    correction : bool
        If True, apply Yates' continuity correction. Default False.

    Returns
    -------
    (statistic, p_value) : tuple of numpy.ndarray (float64, length N)
        Chi-squared statistics and corresponding p-values.
        Tables with zero expected values produce NaN for both.
    """
    cdef Py_ssize_t n = a_arr.shape[0]
    if b_arr.shape[0] != n or c_arr.shape[0] != n or d_arr.shape[0] != n:
        raise ValueError("All input arrays must have the same length")

    cdef Py_ssize_t i
    cdef int corr_flag = 1 if correction else 0
    cdef double chi2_val

    stats_np = np.empty(n, dtype=np.float64)
    pvals_np = np.empty(n, dtype=np.float64)
    cdef double[::1] stats = stats_np
    cdef double[::1] pvals = pvals_np

    with nogil:
        for i in range(n):
            chi2_val = _chi2_stat(a_arr[i], b_arr[i], c_arr[i], d_arr[i],
                                  corr_flag)
            stats[i] = chi2_val
            if chi2_val != chi2_val:  # NaN check
                pvals[i] = NAN_VAL
            else:
                pvals[i] = chdtrc(1.0, chi2_val)

    return stats_np, pvals_np


def batch_log_likelihood(np.int64_t[::1] a_arr, np.int64_t[::1] b_arr,
                         np.int64_t[::1] c_arr, np.int64_t[::1] d_arr):
    """
    Vectorised log-likelihood ratio (G2) test for N independent 2x2 tables.

    Parameters
    ----------
    a_arr, b_arr, c_arr, d_arr : 1-D int64 arrays (contiguous)
        Cell values for each table: [[a, b], [c, d]].

    Returns
    -------
    (statistic, p_value) : tuple of numpy.ndarray (float64, length N)
        G2 statistics and corresponding p-values (chi-squared df=1).
        Tables with N=0 produce NaN for both.
    """
    cdef Py_ssize_t n = a_arr.shape[0]
    if b_arr.shape[0] != n or c_arr.shape[0] != n or d_arr.shape[0] != n:
        raise ValueError("All input arrays must have the same length")

    cdef Py_ssize_t i
    cdef double g2_val

    stats_np = np.empty(n, dtype=np.float64)
    pvals_np = np.empty(n, dtype=np.float64)
    cdef double[::1] stats = stats_np
    cdef double[::1] pvals = pvals_np

    with nogil:
        for i in range(n):
            g2_val = _g2_stat(a_arr[i], b_arr[i], c_arr[i], d_arr[i])
            stats[i] = g2_val
            if g2_val != g2_val:  # NaN check
                pvals[i] = NAN_VAL
            else:
                pvals[i] = chdtrc(1.0, g2_val)

    return stats_np, pvals_np
