"""
Fast GloVe epoch update kernel.

Implements weighted least-squares updates with AdaGrad:
  J = 0.5 * f(X_ij) * (w_i^T w_j_tilde + b_i + b_j_tilde - log(X_ij))^2
"""

# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt

ctypedef np.float32_t REAL_t
ctypedef np.int32_t ITYPE_t


def train_epoch_glove(
    np.ndarray[REAL_t, ndim=2] W_input,
    np.ndarray[REAL_t, ndim=2] W_context,
    np.ndarray[REAL_t, ndim=1] b_input,
    np.ndarray[REAL_t, ndim=1] b_context,
    np.ndarray[REAL_t, ndim=2] grad_sq_input,
    np.ndarray[REAL_t, ndim=2] grad_sq_context,
    np.ndarray[REAL_t, ndim=1] grad_sq_b_input,
    np.ndarray[REAL_t, ndim=1] grad_sq_b_context,
    np.ndarray[ITYPE_t, ndim=1] row_idx,
    np.ndarray[ITYPE_t, ndim=1] col_idx,
    np.ndarray[REAL_t, ndim=1] values,
    float learning_rate,
    float x_max,
    float power,
    int random_seed,
    bint shuffle,
):
    """
    Run one full GloVe epoch over sparse co-occurrence entries.

    Args:
        W_input/W_context: trainable embedding matrices.
        b_input/b_context: bias vectors.
        grad_sq_*: AdaGrad accumulators.
        row_idx/col_idx/values: sparse COO-like triplets (X_ij values > 0).
        learning_rate: base step size.
        x_max/power: weighting function parameters.
        random_seed: deterministic shuffle seed.
        shuffle: whether to shuffle pair order each epoch.

    Returns:
        (epoch_loss, pair_count)
    """
    cdef:
        Py_ssize_t n_pairs = row_idx.shape[0]
        int dims = W_input.shape[1]
        np.ndarray[ITYPE_t, ndim=1] order
        REAL_t[:] b_i_view = b_input
        REAL_t[:] b_j_view = b_context
        REAL_t[:] gsb_i_view = grad_sq_b_input
        REAL_t[:] gsb_j_view = grad_sq_b_context
        REAL_t[:, :] wi_view = W_input
        REAL_t[:, :] wj_view = W_context
        REAL_t[:, :] gsi_view = grad_sq_input
        REAL_t[:, :] gsj_view = grad_sq_context
        ITYPE_t[:] row_view = row_idx
        ITYPE_t[:] col_view = col_idx
        REAL_t[:] val_view = values
        Py_ssize_t p
        int d
        ITYPE_t ord_idx
        ITYPE_t i
        ITYPE_t j
        REAL_t x
        double weight
        double dot
        double diff
        double weighted_diff
        double grad_i
        double grad_j
        double grad_bi
        double grad_bj
        double epoch_loss = 0.0
        double eps = 1e-8
        REAL_t old_wi

    if n_pairs == 0:
        return 0.0, 0

    order = np.arange(n_pairs, dtype=np.int32)
    if shuffle and n_pairs > 1:
        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(order)
    cdef ITYPE_t[:] order_view = order

    for p in range(n_pairs):
        ord_idx = order_view[p]
        i = row_view[ord_idx]
        j = col_view[ord_idx]
        x = val_view[ord_idx]
        if x <= 0:
            continue

        if x < x_max:
            weight = (x / x_max) ** power
        else:
            weight = 1.0

        dot = 0.0
        for d in range(dims):
            dot += wi_view[i, d] * wj_view[j, d]

        diff = dot + b_i_view[i] + b_j_view[j] - log(x)
        weighted_diff = weight * diff
        epoch_loss += 0.5 * weight * diff * diff

        for d in range(dims):
            old_wi = wi_view[i, d]
            grad_i = weighted_diff * wj_view[j, d]
            grad_j = weighted_diff * old_wi

            gsi_view[i, d] += grad_i * grad_i
            gsj_view[j, d] += grad_j * grad_j

            wi_view[i, d] -= <REAL_t>(learning_rate * grad_i / sqrt(gsi_view[i, d] + eps))
            wj_view[j, d] -= <REAL_t>(learning_rate * grad_j / sqrt(gsj_view[j, d] + eps))

        grad_bi = weighted_diff
        grad_bj = weighted_diff
        gsb_i_view[i] += grad_bi * grad_bi
        gsb_j_view[j] += grad_bj * grad_bj
        b_i_view[i] -= <REAL_t>(learning_rate * grad_bi / sqrt(gsb_i_view[i] + eps))
        b_j_view[j] -= <REAL_t>(learning_rate * grad_bj / sqrt(gsb_j_view[j] + eps))

    return float(epoch_loss), int(n_pairs)
