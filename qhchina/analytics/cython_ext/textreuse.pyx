# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True
"""
Cython-accelerated text reuse detection via seed-and-extend.

Provides fast n-gram fingerprinting and seed merging with banded
edit distance verification, all with the GIL released for maximum
throughput on large corpora.
"""
import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t, int64_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset


def fingerprint_documents(list encoded_docs, int n):
    """
    Compute n-gram fingerprints for a list of integer-encoded documents.

    For each document (a 1-D int32 array of token IDs), produces a
    polynomial hash of the n-gram at each position.

    Parameters
    ----------
    encoded_docs : list of numpy.ndarray (int32, contiguous)
        Each element is a 1-D array of token IDs for one document.
    n : int
        N-gram size for fingerprinting.

    Returns
    -------
    tuple of (ngram_ids, doc_ids, positions)
        ngram_ids : int64 array — hash of the n-gram at each position
        doc_ids : int32 array — which document each fingerprint belongs to
        positions : int32 array — token position within the document
    """
    cdef Py_ssize_t num_docs = len(encoded_docs)
    cdef Py_ssize_t total_ngrams = 0
    cdef Py_ssize_t d, i, doc_len
    cdef uint64_t h
    cdef int32_t[::1] doc_view
    cdef uint64_t base = 31

    for d in range(num_docs):
        doc_len = len(encoded_docs[d])
        if doc_len >= n:
            total_ngrams += doc_len - n + 1

    ngram_ids_np = np.empty(total_ngrams, dtype=np.int64)
    doc_ids_np = np.empty(total_ngrams, dtype=np.int32)
    positions_np = np.empty(total_ngrams, dtype=np.int32)

    cdef int64_t[::1] ngram_ids = ngram_ids_np
    cdef int32_t[::1] doc_ids = doc_ids_np
    cdef int32_t[::1] positions = positions_np

    # Precompute base^n for the rolling hash removal step
    cdef uint64_t base_n = 1
    cdef Py_ssize_t k
    for k in range(n):
        base_n = base_n * base

    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t j

    for d in range(num_docs):
        doc_arr = encoded_docs[d]
        doc_view = doc_arr
        doc_len = doc_view.shape[0]

        if doc_len < n:
            continue

        # Compute hash for the first n-gram from scratch
        h = 0
        for j in range(n):
            h = h * base + <uint64_t>doc_view[j]
        ngram_ids[idx] = <int64_t>h
        doc_ids[idx] = <int32_t>d
        positions[idx] = 0
        idx += 1

        # Rabin-Karp rolling hash: O(1) per subsequent position
        for i in range(1, doc_len - n + 1):
            h = h * base - <uint64_t>doc_view[i - 1] * base_n + <uint64_t>doc_view[i + n - 1]
            ngram_ids[idx] = <int64_t>h
            doc_ids[idx] = <int32_t>d
            positions[idx] = <int32_t>i
            idx += 1

    return ngram_ids_np, doc_ids_np, positions_np


def find_candidate_pairs(int64_t[::1] ngram_ids, int32_t[::1] doc_ids,
                         int32_t[::1] positions, int n_docs_a,
                         bint cross_corpus):
    """
    Build candidate seed-match pairs from fingerprint arrays using sorting.

    Replaces the Python inverted-index + candidate-pair extraction with a
    single sort-based pass over typed arrays.  For each group of fingerprints
    that share the same n-gram hash, emits all valid (doc_a, doc_b, pos_a,
    pos_b) seed pairs.

    Parameters
    ----------
    ngram_ids : int64 array
        N-gram hashes (from ``fingerprint_documents``).
    doc_ids : int32 array
        Document index for each fingerprint.
    positions : int32 array
        Token position within the document for each fingerprint.
    n_docs_a : int
        Number of documents in corpus A.  In cross-corpus mode, documents
        with ``doc_id < n_docs_a`` belong to A and the rest to B.
    cross_corpus : bool
        If True, only pair documents across corpora (A vs B).
        If False, pair all distinct documents within one corpus.

    Returns
    -------
    dict of (int, int) -> (ndarray, ndarray)
        Maps ``(doc_a, doc_b)`` to a pair of int32 arrays ``(pos_a, pos_b)``
        containing the seed positions, sorted by ``(pos_a, pos_b)``.
    """
    cdef Py_ssize_t N = ngram_ids.shape[0]
    if N == 0:
        return {}

    # Argsort by ngram_id to group identical hashes together
    order_np = np.argsort(np.asarray(ngram_ids))
    cdef int64_t[::1] order = order_np.astype(np.int64)

    # We'll collect seeds into a Python dict: (da, db) -> list of (pa, pb).
    # The inner loop is pure C-typed arithmetic; only the dict insert touches
    # Python objects, which is unavoidable but much cheaper than the old
    # pure-Python version that also did per-element int() conversions.
    candidates = {}

    cdef Py_ssize_t grp_start = 0
    cdef Py_ssize_t grp_end, gi, gj, n_seeds
    cdef int64_t cur_hash
    cdef int32_t da, db, pa, pb, key_a, key_b
    cdef int32_t[::1] va, vb

    while grp_start < N:
        cur_hash = ngram_ids[order[grp_start]]
        grp_end = grp_start + 1
        while grp_end < N and ngram_ids[order[grp_end]] == cur_hash:
            grp_end += 1

        if grp_end - grp_start < 2:
            grp_start = grp_end
            continue

        if cross_corpus:
            for gi in range(grp_start, grp_end):
                da = doc_ids[order[gi]]
                if da >= n_docs_a:
                    continue
                pa = positions[order[gi]]
                for gj in range(grp_start, grp_end):
                    db = doc_ids[order[gj]]
                    if db < n_docs_a:
                        continue
                    pb = positions[order[gj]]
                    key_b = db - <int32_t>n_docs_a
                    key = (da, key_b)
                    try:
                        (<list>candidates[key]).append((pa, pb))
                    except KeyError:
                        candidates[key] = [(pa, pb)]
        else:
            for gi in range(grp_start, grp_end):
                da = doc_ids[order[gi]]
                pa = positions[order[gi]]
                for gj in range(gi + 1, grp_end):
                    db = doc_ids[order[gj]]
                    if da == db:
                        continue
                    pb = positions[order[gj]]
                    if da < db:
                        key_a = da
                        key_b = db
                        key = (key_a, key_b)
                        try:
                            (<list>candidates[key]).append((pa, pb))
                        except KeyError:
                            candidates[key] = [(pa, pb)]
                    else:
                        key_a = db
                        key_b = da
                        key = (key_a, key_b)
                        try:
                            (<list>candidates[key]).append((pb, pa))
                        except KeyError:
                            candidates[key] = [(pb, pa)]

        grp_start = grp_end

    # Convert list-of-tuples to sorted parallel int32 arrays per pair
    result = {}
    for pair_key, seed_list in candidates.items():
        seed_list.sort()
        n_seeds = len(seed_list)
        arr_a = np.empty(n_seeds, dtype=np.int32)
        arr_b = np.empty(n_seeds, dtype=np.int32)
        va = arr_a
        vb = arr_b
        for gi in range(n_seeds):
            va[gi] = seed_list[gi][0]
            vb[gi] = seed_list[gi][1]
        result[pair_key] = (arr_a, arr_b)

    return result


cdef inline int _min_int(int a, int b) noexcept nogil:
    return a if a < b else b


def merge_and_verify(int32_t[::1] seeds_pos_a, int32_t[::1] seeds_pos_b,
                     int32_t[::1] tokens_a, int32_t[::1] tokens_b,
                     int n, int max_gap, int max_distance):
    """
    Merge nearby seed matches and verify with banded edit distance.

    Given parallel arrays of matching seed positions in documents A and B,
    merge seeds that are close together into passage candidates, then
    verify each candidate using banded Levenshtein distance.

    Parameters
    ----------
    seeds_pos_a, seeds_pos_b : 1-D int32 arrays (contiguous)
        Parallel arrays of matching positions in doc A and doc B,
        sorted by seeds_pos_a.
    tokens_a, tokens_b : 1-D int32 arrays (contiguous)
        Full token ID sequences for documents A and B.
    n : int
        N-gram size used for seeding. Each seed covers positions
        [pos, pos+n), so passage length is (end - start + n).
    max_gap : int
        Maximum gap between consecutive seeds to merge into one passage.
    max_distance : int
        Maximum edit distance (band width) for verification.

    Returns
    -------
    list of dict
        Each dict has keys: 'pos_a', 'pos_b', 'len_a', 'len_b', 'distance'.
        Only passages with distance <= max_distance are returned.
    """
    cdef Py_ssize_t n_seeds = seeds_pos_a.shape[0]
    if n_seeds == 0:
        return []

    # Merge seeds into passage candidates
    cdef list passages = []
    cdef int32_t start_a = seeds_pos_a[0]
    cdef int32_t start_b = seeds_pos_b[0]
    cdef int32_t end_a = seeds_pos_a[0]
    cdef int32_t end_b = seeds_pos_b[0]
    cdef Py_ssize_t i
    cdef int32_t cur_a, cur_b

    for i in range(1, n_seeds):
        cur_a = seeds_pos_a[i]
        cur_b = seeds_pos_b[i]

        if (cur_a - end_a) <= max_gap and (cur_b - end_b) <= max_gap:
            end_a = cur_a
            end_b = cur_b
        else:
            passages.append((start_a, start_b, end_a, end_b))
            start_a = cur_a
            start_b = cur_b
            end_a = cur_a
            end_b = cur_b

    passages.append((start_a, start_b, end_a, end_b))

    # Verify each passage with banded edit distance.
    # The passage spans exactly [start, end + n) — the region covered by
    # the merged seeds.  No extension beyond that: additional context would
    # pull in unrelated tokens that inflate the edit distance and cause
    # valid matches to be rejected.
    cdef list results = []
    cdef int32_t pa, pb, ea, eb
    cdef Py_ssize_t len_a_val, len_b_val
    cdef int dist

    for (pa, pb, ea, eb) in passages:
        len_a_val = <Py_ssize_t>(ea - pa + n)
        len_b_val = <Py_ssize_t>(eb - pb + n)

        if pa + len_a_val > tokens_a.shape[0]:
            len_a_val = tokens_a.shape[0] - pa
        if pb + len_b_val > tokens_b.shape[0]:
            len_b_val = tokens_b.shape[0] - pb

        if len_a_val <= 0 or len_b_val <= 0:
            continue

        dist = _banded_edit_distance(tokens_a, pa, len_a_val,
                                     tokens_b, pb, len_b_val,
                                     max_distance)

        if dist <= max_distance:
            results.append({
                'pos_a': int(pa),
                'pos_b': int(pb),
                'len_a': int(len_a_val),
                'len_b': int(len_b_val),
                'distance': dist,
            })

    return results


cdef int _banded_edit_distance(int32_t[::1] seq_a, Py_ssize_t start_a, Py_ssize_t len_a,
                               int32_t[::1] seq_b, Py_ssize_t start_b, Py_ssize_t len_b,
                               int band) nogil:
    """
    Banded Levenshtein distance between two subsequences.

    Only computes cells within `band` diagonals of the main diagonal,
    giving O(n * band) time instead of O(n * m).
    Returns band+1 if the true distance exceeds the band.
    """
    cdef Py_ssize_t i, j, j_start, j_end
    cdef int sub_cost, ins_cost, del_cost, best
    cdef int* prev_row
    cdef int* curr_row
    cdef Py_ssize_t alloc_size = len_b + 1

    prev_row = <int*>malloc(alloc_size * sizeof(int))
    curr_row = <int*>malloc(alloc_size * sizeof(int))

    if prev_row == NULL or curr_row == NULL:
        if prev_row != NULL:
            free(prev_row)
        if curr_row != NULL:
            free(curr_row)
        return band + 1

    for j in range(alloc_size):
        prev_row[j] = band + 1

    prev_row[0] = 0
    for j in range(1, _min_int(band + 1, <int>alloc_size)):
        prev_row[j] = j

    for i in range(1, len_a + 1):
        j_start = 1 if (i - band) < 1 else (i - band)
        j_end = len_b if (i + band) > len_b else (i + band)

        for j in range(alloc_size):
            curr_row[j] = band + 1

        if j_start == 1:
            curr_row[0] = i if i <= band else band + 1

        for j in range(j_start, j_end + 1):
            if seq_a[start_a + i - 1] == seq_b[start_b + j - 1]:
                sub_cost = prev_row[j - 1]
            else:
                sub_cost = prev_row[j - 1] + 1

            del_cost = prev_row[j] + 1
            ins_cost = curr_row[j - 1] + 1

            best = sub_cost
            if del_cost < best:
                best = del_cost
            if ins_cost < best:
                best = ins_cost
            curr_row[j] = best

        # Swap rows
        prev_row, curr_row = curr_row, prev_row

    best = prev_row[len_b]

    free(prev_row)
    free(curr_row)

    return best
