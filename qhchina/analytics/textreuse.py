"""
Text reuse and near-duplicate detection for corpus analysis.

Uses a seed-and-extend algorithm (similar to BLAST in bioinformatics):
1. Build vocabulary of n-grams and map to integer IDs
2. Fingerprint each document with n-gram hashes at each position
3. Build an inverted index to find candidate matching positions
4. Merge nearby seeds and verify with banded edit distance

Each corpus is a ``list[list[str]]`` — a list of tokenized documents.
For character-level analysis of raw strings, convert with
``[list(text) for text in raw_texts]``.

Requires Cython extensions to be compiled (see setup.py).
"""

import logging

import numpy as np
import pandas as pd

from .cython_ext.textreuse import fingerprint_documents, find_candidate_pairs, merge_and_verify

logger = logging.getLogger("qhchina.analytics.textreuse")


__all__ = [
    'find_shared_sequences',
]

_RESULT_COLUMNS = ['doc_a', 'doc_b', 'pos_a', 'pos_b', 'length', 'similarity', 'passage_a', 'passage_b']


def _validate_corpus(corpus):
    """Validate that *corpus* is a ``list[list[str]]`` and return it."""
    if not isinstance(corpus, list):
        raise TypeError(
            f"corpus must be a list[list[str]], got {type(corpus).__name__}"
        )
    for i, doc in enumerate(corpus):
        if not isinstance(doc, list):
            raise TypeError(
                f"Each document must be a list[str], "
                f"but document {i} is {type(doc).__name__}"
            )
    return corpus


def _build_token_vocab(docs):
    """Build vocabulary mapping tokens to integer IDs."""
    vocab = {}
    idx = 0
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def _encode_docs(docs, token_vocab):
    """Encode documents as int32 arrays using token vocabulary."""
    encoded = []
    for doc in docs:
        arr = np.array([token_vocab[t] for t in doc], dtype=np.int32)
        encoded.append(arr)
    return encoded


def find_shared_sequences(
    corpus_a: list[list[str]],
    corpus_b: list[list[str]] | None = None,
    n: int = 5,
    min_length: int = 10,
    max_gap: int | None = None,
    min_similarity: float = 0.8,
    max_distance: int | None = None,
    as_dataframe: bool = True,
) -> pd.DataFrame | list[dict]:
    """
    Find shared sequences (text reuse) between two corpora or within one.

    Uses a seed-and-extend algorithm: finds exact n-gram matches as seeds,
    merges nearby seeds into passage candidates, then verifies each with
    banded edit distance. This allows fuzzy matching (insertions, deletions,
    substitutions) while remaining fast at scale.

    Each corpus is a ``list[list[str]]`` — a list of tokenized documents.
    For character-level analysis of raw strings, convert with
    ``[list(text) for text in raw_texts]``.

    Args:
        corpus_a: First corpus as a list of tokenized documents
            (``list[list[str]]``).
        corpus_b: Second corpus (same format as *corpus_a*). If None,
            finds shared sequences within corpus_a (all-pairs comparison).
        n (int): N-gram size for seeding. Smaller values find more matches
            but are slower. Default 5.
        min_length (int): Minimum passage length (in tokens) to report.
            Default 10.
        max_gap (int | None): Maximum gap between consecutive seeds to merge
            into one passage candidate. If None, defaults to ``n + 1``, which
            tolerates a single-token substitution (one substitution destroys
            ``n`` consecutive seeds, creating a gap of ``n``). Default None.
        min_similarity (float): Minimum similarity score (0-1) for a passage
            to be reported. Computed as ``1 - distance / max_length``.
            Default 0.8.
        max_distance (int | None): Maximum edit distance for verification.
            If None, derived from ``min_similarity`` and ``min_length``:
            ``int((1 - min_similarity) * min_length)``. Default None.
        as_dataframe (bool): If True, return a pandas DataFrame. Default True.

    Returns:
        pd.DataFrame or list[dict] with columns/keys:

            - **doc_a** (int): Document index in corpus_a.
            - **doc_b** (int): Document index in corpus_b (or corpus_a).
            - **pos_a** (int): Start position in document A.
            - **pos_b** (int): Start position in document B.
            - **length** (int): Length of the matched passage (maximum of
              len_a and len_b).
            - **similarity** (float): Similarity score (1 - distance/length).
            - **passage_a** (str): Matched text from document A.
            - **passage_b** (str): Matched text from document B.

    Examples:
        Character-level comparison of raw strings:

        >>> from qhchina.analytics import find_shared_sequences
        >>> corpus_a = [list("天地玄黄宇宙洪荒"), list("日月盈昃辰宿列张")]
        >>> corpus_b = [list("天地玄黄宇宙洪荒日月"), list("寒来暑往秋收冬藏")]
        >>> find_shared_sequences(corpus_a, corpus_b, n=3, min_length=5)

        Pre-tokenized documents:

        >>> doc_a = [["天地", "玄黄", "宇宙", "洪荒", "日月", "盈昃"]]
        >>> doc_b = [["天地", "玄黄", "宇宙", "洪荒", "寒来", "暑往"]]
        >>> find_shared_sequences(doc_a, doc_b, n=2, min_length=3)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if not 0 < min_similarity <= 1.0:
        raise ValueError(f"min_similarity must be in (0, 1], got {min_similarity}")

    if max_gap is None:
        max_gap = n + 1

    if max_distance is None:
        max_distance = max(1, int((1.0 - min_similarity) * min_length))

    docs_a = _validate_corpus(corpus_a)
    cross_corpus = corpus_b is not None
    if cross_corpus:
        docs_b = _validate_corpus(corpus_b)
    else:
        docs_b = []

    all_docs = docs_a + docs_b
    n_docs_a = len(docs_a)

    if not all_docs:
        return pd.DataFrame(columns=_RESULT_COLUMNS) if as_dataframe else []

    token_vocab = _build_token_vocab(all_docs)
    encoded = _encode_docs(all_docs, token_vocab)

    ngram_ids, doc_ids, positions = fingerprint_documents(encoded, n)

    if len(ngram_ids) == 0:
        return pd.DataFrame(columns=_RESULT_COLUMNS) if as_dataframe else []

    candidate_pairs = find_candidate_pairs(
        ngram_ids, doc_ids, positions, n_docs_a, cross_corpus
    )

    results = []
    for (da, db), (sp_a, sp_b) in candidate_pairs.items():
        if cross_corpus:
            doc_tokens_a = docs_a[da]
            doc_tokens_b = docs_b[db]
            enc_a = encoded[da]
            enc_b = encoded[n_docs_a + db]
        else:
            doc_tokens_a = all_docs[da]
            doc_tokens_b = all_docs[db]
            enc_a = encoded[da]
            enc_b = encoded[db]
        passages = merge_and_verify(sp_a, sp_b, enc_a, enc_b, n, max_gap, max_distance)

        for p in passages:
            length = max(p['len_a'], p['len_b'])
            if length < min_length:
                continue

            similarity = 1.0 - p['distance'] / length if length > 0 else 0.0
            if similarity < min_similarity:
                continue

            pa_start = p['pos_a']
            pb_start = p['pos_b']
            passage_a_tokens = doc_tokens_a[pa_start:pa_start + p['len_a']]
            passage_b_tokens = doc_tokens_b[pb_start:pb_start + p['len_b']]

            results.append({
                'doc_a': da,
                'doc_b': db,
                'pos_a': pa_start,
                'pos_b': pb_start,
                'length': length,
                'similarity': round(similarity, 4),
                'passage_a': ''.join(passage_a_tokens),
                'passage_b': ''.join(passage_b_tokens),
            })

    results.sort(key=lambda r: (-r['similarity'], -r['length']))

    if as_dataframe:
        return pd.DataFrame(results, columns=_RESULT_COLUMNS) if results else pd.DataFrame(columns=_RESULT_COLUMNS)
    return results
