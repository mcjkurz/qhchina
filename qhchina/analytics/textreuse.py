"""
Text reuse detection across a collection of tokenized documents.

Uses a seed-and-extend algorithm (similar to BLAST in bioinformatics):
1. Build vocabulary of n-grams and map to integer IDs
2. Fingerprint each document with n-gram hashes at each position
3. Build an inverted index to find candidate matching positions
4. Merge nearby seeds and verify with banded edit distance

Each document is a ``list[str]`` of tokens.  For character-level analysis
of raw strings, convert with ``[list(text) for text in raw_texts]``.

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


def _validate_documents(documents):
    """Validate that *documents* is a ``list[list[str]]`` and return it."""
    if not isinstance(documents, list):
        raise TypeError(
            f"documents must be a list[list[str]], got {type(documents).__name__}"
        )
    for i, doc in enumerate(documents):
        if not isinstance(doc, list):
            raise TypeError(
                f"Each document must be a list[str], "
                f"but document {i} is {type(doc).__name__}"
            )
    return documents


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
    documents: list[list[str]],
    *,
    n: int = 5,
    min_length: int = 10,
    min_similarity: float = 0.8,
    within_documents: bool = False,
    as_dataframe: bool = True,
    _max_gap: int | None = None,
    _max_distance: int | None = None,
) -> pd.DataFrame | list[dict]:
    """
    Find shared sequences (text reuse) across a collection of documents.

    Uses a seed-and-extend algorithm: finds exact n-gram matches as seeds,
    merges nearby seeds into passage candidates, then verifies each with
    banded edit distance. This allows fuzzy matching (insertions, deletions,
    substitutions) while remaining fast at scale.

    Each document is a ``list[str]`` of tokens.  For character-level
    analysis of raw strings, convert with
    ``[list(text) for text in raw_texts]``.

    Args:
        documents: List of tokenized documents (``list[list[str]]``).
        n (int): N-gram size for seeding. Smaller values find more matches
            but are slower. Default 5.
        min_length (int): Minimum passage length (in tokens) to report.
            Default 10.
        min_similarity (float): Minimum similarity score (0-1) for a passage
            to be reported. Computed as ``1 - distance / max_length``.
            Default 0.8.
        within_documents (bool): If True, also detect repeated passages
            within a single document. If False (default), only compare
            distinct document pairs.
        as_dataframe (bool): If True, return a pandas DataFrame. Default True.
        _max_gap (int | None): Maximum gap (in token positions)
            between consecutive seeds to still merge them into one passage.
            Defaults to ``n + 1``.  Why: a single substitution destroys
            ``n`` consecutive n-gram seeds, producing a gap of exactly
            ``n``; the default ``n + 1`` bridges that gap.  Increase to
            tolerate longer insertions/deletions between seed clusters.
        _max_distance (int | None): Maximum edit distance
            allowed during passage verification.  Also controls the band
            width of the banded Levenshtein computation, so it affects
            both accuracy and speed.  Defaults to
            ``int((1 - min_similarity) * min_length)`` — i.e. the number
            of edits that would reduce a minimal-length passage to exactly
            the similarity threshold.  Increase if you expect long passages
            with many local edits that still meet ``min_similarity``
            overall.

    Returns:
        pd.DataFrame or list[dict] with columns/keys:

            - **doc_a** (int): Index of the first document.
            - **doc_b** (int): Index of the second document (may equal
              doc_a when ``within_documents=True``).
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
        >>> docs = [list("天地玄黄宇宙洪荒"), list("天地玄黄宇宙日月盈昃")]
        >>> find_shared_sequences(docs, n=3, min_length=5)

        Pre-tokenized documents:

        >>> docs = [["天地", "玄黄", "宇宙", "洪荒", "日月", "盈昃"],
        ...         ["天地", "玄黄", "宇宙", "洪荒", "寒来", "暑往"]]
        >>> find_shared_sequences(docs, n=2, min_length=3)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if not 0 < min_similarity <= 1.0:
        raise ValueError(f"min_similarity must be in (0, 1], got {min_similarity}")

    max_gap = _max_gap if _max_gap is not None else n + 1
    max_distance = _max_distance if _max_distance is not None else max(1, int((1.0 - min_similarity) * min_length))

    docs = _validate_documents(documents)

    if not docs:
        return pd.DataFrame(columns=_RESULT_COLUMNS) if as_dataframe else []

    token_vocab = _build_token_vocab(docs)
    encoded = _encode_docs(docs, token_vocab)

    ngram_ids, doc_ids, positions = fingerprint_documents(encoded, n)

    if len(ngram_ids) == 0:
        return pd.DataFrame(columns=_RESULT_COLUMNS) if as_dataframe else []

    candidate_pairs = find_candidate_pairs(
        ngram_ids, doc_ids, positions, within_documents
    )

    results = []
    for (da, db), (sp_a, sp_b) in candidate_pairs.items():
        passages = merge_and_verify(
            sp_a, sp_b, encoded[da], encoded[db], n, max_gap, max_distance
        )

        for p in passages:
            length = max(p['len_a'], p['len_b'])
            if length < min_length:
                continue

            similarity = 1.0 - p['distance'] / length if length > 0 else 0.0
            if similarity < min_similarity:
                continue

            pa_start = p['pos_a']
            pb_start = p['pos_b']

            results.append({
                'doc_a': da,
                'doc_b': db,
                'pos_a': pa_start,
                'pos_b': pb_start,
                'length': length,
                'similarity': round(similarity, 4),
                'passage_a': ''.join(docs[da][pa_start:pa_start + p['len_a']]),
                'passage_b': ''.join(docs[db][pb_start:pb_start + p['len_b']]),
            })

    results.sort(key=lambda r: (-r['similarity'], -r['length']))

    if as_dataframe:
        return pd.DataFrame(results, columns=_RESULT_COLUMNS) if results else pd.DataFrame(columns=_RESULT_COLUMNS)
    return results
