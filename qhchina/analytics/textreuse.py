"""
Text reuse and near-duplicate detection for corpus analysis.

Uses a seed-and-extend algorithm (similar to BLAST in bioinformatics):
1. Build vocabulary of n-grams and map to integer IDs
2. Fingerprint each document with n-gram hashes at each position
3. Build an inverted index to find candidate matching positions
4. Merge nearby seeds and verify with banded edit distance

Corpus input is flexible: a single raw string, a list of raw strings,
or a list of pre-tokenized documents (list[list[str]]). For raw strings,
each character is treated as a token.

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


def _normalize_corpus(corpus):
    """
    Normalize corpus input to ``list[list[str]]``.

    Accepted formats:

    - ``str`` -- single raw document (each character becomes a token).
    - ``list[str]`` -- multiple raw documents (each character becomes a token).
    - ``list[list[str]]`` -- multiple pre-tokenized documents (used as-is).

    A single pre-tokenized document must be wrapped explicitly:
    ``[["tok1", "tok2", ...]]``.
    """
    if isinstance(corpus, str):
        return [list(corpus)]

    docs = []
    for item in corpus:
        if isinstance(item, str):
            docs.append(list(item))
        elif isinstance(item, list):
            docs.append(item)
        else:
            raise TypeError(
                f"Each corpus element must be a str or list[str], "
                f"got {type(item).__name__}"
            )
    return docs


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
    corpus_a: str | list[str] | list[list[str]],
    corpus_b: str | list[str] | list[list[str]] | None = None,
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

    Each corpus argument accepts three formats:

    - ``str`` -- a single raw document (each character becomes a token).
    - ``list[str]`` -- multiple raw documents (each character becomes a token).
    - ``list[list[str]]`` -- multiple pre-tokenized documents.

    To pass a single pre-tokenized document, wrap it in an outer list:
    ``[["tok1", "tok2", ...]]``.

    Args:
        corpus_a: First corpus. A raw string, a list of raw strings
            (``list[str]``), or a list of tokenized documents
            (``list[list[str]]``).
        corpus_b: Second corpus (same formats as *corpus_a*). If None,
            finds shared sequences within corpus_a (all-pairs comparison).
        n (int): N-gram size for seeding. Smaller values find more matches
            but are slower. Default 5.
        min_length (int): Minimum passage length (in tokens/characters) to
            report. Default 10.
        max_gap (int | None): Maximum gap between consecutive seeds to merge
            into one passage candidate. If None, defaults to ``n + 1``, which
            tolerates a single-character substitution (one substitution
            destroys ``n`` consecutive seeds, creating a gap of ``n``).
            Default None.
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
        Compare two corpora of raw strings:

        >>> from qhchina.analytics import find_shared_sequences
        >>> texts_a = ["天地玄黄宇宙洪荒", "日月盈昃辰宿列张"]
        >>> texts_b = ["天地玄黄宇宙洪荒日月", "寒来暑往秋收冬藏"]
        >>> find_shared_sequences(texts_a, texts_b, n=3, min_length=5)

        Compare two single strings directly:

        >>> find_shared_sequences("天地玄黄宇宙洪荒", "天地玄黄宇宙日月", n=3, min_length=5)
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

    docs_a = _normalize_corpus(corpus_a)
    cross_corpus = corpus_b is not None
    if cross_corpus:
        docs_b = _normalize_corpus(corpus_b)
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
