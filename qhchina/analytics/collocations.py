import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact as scipy_fisher_exact

from ..utils import apply_p_value_correction, VALID_CORRECTIONS, iter_batches, build_vocab_from_iter

logger = logging.getLogger("qhchina.analytics.collocations")


__all__ = [
    'find_collocates',
    'cooc_matrix',
    'plot_collocates',
    'FilterOptions',
    'CoocMatrix',
]

try:
    from .cython_ext.collocations import (
        calculate_cooc_matrix_window,
        calculate_cooc_matrix_document,
        calculate_window_counts_batch,
        calculate_sentence_counts_batch,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    calculate_cooc_matrix_window = None
    calculate_cooc_matrix_document = None
    calculate_window_counts_batch = None
    calculate_sentence_counts_batch = None
    logger.warning("Cython extensions not available; using slower Python fallback.")


class CoocMatrix:
    """
    Co-occurrence matrix with flexible indexing by word or index.
    
    Supports flexible indexing:
    
    - ``matrix["word1", "word2"]`` - single count (int)
    - ``matrix[132, 5234]`` - single count (int)
    - ``matrix["word1"]`` - row as dict {word: count}
    - ``matrix["word1", :]`` - row as dict {word: count}
    - ``matrix[:, "word2"]`` - column as dict {word: count}
    
    Internally stores data as a scipy sparse CSR matrix for memory efficiency.
    
    Attributes:
        vocab (list[str]): List of vocabulary words in index order.
        word_to_index (dict[str, int]): Mapping from words to matrix indices.
        index_to_word (dict[int, str]): Mapping from matrix indices to words.
        shape (tuple[int, int]): Shape of the matrix (vocab_size, vocab_size).
        nnz (int): Number of non-zero entries.
    
    Example:
        >>> matrix = cooc_matrix(documents, horizon=5)
        >>> matrix["fox", "dog"]
        42
        >>> matrix["fox"]
        {'quick': 10, 'brown': 8, 'dog': 42, ...}
        >>> df = matrix.to_dataframe()
        >>> arr = matrix.to_dense()
    """
    
    def __init__(
        self, 
        matrix: sparse.csr_matrix, 
        vocab_list: list[str], 
        word_to_index: dict[str, int]
    ):
        """
        Initialize CoocMatrix.
        
        Args:
            matrix: scipy CSR sparse matrix containing co-occurrence counts.
            vocab_list: List of vocabulary words in index order.
            word_to_index: Dictionary mapping words to their matrix indices.
        """
        self._matrix = matrix
        self._vocab = vocab_list
        self._w2i = word_to_index
        self._i2w = {i: w for w, i in word_to_index.items()}
    
    def _resolve_index(self, key) -> int:
        """Convert a word string or int to a matrix index."""
        if isinstance(key, str):
            if key not in self._w2i:
                raise KeyError(f"Word '{key}' not in vocabulary")
            return self._w2i[key]
        elif isinstance(key, (int, np.integer)):
            if key < 0 or key >= len(self._vocab):
                raise IndexError(f"Index {key} out of range [0, {len(self._vocab)})")
            return int(key)
        else:
            raise TypeError(f"Index must be str or int, got {type(key).__name__}")
    
    def __getitem__(self, key):
        """
        Flexible indexing for co-occurrence lookups.
        
        Args:
            key: Can be:
                - (word1, word2) or (idx1, idx2): Returns single count
                - word or idx: Returns row as dict
                - (word, slice) or (slice, word): Returns row/column as dict
        
        Returns:
            int for single cell lookup, dict for row/column lookup.
        
        Examples:
            >>> matrix["fox", "dog"]      # Single count
            42
            >>> matrix["fox"]             # Row as dict
            {'quick': 10, 'brown': 8, ...}
            >>> matrix["fox", :]          # Same as above
            >>> matrix[:, "dog"]          # Column as dict
        """
        # Handle single key (row lookup)
        if not isinstance(key, tuple):
            row_idx = self._resolve_index(key)
            return self._row_to_dict(row_idx)
        
        # Handle tuple key
        if len(key) != 2:
            raise IndexError("Index must be a single key or a pair (row, col)")
        
        row_key, col_key = key
        
        # Handle slice cases
        if isinstance(row_key, slice) and row_key == slice(None):
            # [:, col] - column lookup
            col_idx = self._resolve_index(col_key)
            return self._col_to_dict(col_idx)
        
        if isinstance(col_key, slice) and col_key == slice(None):
            # [row, :] - row lookup
            row_idx = self._resolve_index(row_key)
            return self._row_to_dict(row_idx)
        
        # Both are concrete indices - single cell lookup
        row_idx = self._resolve_index(row_key)
        col_idx = self._resolve_index(col_key)
        return int(self._matrix[row_idx, col_idx])
    
    def _row_to_dict(self, row_idx: int) -> dict[str, int]:
        """Convert a matrix row to a dict of {word: count} for non-zero entries."""
        row = self._matrix.getrow(row_idx)
        return {self._i2w[col]: int(val) for col, val in zip(row.indices, row.data)}
    
    def _col_to_dict(self, col_idx: int) -> dict[str, int]:
        """Convert a matrix column to a dict of {word: count} for non-zero entries."""
        col = self._matrix.getcol(col_idx)
        return {self._i2w[row]: int(val) for row, val in zip(col.indices, col.data)}
    
    def get(self, row_key, col_key, default: int = 0) -> int:
        """
        Get a co-occurrence count with a default value for missing pairs.
        
        Args:
            row_key: Row word or index.
            col_key: Column word or index.
            default: Value to return if the pair is not found or out of vocabulary.
        
        Returns:
            Co-occurrence count, or default if not found.
        """
        try:
            return self[row_key, col_key]
        except (KeyError, IndexError):
            return default
    
    def to_dense(self) -> np.ndarray:
        """
        Convert to a dense NumPy array.
        
        Warning: This may use significant memory for large vocabularies.
        
        Returns:
            2D numpy array of shape (vocab_size, vocab_size).
        """
        return self._matrix.toarray()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with word labels.
        
        Warning: This may use significant memory for large vocabularies.
        
        Returns:
            DataFrame with vocabulary words as both index and columns.
        """
        return pd.DataFrame(
            self._matrix.toarray(),
            index=self._vocab,
            columns=self._vocab
        )
    
    @property
    def sparse(self) -> sparse.csr_matrix:
        """Access the underlying scipy sparse CSR matrix."""
        return self._matrix
    
    @property
    def vocab(self) -> list[str]:
        """List of vocabulary words in index order. Returns the internal list directly."""
        return self._vocab
    
    @property
    def word_to_index(self) -> dict[str, int]:
        """Dictionary mapping words to their matrix indices. Returns the internal dict directly."""
        return self._w2i
    
    @property
    def index_to_word(self) -> dict[int, str]:
        """Dictionary mapping matrix indices to words. Returns the internal dict directly."""
        return self._i2w
    
    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix (vocab_size, vocab_size)."""
        return self._matrix.shape
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries in the matrix."""
        return self._matrix.nnz
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)
    
    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word in self._w2i
    
    def __repr__(self) -> str:
        return (
            f"CoocMatrix(vocab_size={len(self._vocab)}, "
            f"nnz={self._matrix.nnz}, "
            f"density={self._matrix.nnz / (len(self._vocab)**2):.2%})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()

class FilterOptions(TypedDict, total=False):
    """Type definition for filter options in collocation analysis."""
    max_p: float
    max_adjusted_p: float
    stopwords: list[str]
    min_word_length: int
    min_exp_local: float
    max_exp_local: float
    min_obs_local: int
    max_obs_local: int
    min_ratio_local: float
    max_ratio_local: float
    min_obs_global: int
    max_obs_global: int


def _compute_collocation_result(target, candidate, a, b, c, d, alternative='greater'):
    """
    Compute collocation statistics for a single target-collocate pair.
    
    Shared by both window/sentence methods and Python/Cython backends.
    Builds a 2x2 contingency table and runs Fisher's exact test.
    
    Args:
        target: Target word.
        candidate: Collocate word.
        a: Co-occurrence count (target with collocate).
        b: Target without collocate.
        c: Collocate without target.
        d: Neither target nor collocate.
        alternative: Alternative hypothesis for Fisher's exact test.
    
    Returns:
        dict with keys: target, collocate, exp_local, obs_local,
        ratio_local, obs_global, p_value.
    """
    # N = sample size from contingency table (a + b + c + d)
    # For window method: N excludes positions where target is at center (per Evert)
    # For sentence method: N = total sentences
    N = a + b + c + d
    expected = (a + b) * (a + c) / N if N > 0 else 0
    ratio = a / expected if expected > 0 else 0
    
    table = [[a, b], [c, d]]
    _, p_value = scipy_fisher_exact(table, alternative=alternative)
    
    return {
        "target": target,
        "collocate": candidate,
        "exp_local": expected,
        "obs_local": int(a),
        "ratio_local": ratio,
        "obs_global": int(a + c), 
        "p_value": p_value,
    }


def _build_results_from_counts(target_words, target_counts, candidate_counts, global_counts, total, alternative='greater', method='window'):
    """
    Build collocation result dicts from Python-accumulated counts.
    
    Shared by both Python window and sentence backends.
    
    Args:
        target_words: Target words to process.
        target_counts: {target: context count containing target}.
        candidate_counts: {target: Counter of candidate co-occurrences}.
        global_counts: {token: global count}.
        total: Total tokens (window) or sentences (sentence method).
        alternative: Alternative hypothesis for Fisher's exact test.
        method: 'window' or 'sentence' (affects *d* cell calculation).
    
    Returns:
        list[dict]: Collocation statistics per target-collocate pair.
    """
    results = []
    
    for target in target_words:
        for candidate, a in candidate_counts[target].items():
            if candidate == target:
                continue
            # a = the number of positions occupied by the candidate where target is near
            # b = the number of positions occupied by the non-candidates where target is near
            # c = the number of positions occupied by the candidate where target is not near
            # d = the number of positions occupied by the non-candidates where target is not near)
            b = target_counts[target] - a
            c = global_counts[candidate] - a
            
            if method == 'window':
                # here, as per (Evert, 2008), we exclude positions where target is at center; 
                # we are only interested in positions AROUND the target and how non-target candidates
                # are distributed there
                d = (total - global_counts[target]) - (a + b + c)
            else:  # sentence
                d = total - a - b - c
            
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, alternative
            ))
    
    return results


def _calculate_collocations_window_cython(
    sentences, target_words, horizon=5, alternative='greater',
    batch_words=100_000, max_sentence_length=256,
):
    """
    Cython-accelerated window-based collocation counting (two-pass, streaming).
    
    Pass 1 builds vocabulary; pass 2 streams batches through Cython nogil
    counting and accumulates into pre-allocated numpy arrays.
    
    Args:
        sentences: Restartable iterable of tokenized sentences.
        target_words: Target words to find collocates for.
        horizon: int (symmetric) or tuple (left, right) window size.
        alternative: Alternative hypothesis for Fisher's exact test.
        batch_words: Target token count per batch.
        max_sentence_length: Truncate longer sentences. None disables.
    
    Returns:
        list[dict]: Collocation statistics per target-collocate pair.
    """
    # Normalize horizon
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        # Swap: user's "right of target" = algorithm's "left from candidate"
        left_horizon, right_horizon = horizon[1], horizon[0]
    
    # Pass 1: build vocabulary
    word_counts, _, _ = build_vocab_from_iter(sentences, max_sentence_length)
    word2idx = {w: i for i, w in enumerate(word_counts)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    
    # Resolve target indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if not target_words_filtered:
        return []
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    n_targets = len(target_indices)
    
    # Pre-allocate accumulators
    T_count_total = np.zeros(n_targets, dtype=np.int64)
    candidate_counts_total = np.zeros((n_targets, vocab_size), dtype=np.int64)
    token_counter_total = np.zeros(vocab_size, dtype=np.int64)
    total_tokens = 0
    
    # Pass 2: batch counting
    for batch in iter_batches(sentences, batch_words, max_sentence_length):
        batch_T, batch_cand, batch_tok, batch_total = calculate_window_counts_batch(
            batch, word2idx, target_indices, left_horizon, right_horizon, vocab_size
        )
        T_count_total += batch_T
        candidate_counts_total += batch_cand
        token_counter_total += batch_tok
        total_tokens += batch_total
    
    # Build results from accumulated counts
    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        nonzero = np.nonzero(candidate_counts_total[t_idx])[0]
        for candidate_idx in nonzero:
            if candidate_idx == target_word_idx:
                continue
            a = candidate_counts_total[t_idx, candidate_idx]
            candidate = idx2word[int(candidate_idx)]
            b = T_count_total[t_idx] - a
            c = token_counter_total[candidate_idx] - a
            d = (total_tokens - token_counter_total[target_word_idx]) - (a + b + c)
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, alternative
            ))
    
    return results


def _calculate_collocations_window_python(
    sentences, target_words, horizon=5, alternative='greater',
    batch_words=100_000, max_sentence_length=256,
):
    """
    Pure Python window-based collocation counting (streaming, single-pass).
    
    Streams batches via :func:`~qhchina.utils.iter_batches` and accumulates
    into sparse Python dicts. Fallback when Cython is not available.
    
    Args:
        sentences: Iterable of tokenized sentences.
        target_words: Target words to find collocates for.
        horizon: int (symmetric) or tuple (left, right) window size.
        alternative: Alternative hypothesis for Fisher's exact test.
        batch_words: Target token count per batch.
        max_sentence_length: Truncate longer sentences. None disables.
    
    Returns:
        list[dict]: Collocation statistics per target-collocate pair.
    """
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        left_horizon, right_horizon = horizon[1], horizon[0]
    
    total_tokens = 0
    target_set = set(target_words)
    T_count = {target: 0 for target in target_words}
    candidate_in_context = {target: Counter() for target in target_words}
    token_counter = Counter()

    for batch in iter_batches(sentences, batch_words, max_sentence_length):
        for sentence in batch:
            for i, token in enumerate(sentence):
                total_tokens += 1
                token_counter[token] += 1

                start = max(0, i - left_horizon)
                end = min(len(sentence), i + right_horizon + 1)

                seen_targets = set()
                for j in range(start, end):
                    if j != i:
                        word = sentence[j]
                        if word in target_set and word not in seen_targets:
                            seen_targets.add(word)
                            T_count[word] += 1
                            candidate_in_context[word][token] += 1

    return _build_results_from_counts(
        target_words, T_count, candidate_in_context, token_counter, total_tokens, alternative, method='window'
    )


def _calculate_collocations_sentence_cython(
    sentences, target_words, alternative='greater',
    batch_words=100_000, max_sentence_length=256,
):
    """
    Cython-accelerated sentence-based collocation counting (two-pass, streaming).
    
    Pass 1 builds vocabulary; pass 2 streams batches through Cython nogil
    counting and accumulates into pre-allocated numpy arrays.
    
    Args:
        sentences: Restartable iterable of tokenized sentences.
        target_words: Target words to find collocates for.
        alternative: Alternative hypothesis for Fisher's exact test.
        batch_words: Target token count per batch.
        max_sentence_length: Truncate longer sentences. None disables.
    
    Returns:
        list[dict]: Collocation statistics per target-collocate pair.
    """
    # Pass 1: build vocabulary
    word_counts, _, _ = build_vocab_from_iter(sentences, max_sentence_length)
    word2idx = {w: i for i, w in enumerate(word_counts)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    
    # Resolve target indices
    target_words_filtered = [w for w in target_words if w in word2idx]
    if not target_words_filtered:
        return []
    target_indices = np.array([word2idx[w] for w in target_words_filtered], dtype=np.int32)
    n_targets = len(target_indices)
    
    # Pre-allocate accumulators
    candidate_sentences_total = np.zeros((n_targets, vocab_size), dtype=np.int64)
    sentences_with_token_total = np.zeros(vocab_size, dtype=np.int64)
    total_sentences = 0
    
    # Pass 2: batch counting
    for batch in iter_batches(sentences, batch_words, max_sentence_length):
        batch_cand, batch_swt, batch_n = calculate_sentence_counts_batch(
            batch, word2idx, target_indices, vocab_size
        )
        candidate_sentences_total += batch_cand
        sentences_with_token_total += batch_swt
        total_sentences += batch_n
    
    # Build results from accumulated counts
    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        nonzero = np.nonzero(candidate_sentences_total[t_idx])[0]
        for candidate_idx in nonzero:
            if candidate_idx == target_word_idx:
                continue
            a = candidate_sentences_total[t_idx, candidate_idx]
            candidate = idx2word[int(candidate_idx)]
            b = sentences_with_token_total[target_word_idx] - a
            c = sentences_with_token_total[candidate_idx] - a
            d = total_sentences - a - b - c
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, alternative
            ))
    
    return results


def _calculate_collocations_sentence_python(
    sentences, target_words, alternative='greater',
    batch_words=100_000, max_sentence_length=256,
):
    """
    Pure Python sentence-based collocation counting (streaming, single-pass).
    
    Streams batches via :func:`~qhchina.utils.iter_batches` and accumulates
    into sparse Python dicts. Fallback when Cython is not available.
    
    Args:
        sentences: Iterable of tokenized sentences.
        target_words: Target words to find collocates for.
        alternative: Alternative hypothesis for Fisher's exact test.
        batch_words: Target token count per batch.
        max_sentence_length: Truncate longer sentences. None disables.
    
    Returns:
        list[dict]: Collocation statistics per target-collocate pair.
    """
    total_sentences = 0
    candidate_in_sentences = {target: Counter() for target in target_words}
    sentences_with_token = defaultdict(int)

    for batch in iter_batches(sentences, batch_words, max_sentence_length):
        for sentence in batch:
            total_sentences += 1
            unique_tokens = set(sentence)
            for token in unique_tokens:
                sentences_with_token[token] += 1
            for target in target_words:
                if target in unique_tokens:
                    candidate_in_sentences[target].update(unique_tokens)

    return _build_results_from_counts(
        target_words, sentences_with_token, candidate_in_sentences, sentences_with_token, total_sentences, alternative, method='sentence'
    )

def find_collocates(
    sentences: Iterable[list[str]], 
    target_words: str | list[str], 
    method: str = 'window', 
    horizon: int | tuple | None = None, 
    filters: FilterOptions | None = None, 
    correction: str | None = None,
    as_dataframe: bool = True,
    max_sentence_length: int | None = 256,
    alternative: str = 'greater',
    sort_by: str = 'obs_local',
    ascending: bool = False,
    batch_words: int = 100_000,
) -> list[dict] | pd.DataFrame:
    """
    Find collocates for target words in a corpus of sentences.
    
    Processes data in streaming batches to keep memory low even for very large
    corpora. The data is iterated twice (vocabulary building, then counting), 
    so a restartable iterable is required. Lists, file-backed iterators, and 
    restartable generator classes all work; single-use generators do not.
    
    Args:
        sentences (Iterable[list[str]]): Restartable iterable of tokenized
            sentences (each sentence a list of string tokens).
        target_words (str | list[str]): Target word(s) to find collocates for.
        method (str): Method to use for calculating collocations. Either 'window' or 
            'sentence'. 'window' uses a sliding window of specified horizon around each 
            token. 'sentence' considers whole sentences as context units (horizon not 
            applicable). Default is 'window'.
        horizon (int | tuple | None): Context window size relative to the target 
            word. Only applicable when method='window'. Must be None when method='sentence'.
            - int: Symmetric window (e.g., 5 means 5 words on each side of target)
            - tuple: Asymmetric window (left, right) specifying how many words to look
              on each side of the target word: (0, 5) finds collocates up to 5 words to 
              the RIGHT of target; (5, 0) finds collocates up to 5 words to the LEFT; 
              (2, 3) finds collocates 2 words left and 3 words right of target.
            - None: Uses default of 5 for 'window' method
        filters (FilterOptions | None): Dictionary of filters to apply to results.
            All filters (except ``max_adjusted_p``) are applied BEFORE multiple testing 
            correction, defining the "family" of hypotheses being tested. This maximizes 
            statistical power by not correcting for collocates that were never of interest.
            
            Available filters:
            
            - 'stopwords': list[str] - Words to exclude from results
            - 'min_word_length': int - Minimum character length for collocates
            - 'min_obs_local': int - Minimum observed local frequency
            - 'max_obs_local': int - Maximum observed local frequency
            - 'min_obs_global': int - Minimum global frequency
            - 'max_obs_global': int - Maximum global frequency
            - 'min_exp_local': float - Minimum expected local frequency
            - 'max_exp_local': float - Maximum expected local frequency
            - 'min_ratio_local': float - Minimum local frequency ratio (obs/exp)
            - 'max_ratio_local': float - Maximum local frequency ratio (obs/exp)
            - 'max_p': float - Maximum raw p-value threshold
            - 'max_adjusted_p': float - Maximum adjusted p-value (requires correction,
              applied after correction is computed)
            
        correction (str, optional): Multiple testing correction method. When set,
            an ``adjusted_p_value`` column is added to the results. The correction
            is applied AFTER all other filters, so only collocates that pass those
            filters count toward the number of tests.
            
            - 'bonferroni': Bonferroni correction (conservative, controls family-wise 
              error rate).
            - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate).
            - None: No correction (default).
        as_dataframe (bool): If True, return results as a pandas DataFrame. Default is True.
        max_sentence_length (int | None): Maximum sentence length. Longer sentences 
            are truncated to avoid memory bloat from outliers. Set to None for no limit.
            Default is 256.
        alternative (str): Alternative hypothesis for Fisher's exact test. Options are:
            'greater' (test if observed co-occurrence is greater than expected, default),
            'less' (test if observed is less than expected), or 'two-sided' (test if 
            observed differs from expected).
        sort_by (str): Field to sort results by. Default is 'obs_local'.
        ascending (bool): Sort direction. Default is False (descending).
        batch_words (int): Target number of tokens per processing batch. Larger values
            use more memory but reduce per-batch overhead. Default is 100,000.
    
    Returns:
        list[dict] | pd.DataFrame: Collocation results with the following fields:
        
            - **target** (str): The target word.
            - **collocate** (str): The co-occurring word.
            - **obs_local** (int): Observed co-occurrence count (contexts where both appear).
            - **exp_local** (float): Expected co-occurrence count under independence.
            - **ratio_local** (float): Ratio of observed to expected (obs_local / exp_local).
              Values > 1 indicate attraction, < 1 indicate repulsion.
            - **obs_global** (int): Total occurrences of the collocate in the corpus.
            - **p_value** (float): P-value from Fisher's exact test.
            - **adjusted_p_value** (float, optional): Present only if ``correction`` is set.
    """
    # Validate parameters that don't require data access
    if correction is not None and correction not in VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction method '{correction}'. "
            f"Valid methods are: {VALID_CORRECTIONS}"
        )
    if not isinstance(sort_by, str):
        raise ValueError("sort_by must be a string")
    if not isinstance(ascending, bool):
        raise ValueError("ascending must be a boolean")
    valid_sort_keys = {
        "target", "collocate", "exp_local", "obs_local",
        "ratio_local", "obs_global", "p_value", "adjusted_p_value",
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f"Invalid sort_by '{sort_by}'. Valid keys are: {valid_sort_keys}")
    if sort_by == "adjusted_p_value" and correction is None:
        raise ValueError("sort_by='adjusted_p_value' requires a correction method to be set")
    
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = list(set(target_words))
    
    if not target_words:
        raise ValueError("target_words cannot be empty")
    
    if method not in ['window', 'sentence']:
        raise ValueError(f"Invalid method: {method}. Valid methods are 'window' and 'sentence'.")

    if method == 'sentence':
        if horizon is not None:
            raise ValueError(
                "The 'horizon' parameter is not applicable when method='sentence'. "
                "Sentence-based collocation uses entire sentences as context units. "
                "Please remove the 'horizon' argument or use method='window'."
            )
    elif method == 'window':
        if horizon is None:
            horizon = 5
    
    if filters:
        filter_strs = []
        if 'max_p' in filters:
            filter_strs.append(f"max_p={filters['max_p']}")
        if 'stopwords' in filters:
            filter_strs.append(f"stopwords=<{len(filters['stopwords'])} words>")
        if 'min_word_length' in filters:
            filter_strs.append(f"min_word_length={filters['min_word_length']}")
        if 'min_exp_local' in filters:
            filter_strs.append(f"min_exp_local={filters['min_exp_local']}")
        if 'max_exp_local' in filters:
            filter_strs.append(f"max_exp_local={filters['max_exp_local']}")
        if 'min_obs_local' in filters:
            filter_strs.append(f"min_obs_local={filters['min_obs_local']}")
        if 'max_obs_local' in filters:
            filter_strs.append(f"max_obs_local={filters['max_obs_local']}")
        if 'min_ratio_local' in filters:
            filter_strs.append(f"min_ratio_local={filters['min_ratio_local']}")
        if 'max_ratio_local' in filters:
            filter_strs.append(f"max_ratio_local={filters['max_ratio_local']}")
        if 'min_obs_global' in filters:
            filter_strs.append(f"min_obs_global={filters['min_obs_global']}")
        if 'max_obs_global' in filters:
            filter_strs.append(f"max_obs_global={filters['max_obs_global']}")
        logger.info(f"Filters: {', '.join(filter_strs)}")
    
    # Dispatch to backend (all backends now accept iterables and handle batching)
    backend_kwargs = dict(
        alternative=alternative,
        batch_words=batch_words,
        max_sentence_length=max_sentence_length,
    )
    
    if CYTHON_AVAILABLE:
        if method == 'window':
            results = _calculate_collocations_window_cython(
                sentences, target_words, horizon=horizon, **backend_kwargs
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_cython(
                sentences, target_words, **backend_kwargs
            )
    else:
        logger.debug("Using pure Python fallback for find_collocates (Cython not available)")
        if method == 'window':
            results = _calculate_collocations_window_python(
                sentences, target_words, horizon=horizon, **backend_kwargs
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_python(
                sentences, target_words, **backend_kwargs
            )

    # =========================================================================
    # STAGE 1: Apply all filters BEFORE multiple testing correction
    # All user-specified filters define the "family" of hypotheses being tested.
    # This maximizes statistical power by not correcting for collocates that
    # were never of interest in the first place.
    # =========================================================================
    if filters:
        valid_keys = {
            'max_p', 'max_adjusted_p', 'stopwords', 'min_word_length', 
            'min_exp_local', 'max_exp_local',
            'min_obs_local', 'max_obs_local', 'min_ratio_local', 'max_ratio_local',
            'min_obs_global', 'max_obs_global'
        }
        invalid_keys = set(filters.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid filter keys: {invalid_keys}. Valid keys are: {valid_keys}")
        
        # Validate and extract filter values upfront
        stopwords_set = None
        if 'stopwords' in filters:
            stopwords = filters['stopwords']
            if not isinstance(stopwords, (list, set)):
                raise ValueError("stopwords must be a list or set of strings")
            stopwords_set = set(stopwords)
        
        min_word_length = filters.get('min_word_length')
        if min_word_length is not None:
            if not isinstance(min_word_length, int) or min_word_length < 1:
                raise ValueError("min_word_length must be a positive integer")
        
        min_obs_local = filters.get('min_obs_local')
        if min_obs_local is not None:
            if not isinstance(min_obs_local, int) or min_obs_local < 0:
                raise ValueError("min_obs_local must be a non-negative integer")
        
        max_obs_local = filters.get('max_obs_local')
        if max_obs_local is not None:
            if not isinstance(max_obs_local, int) or max_obs_local < 0:
                raise ValueError("max_obs_local must be a non-negative integer")
        
        min_obs_global = filters.get('min_obs_global')
        if min_obs_global is not None:
            if not isinstance(min_obs_global, int) or min_obs_global < 0:
                raise ValueError("min_obs_global must be a non-negative integer")
        
        max_obs_global = filters.get('max_obs_global')
        if max_obs_global is not None:
            if not isinstance(max_obs_global, int) or max_obs_global < 0:
                raise ValueError("max_obs_global must be a non-negative integer")
        
        min_exp_local = filters.get('min_exp_local')
        if min_exp_local is not None:
            if not isinstance(min_exp_local, (int, float)) or min_exp_local < 0:
                raise ValueError("min_exp_local must be a non-negative number")
        
        max_exp_local = filters.get('max_exp_local')
        if max_exp_local is not None:
            if not isinstance(max_exp_local, (int, float)) or max_exp_local < 0:
                raise ValueError("max_exp_local must be a non-negative number")
        
        min_ratio_local = filters.get('min_ratio_local')
        if min_ratio_local is not None:
            if not isinstance(min_ratio_local, (int, float)) or min_ratio_local < 0:
                raise ValueError("min_ratio_local must be a non-negative number")
        
        max_ratio_local = filters.get('max_ratio_local')
        if max_ratio_local is not None:
            if not isinstance(max_ratio_local, (int, float)) or max_ratio_local < 0:
                raise ValueError("max_ratio_local must be a non-negative number")
        
        max_p = filters.get('max_p')
        if max_p is not None:
            if not isinstance(max_p, (int, float)) or max_p < 0 or max_p > 1:
                raise ValueError("max_p must be a number between 0 and 1")
        
        # Single-pass filtering
        def passes_filters(r):
            collocate = r["collocate"]
            if stopwords_set is not None and collocate in stopwords_set:
                return False
            if min_word_length is not None and len(collocate) < min_word_length:
                return False
            if min_obs_local is not None and r["obs_local"] < min_obs_local:
                return False
            if max_obs_local is not None and r["obs_local"] > max_obs_local:
                return False
            if min_obs_global is not None and r["obs_global"] < min_obs_global:
                return False
            if max_obs_global is not None and r["obs_global"] > max_obs_global:
                return False
            if min_exp_local is not None and r["exp_local"] < min_exp_local:
                return False
            if max_exp_local is not None and r["exp_local"] > max_exp_local:
                return False
            if min_ratio_local is not None and r["ratio_local"] < min_ratio_local:
                return False
            if max_ratio_local is not None and r["ratio_local"] > max_ratio_local:
                return False
            if max_p is not None and r["p_value"] > max_p:
                return False
            return True
        
        results = [r for r in results if passes_filters(r)]

    # =========================================================================
    # STAGE 2: Multiple testing correction (based on filtered hypothesis count)
    # Applied only to collocates that passed all filters above.
    # =========================================================================
    if correction is not None and results:
        raw_p_values = [r["p_value"] for r in results]
        adjusted = apply_p_value_correction(raw_p_values, method=correction)
        for r, adj_p in zip(results, adjusted):
            r["adjusted_p_value"] = adj_p

    # =========================================================================
    # STAGE 3: Filter by adjusted p-value (must come after correction)
    # =========================================================================
    if filters and 'max_adjusted_p' in filters:
        max_adj_p = filters['max_adjusted_p']
        if not isinstance(max_adj_p, (int, float)) or max_adj_p < 0 or max_adj_p > 1:
            raise ValueError("max_adjusted_p must be a number between 0 and 1")
        if correction is None:
            raise ValueError("max_adjusted_p filter requires a correction method to be set")
        results = [result for result in results if result["adjusted_p_value"] <= max_adj_p]

    if as_dataframe:
        results = pd.DataFrame(results)
        if sort_by in results.columns:
            results = results.sort_values(sort_by, ascending=ascending, kind="mergesort")
    else:
        results = sorted(results, key=lambda r: r[sort_by], reverse=not ascending)
    return results

def cooc_matrix(
    documents: Iterable[list[str]], 
    horizon: int | tuple[int, int] | None = None,
    method: str = 'window',
    min_word_count: int = 1, 
    min_doc_count: int = 1, 
    max_vocab_size: int | None = None, 
    vocab: list[str] | set | None = None,
    binary: bool = False,
) -> CoocMatrix:
    """
    Calculate a co-occurrence matrix from a corpus of documents.
    
    Processes data in streaming batches to keep memory low even for very large
    corpora. When ``vocab`` is not provided, requires a restartable iterable
    (iterated twice: once for vocabulary building, once for counting). Lists,
    file-backed iterators, and restartable generator classes all work;
    single-use generators require a pre-built ``vocab``.
    
    Returns a CoocMatrix object with flexible indexing:
    
    - ``matrix["word1", "word2"]`` - single count
    - ``matrix["word1"]`` - row as dict {word: count}
    - ``matrix.to_dataframe()`` - pandas DataFrame
    - ``matrix.to_dense()`` - numpy array
    
    Args:
        documents (Iterable[list[str]]): Iterable of tokenized documents, where
            each document is a list of tokens. Must be restartable when ``vocab``
            is not provided (iterated twice).
        horizon: Context window size relative to each word. Only applicable for method='window'.
            If not provided, defaults to 5 for window method. Must not be provided for 
            method='document'.
            - int: Symmetric window (e.g., 5 means 5 words on each side)
            - tuple: Asymmetric window (left, right), e.g., (0, 5) for right-only context
        method: Method for calculating co-occurrences:
            - 'window': Count within sliding window (default, uses horizon)
            - 'document': Bag-of-words within each document (ignores horizon)
        min_word_count: Minimum total count for a word to be included in auto-generated
            vocabulary. Ignored if vocab is provided. Default 1.
        min_doc_count: Minimum number of documents a word must appear in to be included
            in auto-generated vocabulary. Ignored if vocab is provided. Default 1.
        max_vocab_size: Maximum vocabulary size (most frequent words kept). Only applies
            to auto-generated vocabulary. Ignored if vocab is provided. Default None.
        vocab: Predefined vocabulary to use. If provided, this vocabulary is used exactly
            as given without any filtering (min_word_count, min_doc_count, and max_vocab_size
            are ignored). Words in vocab that don't appear in documents will still be
            included in the matrix (with zero counts). When provided, only a single pass
            over documents is needed (single-use generators are accepted).
        binary: If True, count co-occurrences as binary (0/1). Default False.
    
    Returns:
        CoocMatrix: Co-occurrence matrix object.
    
    Example:
        >>> matrix = cooc_matrix(documents, horizon=5)
        >>> matrix["fox", "dog"]      # Get count for word pair
        42
        >>> matrix["fox"]             # Get all co-occurrences for "fox"
        {'quick': 10, 'brown': 8, 'dog': 42, ...}
        >>> df = matrix.to_dataframe()  # Convert to DataFrame if needed
        
        >>> # With predefined vocabulary (no filtering applied)
        >>> matrix = cooc_matrix(documents, vocab=["fox", "dog", "cat"])
    """
    if method not in ('window', 'document'):
        raise ValueError("method must be 'window' or 'document'")
    
    if method == 'document':
        if horizon is not None:
            raise ValueError(
                "The 'horizon' parameter is not applicable when method='document'. "
                "Document-based co-occurrence uses entire documents as context units. "
                "Please remove the 'horizon' argument or use method='window'."
            )
    elif method == 'window':
        if horizon is None:
            horizon = 5
    
    # Pass 1: Build vocabulary (streams through documents once)
    if vocab is not None:
        vocab_list = sorted(set(vocab))
    else:
        word_counts, document_counts, _ = build_vocab_from_iter(documents, max_length=None)
        
        filtered_vocab = {word for word, count in word_counts.items() 
                         if count >= min_word_count and document_counts[word] >= min_doc_count}
        
        if max_vocab_size and len(filtered_vocab) > max_vocab_size:
            filtered_vocab = set(sorted(filtered_vocab, 
                                       key=lambda word: word_counts[word], 
                                       reverse=True)[:max_vocab_size])
        
        vocab_list = sorted(filtered_vocab)
    
    word_to_index = {word: i for i, word in enumerate(vocab_list)}
    
    # Pass 2: Count co-occurrences (streams through documents again)
    if method == 'window':
        if isinstance(horizon, int):
            left_horizon, right_horizon = horizon, horizon
        else:
            left_horizon, right_horizon = horizon[0], horizon[1]
        cooc_sparse = _cooc_window(
            documents, word_to_index, left_horizon, right_horizon, binary
        )
    elif method == 'document':
        cooc_sparse = _cooc_document(documents, word_to_index, binary)
        
    return CoocMatrix(cooc_sparse, vocab_list, word_to_index)


def _cooc_window(
    documents: Iterable[list[str]],
    word_to_index: dict[str, int],
    left_horizon: int,
    right_horizon: int,
    binary: bool,
    batch_words: int = 100_000,
) -> sparse.csr_matrix:
    """
    Calculate window-based co-occurrences by streaming through documents.
    
    Uses Cython acceleration when available, otherwise falls back to pure Python.
    Words not in word_to_index (OOV) are skipped for counting but preserve positional
    distances between vocabulary words.
    
    Args:
        documents: Iterable of tokenized documents.
        word_to_index: Mapping from vocabulary words to matrix indices.
        left_horizon: Number of positions to look left.
        right_horizon: Number of positions to look right.
        binary: If True, count presence (0/1) rather than frequency.
        batch_words: Target token count per batch for Cython path.
    
    Returns:
        Sparse CSR matrix of co-occurrence counts.
    """
    vocab_size = len(word_to_index)
    
    if CYTHON_AVAILABLE and calculate_cooc_matrix_window is not None:
        all_rows = []
        all_cols = []
        all_data = []
        
        for batch in iter_batches(documents, batch_words, max_length=None):
            row_indices, col_indices, data_values = calculate_cooc_matrix_window(
                batch, word_to_index, left_horizon, right_horizon, binary
            )
            if len(row_indices) > 0:
                all_rows.append(row_indices)
                all_cols.append(col_indices)
                all_data.append(data_values)
        
        if not all_rows:
            return sparse.coo_matrix((vocab_size, vocab_size), dtype=np.int64).tocsr()
        
        row_arr = np.concatenate(all_rows)
        col_arr = np.concatenate(all_cols)
        data_arr = np.concatenate(all_data)
        
        cooc_sparse = sparse.coo_matrix(
            (data_arr, (row_arr, col_arr)), shape=(vocab_size, vocab_size), dtype=np.int64
        ).tocsr()
        
        if binary:
            cooc_sparse.data = np.ones_like(cooc_sparse.data)
        
        return cooc_sparse
    
    else:
        logger.debug("Using pure Python fallback for cooc_matrix (Cython not available)")
        cooc_dict: dict[tuple[int, int], int] = defaultdict(int)
        
        for document in documents:
            if not document:
                continue
            for i, word1 in enumerate(document):
                if word1 not in word_to_index:
                    continue
                idx1 = word_to_index[word1]
                start = max(0, i - left_horizon)
                end = min(len(document), i + right_horizon + 1)
                
                for j in range(start, end):
                    if j != i:
                        word2 = document[j]
                        if word2 in word_to_index:
                            idx2 = word_to_index[word2]
                            if binary:
                                cooc_dict[(idx1, idx2)] = 1
                            else:
                                cooc_dict[(idx1, idx2)] += 1
        
        if not cooc_dict:
            return sparse.coo_matrix((vocab_size, vocab_size), dtype=np.int64).tocsr()
        
        rows, cols, data = zip(*((i, j, c) for (i, j), c in cooc_dict.items()))
        return sparse.coo_matrix(
            (data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.int64
        ).tocsr()


def _cooc_document(
    documents: Iterable[list[str]],
    word_to_index: dict[str, int],
    binary: bool,
    batch_words: int = 100_000,
) -> sparse.csr_matrix:
    """
    Calculate document-based (bag-of-words) co-occurrences by streaming.
    
    Each document is treated as a bag of words. Two words co-occur if they appear
    in the same document, regardless of their positions. For binary=False, each
    pair contributes count_i * count_j (matching X @ X.T semantics). Documents
    are processed in batches; document boundaries are never crossed.
    
    Args:
        documents: Iterable of tokenized documents.
        word_to_index: Mapping from vocabulary words to matrix indices.
        binary: If True, count presence (0/1) rather than frequency.
        batch_words: Target token count per batch for Cython path.
    
    Returns:
        Sparse CSR matrix of co-occurrence counts.
    """
    vocab_size = len(word_to_index)
    
    if CYTHON_AVAILABLE and calculate_cooc_matrix_document is not None:
        all_rows = []
        all_cols = []
        all_data = []
        
        for batch in iter_batches(documents, batch_words, max_length=None):
            row_indices, col_indices, data_values = calculate_cooc_matrix_document(
                batch, word_to_index, binary
            )
            if len(row_indices) > 0:
                all_rows.append(row_indices)
                all_cols.append(col_indices)
                all_data.append(data_values)
        
        if not all_rows:
            return sparse.coo_matrix((vocab_size, vocab_size), dtype=np.int64).tocsr()
        
        row_arr = np.concatenate(all_rows)
        col_arr = np.concatenate(all_cols)
        data_arr = np.concatenate(all_data)
        
        cooc_sparse = sparse.coo_matrix(
            (data_arr, (row_arr, col_arr)), shape=(vocab_size, vocab_size), dtype=np.int64
        ).tocsr()
        
        if binary:
            cooc_sparse.data = np.ones_like(cooc_sparse.data)
        
        cooc_sparse.setdiag(0)
        cooc_sparse.eliminate_zeros()
        
        return cooc_sparse
    
    else:
        logger.debug("Using pure Python fallback for cooc_matrix document method (Cython not available)")
        cooc_dict: dict[tuple[int, int], int] = defaultdict(int)
        
        for document in documents:
            if not document:
                continue
            doc_word_counts: dict[int, int] = {}
            for word in document:
                if word in word_to_index:
                    idx = word_to_index[word]
                    doc_word_counts[idx] = doc_word_counts.get(idx, 0) + 1
            
            unique_indices = list(doc_word_counts.keys())
            n_unique = len(unique_indices)
            
            for a in range(n_unique):
                idx_a = unique_indices[a]
                for b in range(a + 1, n_unique):
                    idx_b = unique_indices[b]
                    if binary:
                        cooc_dict[(idx_a, idx_b)] = 1
                        cooc_dict[(idx_b, idx_a)] = 1
                    else:
                        weight = doc_word_counts[idx_a] * doc_word_counts[idx_b]
                        cooc_dict[(idx_a, idx_b)] += weight
                        cooc_dict[(idx_b, idx_a)] += weight
        
        if not cooc_dict:
            return sparse.coo_matrix((vocab_size, vocab_size), dtype=np.int64).tocsr()
        
        rows, cols, data = zip(*((i, j, c) for (i, j), c in cooc_dict.items()))
        return sparse.coo_matrix(
            (data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.int64
        ).tocsr()

def plot_collocates(
    collocates: list[dict] | pd.DataFrame,
    x_col: str = 'ratio_local',
    y_col: str = 'p_value',
    x_scale: str = 'log',
    y_scale: str = 'log',
    color: str | list[str] | None = None,
    colormap: str = 'viridis',
    color_by: str | None = None,
    title: str | None = None,
    figsize: tuple = (10, 8),
    fontsize: int = 10,
    show_labels: bool = False,
    label_top_n: int | None = None,
    alpha: float = 0.6,
    marker_size: int = 50,
    show_diagonal: bool = False,
    diagonal_color: str = 'red',
    filename: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None
) -> None:
    """
    Visualize collocation results as a 2D scatter plot.
    
    Creates a customizable scatter plot from collocation data. By default, plots
    ratio_local (x-axis) vs p_value (y-axis) with logarithmic scales, but allows
    full flexibility to plot any columns with any scale type.
    
    Args:
        collocates (list[dict] | pd.DataFrame): Output from find_collocates, 
            either as a list of dictionaries or DataFrame.
        x_col (str): Column name to plot on x-axis. Common choices: 'ratio_local', 
            'obs_local', 'exp_local', 'obs_global'. Default is 'ratio_local'.
        y_col (str): Column name to plot on y-axis. Common choices: 'p_value', 
            'obs_local', 'ratio_local', 'obs_global'. Default is 'p_value'.
        x_scale (str): Scale for x-axis. Options: 'log', 'linear', 'symlog', 'logit'.
            For ratio_local, 'log' makes the scale symmetric around 1. Default is 'log'.
        y_scale (str): Scale for y-axis. Options: 'log', 'linear', 'symlog', 'logit'.
            For p_value, 'log' is recommended to visualize small values. Default is 'log'.
        color (str | list[str] | None): Color(s) for the points. Can be a single 
            color string, list of colors, or None to use default.
        colormap (str): Matplotlib colormap to use when color_by is specified. 
            Default is 'viridis'.
        color_by (str | None): Column name to use for coloring points (e.g., 
            'obs_local', 'obs_global').
        title (str | None): Title for the plot.
        figsize (tuple): Figure size as (width, height) in inches. Default is (10, 8).
        fontsize (int): Base font size for labels. Default is 10.
        show_labels (bool): Whether to show collocate text labels next to points. 
            Default is False.
        label_top_n (int | None): If specified, only label the top N points. When 
            color_by is set, ranks by that column; otherwise ranks by y-axis values. 
            For p_value, labels smallest (most significant) values; for other metrics, 
            labels largest values.
        alpha (float): Transparency of points (0.0 to 1.0). Default is 0.6.
        marker_size (int): Size of markers. Default is 50.
        show_diagonal (bool): Whether to draw a diagonal reference line (y=x). Useful 
            for observed vs expected plots. Default is False.
        diagonal_color (str): Color of the diagonal reference line. Default is 'red'.
        filename (str | None): If provided, saves the figure to the specified file path.
        xlabel (str | None): Label for x-axis. If None, auto-generated from x_col 
            and x_scale.
        ylabel (str | None): Label for y-axis. If None, auto-generated from y_col 
            and y_scale.
    
    Returns:
        None: Displays the plot using matplotlib. To further customize, use plt.gca() 
            to get the current axes object after calling this function.
    
    Example:
        # Basic usage: ratio vs p-value with log scales (default)
        collocates = find_collocates(sentences, ['天'])
        plot_collocates(collocates)
        
        # Plot observed vs expected frequency
        plot_collocates(collocates, x_col='exp_local', y_col='obs_local',
        ...                 x_scale='linear', y_scale='linear')
        
        # With labels and custom styling
        plot_collocates(collocates, show_labels=True, label_top_n=20,
        ...                 color='red', title='Collocates of 天')
    """
    # Lazy import matplotlib
    import matplotlib.pyplot as plt
    
    if isinstance(collocates, list):
        if not collocates:
            raise ValueError("Empty collocates list provided")
        df = pd.DataFrame(collocates)
    else:
        df = collocates.copy()
    
    required_cols = [x_col, y_col, 'collocate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(df.columns)}")
    
    x = df[x_col].values
    y = df[y_col].values
    labels = df['collocate'].values
    
    # Handle zero/negative values for log scales
    if x_scale == 'log':
        zero_or_neg_x = (x <= 0).sum()
        if zero_or_neg_x > 0:
            logger.warning(f"{zero_or_neg_x} values in {x_col} are ≤ 0. Replacing with 1e-300 for log scale.")
            x = np.where(x <= 0, 1e-300, x)
    
    if y_scale == 'log':
        zero_or_neg_y = (y <= 0).sum()
        if zero_or_neg_y > 0:
            logger.warning(f"{zero_or_neg_y} values in {y_col} are ≤ 0. Replacing with 1e-300 for log scale.")
            y = np.where(y <= 0, 1e-300, y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by is not None:
        if color_by not in df.columns:
            raise ValueError(f"Column '{color_by}' not found in data. Available columns: {list(df.columns)}")
        color_values = df[color_by].values
        scatter = ax.scatter(x, y, c=color_values, cmap=colormap, alpha=alpha, 
                           s=marker_size, edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by, fontsize=fontsize)
    else:
        point_color = color if color is not None else '#1f77b4'
        ax.scatter(x, y, c=point_color, alpha=alpha, s=marker_size, 
                  edgecolors='black', linewidths=0.5)
    
    if show_labels:
        if label_top_n is not None:
            if color_by is not None:
                sort_values = df[color_by].values
                if color_by == 'p_value':
                    indices_to_label = np.argsort(sort_values)[:label_top_n]
                else:
                    indices_to_label = np.argsort(sort_values)[-label_top_n:][::-1]
            else:
                if y_col == 'p_value':
                    indices_to_label = np.argsort(y)[:label_top_n]
                else:
                    indices_to_label = np.argsort(y)[-label_top_n:][::-1]
        else:
            indices_to_label = range(len(labels))
        
        for idx in indices_to_label:
            ax.annotate(labels[idx], (x[idx], y[idx]), 
                       fontsize=fontsize-2, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    if xlabel is None:
        scale_suffix = f' ({x_scale} scale)' if x_scale != 'linear' else ''
        xlabel = f'{x_col}{scale_suffix}'
    if ylabel is None:
        scale_suffix = f' ({y_scale} scale)' if y_scale != 'linear' else ''
        ylabel = f'{y_col}{scale_suffix}'
    
    ax.set_xlabel(xlabel, fontsize=fontsize+2)
    ax.set_ylabel(ylabel, fontsize=fontsize+2)
    if title:
        ax.set_title(title, fontsize=fontsize+4)
    
    if x_scale != 'linear':
        ax.set_xscale(x_scale)
    
    if y_scale != 'linear':
        ax.set_yscale(y_scale)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_diagonal:
        x_data = df[x_col].values
        y_data = df[y_col].values
        min_val = max(np.min(x_data), np.min(y_data))
        max_val = min(np.max(x_data), np.max(y_data))
        ax.plot([min_val, max_val], [min_val, max_val], '--', 
                color=diagonal_color, linewidth=2.5, zorder=1)
    
    if x_col == 'ratio_local':
        ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
                   label='ratio = 1 (expected frequency)')
    
    legend_elements = ax.get_legend_handles_labels()[0]
    
    if len(legend_elements) > 0:
        ax.legend(fontsize=fontsize-2, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.show()