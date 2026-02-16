import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Union, TypedDict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact as scipy_fisher_exact
from tqdm.auto import tqdm

from ..utils import apply_p_value_correction, VALID_CORRECTIONS

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

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
        calculate_collocations_window,
        calculate_collocations_sentence,
        calculate_cooc_matrix_window
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    calculate_collocations_window = None
    calculate_collocations_sentence = None
    calculate_cooc_matrix_window = None
    logger.warning("Cython extensions not available; using slower Python fallback.")


# =============================================================================
# CoocMatrix Class - Vocabulary-aware co-occurrence matrix
# =============================================================================

class CoocMatrix:
    """
    A vocabulary-aware co-occurrence matrix with intuitive indexing.
    
    Supports flexible indexing by word strings or integer indices:
        matrix["word1", "word2"]  → single count (int)
        matrix[132, 5234]         → single count (int)
        matrix["word1"]           → row as dict {word: count}
        matrix["word1", :]        → row as dict {word: count}
        matrix[:, "word2"]        → column as dict {word: count}
    
    Internally stores data as a scipy sparse CSR matrix for memory efficiency.
    
    Attributes:
        vocab (List[str]): List of vocabulary words in index order.
        word_to_index (Dict[str, int]): Mapping from words to matrix indices.
        index_to_word (Dict[int, str]): Mapping from matrix indices to words.
        shape (Tuple[int, int]): Shape of the matrix (vocab_size, vocab_size).
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
        vocab_list: List[str], 
        word_to_index: Dict[str, int]
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
    
    def _row_to_dict(self, row_idx: int) -> Dict[str, int]:
        """Convert a matrix row to a dict of {word: count} for non-zero entries."""
        row = self._matrix.getrow(row_idx)
        return {self._i2w[col]: int(val) for col, val in zip(row.indices, row.data)}
    
    def _col_to_dict(self, col_idx: int) -> Dict[str, int]:
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
    def vocab(self) -> List[str]:
        """List of vocabulary words in index order. Returns the internal list directly."""
        return self._vocab
    
    @property
    def word_to_index(self) -> Dict[str, int]:
        """Dictionary mapping words to their matrix indices. Returns the internal dict directly."""
        return self._w2i
    
    @property
    def index_to_word(self) -> Dict[int, str]:
        """Dictionary mapping matrix indices to words. Returns the internal dict directly."""
        return self._i2w
    
    @property
    def shape(self) -> Tuple[int, int]:
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
    stopwords: List[str]
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
    
    This is the shared statistical computation used by both window and sentence methods,
    and by both Python and Cython implementations.
    
    Args:
        target: Target word string
        candidate: Collocate word string
        a: Co-occurrence count (target with collocate)
        b: Target without collocate count  
        c: Collocate without target count
        d: Neither target nor collocate count
        alternative: Alternative hypothesis for Fisher's exact test
    
    Returns:
        Dictionary with collocation statistics: target, collocate, exp_local,
        obs_local, ratio_local, obs_global, p_value
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
    Build result list from Python-collected collocation counts.
    
    This is the shared result-building logic used by both window and sentence
    Python implementations.
    
    Args:
        target_words: List of target words to process
        target_counts: Dict mapping target -> count of contexts containing target
        candidate_counts: Dict mapping target -> Counter of candidate co-occurrences
        global_counts: Dict/Counter mapping token -> global count
        total: Total count (tokens for window, sentences for sentence method)
        alternative: Alternative hypothesis for Fisher's exact test
        method: 'window' or 'sentence' - determines how d is calculated
    
    Returns:
        List of dictionaries with collocation statistics
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


def _calculate_collocations_window_cython(tokenized_sentences, target_words, horizon=5, alternative='greater'):
    """
    Cython implementation of window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences (already preprocessed)
        target_words: List or set of target words
        horizon: Window size - int for symmetric, or tuple (left, right) where left/right
                 indicate how many words to look on each side OF THE TARGET WORD.
                 E.g., (0, 5) finds collocates up to 5 words to the RIGHT of target.
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    # Normalize horizon to (left, right) tuple
    # User specifies (left, right) relative to TARGET, but internally we need to
    # swap because the algorithm iterates over candidates and looks for targets
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        # Swap: user's "right of target" becomes algorithm's "left from candidate"
        left_horizon, right_horizon = horizon[1], horizon[0]
    
    T_count_total, candidate_counts_total, token_counter_total, total_tokens, word2idx, idx2word, target_indices = calculate_collocations_window(
        tokenized_sentences, target_words, left_horizon, right_horizon
    )
    
    if T_count_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []

    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        # Only iterate over candidates with non-zero co-occurrence counts
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


def _calculate_collocations_window_python(tokenized_sentences, target_words, horizon=5, alternative='greater'):
    """
    Pure Python window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences (already preprocessed)
        target_words: List or set of target words
        horizon: Window size - int for symmetric, or tuple (left, right) where left/right
                 indicate how many words to look on each side OF THE TARGET WORD.
                 E.g., (0, 5) finds collocates up to 5 words to the RIGHT of target.
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    # Normalize horizon to (left, right) tuple
    # User specifies (left, right) relative to TARGET, but internally we need to
    # swap because the algorithm iterates over candidates and looks for targets
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        # Swap: user's "right of target" becomes algorithm's "left from candidate"
        left_horizon, right_horizon = horizon[1], horizon[0]
    
    total_tokens = 0
    target_set = set(target_words)
    T_count = {target: 0 for target in target_words}
    candidate_in_context = {target: Counter() for target in target_words}
    token_counter = Counter()

    for sentence in tqdm(tokenized_sentences):
        for i, token in enumerate(sentence):
            total_tokens += 1
            token_counter[token] += 1

            start = max(0, i - left_horizon)
            end = min(len(sentence), i + right_horizon + 1)

            # Scan window once; O(1) set lookup replaces per-target rescanning
            seen_targets = set()
            for j in range(start, end):
                if j != i:
                    word = sentence[j]
                    if word in target_set and word not in seen_targets:
                        seen_targets.add(word)
                        T_count[word] += 1
                        candidate_in_context[word][token] += 1

    return _build_results_from_counts(
        target_words, T_count, candidate_in_context, token_counter, total_tokens, alternative
    )


def _calculate_collocations_sentence_cython(tokenized_sentences, target_words, alternative='greater'):
    """
    Cython implementation of sentence-based collocation calculation.
    
    Pre-converts all sentences to integer arrays and uses lightweight buffers
    for uniqueness checks. All hot loops run with nogil using memoryviews.
    
    Args:
        tokenized_sentences: List of tokenized sentences (already preprocessed)
        target_words: List or set of target words
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    candidate_sentences_total, sentences_with_token_total, total_sentences, word2idx, idx2word, target_indices = calculate_collocations_sentence(
        tokenized_sentences, target_words
    )
    
    if candidate_sentences_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []

    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        # Only iterate over candidates with non-zero co-occurrence counts
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


def _calculate_collocations_sentence_python(tokenized_sentences, target_words, alternative='greater'):
    """
    Pure Python sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences (already preprocessed)
        target_words: List or set of target words
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    total_sentences = len(tokenized_sentences)
    candidate_in_sentences = {target: Counter() for target in target_words}
    sentences_with_token = defaultdict(int)

    for sentence in tqdm(tokenized_sentences):
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
    sentences: List[List[str]], 
    target_words: Union[str, List[str]], 
    method: str = 'window', 
    horizon: Optional[Union[int, tuple]] = None, 
    filters: Optional[FilterOptions] = None, 
    correction: Optional[str] = None,
    as_dataframe: bool = True,
    max_sentence_length: Optional[int] = 256,
    alternative: str = 'greater'
) -> Union[List[Dict], pd.DataFrame]:
    """
    Find collocates for target words within a corpus of sentences.
    
    Args:
        sentences (List[List[str]]): List of tokenized sentences, where each sentence 
            is a list of tokens.
        target_words (Union[str, List[str]]): Target word(s) to find collocates for.
        method (str): Method to use for calculating collocations. Either 'window' or 
            'sentence'. 'window' uses a sliding window of specified horizon around each 
            token. 'sentence' considers whole sentences as context units (horizon not 
            applicable). Default is 'window'.
        horizon (Optional[Union[int, tuple]]): Context window size relative to the target 
            word. Only applicable when method='window'. Must be None when method='sentence'.
            - int: Symmetric window (e.g., 5 means 5 words on each side of target)
            - tuple: Asymmetric window (left, right) specifying how many words to look
              on each side of the target word: (0, 5) finds collocates up to 5 words to 
              the RIGHT of target; (5, 0) finds collocates up to 5 words to the LEFT; 
              (2, 3) finds collocates 2 words left and 3 words right of target.
            - None: Uses default of 5 for 'window' method
        filters (Optional[FilterOptions]): Dictionary of filters to apply to results.
            All filters (except ``max_adjusted_p``) are applied BEFORE multiple testing 
            correction, defining the "family" of hypotheses being tested. This maximizes 
            statistical power by not correcting for collocates that were never of interest.
            
            Available filters:
            
            - 'stopwords': List[str] - Words to exclude from results
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
        max_sentence_length (Optional[int]): Maximum sentence length for preprocessing. 
            Used by both 'window' and 'sentence' methods. Longer sentences will be truncated 
            to avoid memory bloat from outliers. Set to None for no limit. Default is 256.
        alternative (str): Alternative hypothesis for Fisher's exact test. Options are:
            'greater' (test if observed co-occurrence is greater than expected, default),
            'less' (test if observed is less than expected), or 'two-sided' (test if 
            observed differs from expected).
    
    Returns:
        Union[List[Dict], pd.DataFrame]: Collocation results with the following fields:
        
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
    if not sentences:
        raise ValueError("sentences cannot be empty")
    if not all(isinstance(s, list) for s in sentences):
        raise ValueError("sentences must be a list of lists (tokenized sentences)")
    
    # Validate correction parameter
    if correction is not None and correction not in VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction method '{correction}'. "
            f"Valid methods are: {VALID_CORRECTIONS}"
        )
    
    # Preprocess: filter empty sentences and trim to max length in one pass
    if max_sentence_length is not None:
        sentences = [s[:max_sentence_length] for s in sentences if s]
    else:
        sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("All sentences are empty")
    
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = list(set(target_words))
    
    if not target_words:
        raise ValueError("target_words cannot be empty")
    
    if method not in ['window', 'sentence']:
        raise ValueError(f"Invalid method: {method}. Valid methods are 'window' and 'sentence'.")

    # Validate horizon parameter based on method
    if method == 'sentence':
        if horizon is not None:
            raise ValueError(
                "The 'horizon' parameter is not applicable when method='sentence'. "
                "Sentence-based collocation uses entire sentences as context units. "
                "Please remove the 'horizon' argument or use method='window'."
            )
    elif method == 'window':
        if horizon is None:
            horizon = 5  # Default value for window method
    
    # Print filters if provided
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
    
    if CYTHON_AVAILABLE:
        if method == 'window':
            results = _calculate_collocations_window_cython(
                sentences, target_words, horizon=horizon, alternative=alternative
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_cython(
                sentences, target_words, alternative=alternative
            )
    else:
        logger.debug("Using pure Python fallback for find_collocates (Cython not available)")
        if method == 'window':
            results = _calculate_collocations_window_python(
                sentences, target_words, horizon=horizon, alternative=alternative
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_python(
                sentences, target_words, alternative=alternative
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
    return results

def cooc_matrix(
    documents: List[List[str]], 
    horizon: Optional[Union[int, Tuple[int, int]]] = None,
    method: str = 'window',
    min_word_count: int = 1, 
    min_doc_count: int = 1, 
    max_vocab_size: Optional[int] = None, 
    vocab: Optional[Union[List[str], set]] = None,
    binary: bool = False,
) -> CoocMatrix:
    """
    Calculate a co-occurrence matrix from a list of documents.
    
    Returns a CoocMatrix object with intuitive vocabulary-aware indexing:
        matrix["word1", "word2"]  → single count
        matrix["word1"]           → row as dict {word: count}
        matrix.to_dataframe()     → pandas DataFrame
        matrix.to_dense()         → numpy array
    
    Args:
        documents: List of tokenized documents, where each document is a list of tokens.
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
            included in the matrix (with zero counts).
        binary: If True, count co-occurrences as binary (0/1). Default False.
    
    Returns:
        CoocMatrix: A vocabulary-aware co-occurrence matrix object.
    
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
    # Validation
    if not documents:
        raise ValueError("documents cannot be empty")
    if not all(isinstance(doc, list) for doc in documents):
        raise ValueError("documents must be a list of lists (tokenized documents)")
    if method not in ('window', 'document'):
        raise ValueError("method must be 'window' or 'document'")
    
    # Validate horizon parameter based on method
    if method == 'document':
        if horizon is not None:
            raise ValueError(
                "The 'horizon' parameter is not applicable when method='document'. "
                "Document-based co-occurrence uses entire documents as context units. "
                "Please remove the 'horizon' argument or use method='window'."
            )
    elif method == 'window':
        if horizon is None:
            horizon = 5  # Default value for window method
    
    # Build vocabulary
    if vocab is not None:
        # User-provided vocabulary: use exactly as given, no filtering
        vocab_list = sorted(set(vocab))
    else:
        # Auto-generate vocabulary with filtering
        word_counts = Counter()
        document_counts = Counter()
        for document in documents:
            word_counts.update(document)
            document_counts.update(set(document))
        
        filtered_vocab = {word for word, count in word_counts.items() 
                         if count >= min_word_count and document_counts[word] >= min_doc_count}
        
        if max_vocab_size and len(filtered_vocab) > max_vocab_size:
            filtered_vocab = set(sorted(filtered_vocab, 
                                       key=lambda word: word_counts[word], 
                                       reverse=True)[:max_vocab_size])
        
        vocab_list = sorted(filtered_vocab)
    
    word_to_index = {word: i for i, word in enumerate(vocab_list)}
    
    # Calculate co-occurrences
    if method == 'window':
        # Normalize horizon for window method
        if isinstance(horizon, int):
            left_horizon, right_horizon = horizon, horizon
        else:
            left_horizon, right_horizon = horizon[0], horizon[1]
        cooc_sparse = _cooc_window(
            documents, word_to_index, left_horizon, right_horizon, binary
        )
    elif method == 'document':
        cooc_sparse = _cooc_document(documents, word_to_index, binary)
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are 'window' and 'document'.")
    
    return CoocMatrix(cooc_sparse, vocab_list, word_to_index)


def _cooc_window(
    documents: List[List[str]],
    word_to_index: Dict[str, int],
    left_horizon: int,
    right_horizon: int,
    binary: bool
) -> sparse.csr_matrix:
    """
    Calculate window-based co-occurrences.
    
    Uses Cython acceleration when available, otherwise falls back to pure Python.
    Words not in word_to_index (OOV) are skipped for counting but preserve positional
    distances between vocabulary words.
    
    Args:
        documents: List of tokenized documents.
        word_to_index: Mapping from vocabulary words to matrix indices.
        left_horizon: Number of positions to look left.
        right_horizon: Number of positions to look right.
        binary: If True, count presence (0/1) rather than frequency.
    
    Returns:
        Sparse CSR matrix of co-occurrence counts.
    """
    vocab_size = len(word_to_index)
    
    if CYTHON_AVAILABLE and calculate_cooc_matrix_window is not None:
        # Fast Cython path
        row_indices, col_indices, data_values = calculate_cooc_matrix_window(
            documents, word_to_index, left_horizon, right_horizon, binary
        )
        
        if len(row_indices) == 0:
            return sparse.coo_matrix((vocab_size, vocab_size), dtype=np.int64).tocsr()
        
        # tocsr() sums duplicate entries automatically
        cooc_sparse = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)), shape=(vocab_size, vocab_size), dtype=np.int64
        ).tocsr()
        
        # For binary mode, reset summed duplicates back to 1
        if binary:
            cooc_sparse.data = np.ones_like(cooc_sparse.data)
        
        return cooc_sparse
    
    else:
        # Python fallback - preserves distances by keeping OOV positions
        logger.debug("Using pure Python fallback for cooc_matrix (Cython not available)")
        cooc_dict = defaultdict(int)
        
        for document in documents:
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
    documents: List[List[str]],
    word_to_index: Dict[str, int],
    binary: bool
) -> sparse.csr_matrix:
    """
    Calculate document-based (bag-of-words) co-occurrences using matrix multiplication.
    
    Each document is treated as a bag of words. Two words co-occur if they appear
    in the same document, regardless of their positions. Uses efficient sparse
    matrix multiplication: cooc = X @ X.T where X is the term-document matrix.
    
    Args:
        documents: List of tokenized documents.
        word_to_index: Mapping from vocabulary words to matrix indices.
        binary: If True, count presence (0/1) rather than frequency.
    
    Returns:
        Sparse CSR matrix of co-occurrence counts.
    """
    vocab_size = len(word_to_index)
    n_docs = len(documents)
    
    # Build term-document matrix (vocab_size x n_docs)
    row_indices = []
    col_indices = []
    data = []
    
    for doc_idx, document in enumerate(documents):
        doc_words = [w for w in document if w in word_to_index]
        doc_word_counts = Counter(doc_words)
        
        for word, count in doc_word_counts.items():
            word_idx = word_to_index[word]
            row_indices.append(word_idx)
            col_indices.append(doc_idx)
            data.append(1 if binary else count)
    
    if not row_indices:
        return sparse.csr_matrix((vocab_size, vocab_size), dtype=np.int64)
    
    # Create sparse term-document matrix
    X = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(vocab_size, n_docs),
        dtype=np.int64
    )
    
    # Co-occurrence = X @ X.T
    cooc = X @ X.T
    
    # Zero out diagonal (no self-co-occurrence)
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    
    return cooc

def plot_collocates(
    collocates: Union[List[Dict], pd.DataFrame],
    x_col: str = 'ratio_local',
    y_col: str = 'p_value',
    x_scale: str = 'log',
    y_scale: str = 'log',
    color: Optional[Union[str, List[str]]] = None,
    colormap: str = 'viridis',
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    fontsize: int = 10,
    show_labels: bool = False,
    label_top_n: Optional[int] = None,
    alpha: float = 0.6,
    marker_size: int = 50,
    show_diagonal: bool = False,
    diagonal_color: str = 'red',
    filename: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> None:
    """
    Visualize collocation results as a 2D scatter plot.
    
    Creates a customizable scatter plot from collocation data. By default, plots
    ratio_local (x-axis) vs p_value (y-axis) with logarithmic scales, but allows
    full flexibility to plot any columns with any scale type.
    
    Args:
        collocates (Union[List[Dict], pd.DataFrame]): Output from find_collocates, 
            either as a list of dictionaries or DataFrame.
        x_col (str): Column name to plot on x-axis. Common choices: 'ratio_local', 
            'obs_local', 'exp_local', 'obs_global'. Default is 'ratio_local'.
        y_col (str): Column name to plot on y-axis. Common choices: 'p_value', 
            'obs_local', 'ratio_local', 'obs_global'. Default is 'p_value'.
        x_scale (str): Scale for x-axis. Options: 'log', 'linear', 'symlog', 'logit'.
            For ratio_local, 'log' makes the scale symmetric around 1. Default is 'log'.
        y_scale (str): Scale for y-axis. Options: 'log', 'linear', 'symlog', 'logit'.
            For p_value, 'log' is recommended to visualize small values. Default is 'log'.
        color (Optional[Union[str, List[str]]]): Color(s) for the points. Can be a single 
            color string, list of colors, or None to use default.
        colormap (str): Matplotlib colormap to use when color_by is specified. 
            Default is 'viridis'.
        color_by (Optional[str]): Column name to use for coloring points (e.g., 
            'obs_local', 'obs_global').
        title (Optional[str]): Title for the plot.
        figsize (tuple): Figure size as (width, height) in inches. Default is (10, 8).
        fontsize (int): Base font size for labels. Default is 10.
        show_labels (bool): Whether to show collocate text labels next to points. 
            Default is False.
        label_top_n (Optional[int]): If specified, only label the top N points. When 
            color_by is set, ranks by that column; otherwise ranks by y-axis values. 
            For p_value, labels smallest (most significant) values; for other metrics, 
            labels largest values.
        alpha (float): Transparency of points (0.0 to 1.0). Default is 0.6.
        marker_size (int): Size of markers. Default is 50.
        show_diagonal (bool): Whether to draw a diagonal reference line (y=x). Useful 
            for observed vs expected plots. Default is False.
        diagonal_color (str): Color of the diagonal reference line. Default is 'red'.
        filename (Optional[str]): If provided, saves the figure to the specified file path.
        xlabel (Optional[str]): Label for x-axis. If None, auto-generated from x_col 
            and x_scale.
        ylabel (Optional[str]): Label for y-axis. If None, auto-generated from y_col 
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
    
    if color is not None:
        colors = color if isinstance(color, str) else color
    elif color_by is not None:
        if color_by not in df.columns:
            raise ValueError(f"Column '{color_by}' not found in data. Available columns: {list(df.columns)}")
        color_values = df[color_by].values
        scatter = ax.scatter(x, y, c=color_values, cmap=colormap, alpha=alpha, 
                           s=marker_size, edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by, fontsize=fontsize)
    else:
        colors = '#1f77b4'
    
    if color_by is None:
        ax.scatter(x, y, c=colors, alpha=alpha, s=marker_size, 
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