import logging
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, TypedDict, Tuple, Any
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact as scipy_fisher_exact

logger = logging.getLogger("qhchina.analytics.collocations")

try:
    from .cython_ext.collocations import (
        calculate_collocations_window,
        calculate_collocations_sentence
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    calculate_collocations_window = None
    calculate_collocations_sentence = None

class FilterOptions(TypedDict, total=False):
    """Type definition for filter options in collocation analysis."""
    max_p: float
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


def _compute_collocation_result(target, candidate, a, b, c, d, total, obs_global, table, alternative='greater'):
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
        total: Total count (tokens or sentences depending on method)
        obs_global: Global observation count for the collocate
        table: Pre-allocated 2x2 numpy array for Fisher's exact test (reused for efficiency)
        alternative: Alternative hypothesis for Fisher's exact test
    
    Returns:
        Dictionary with collocation statistics: target, collocate, exp_local,
        obs_local, ratio_local, obs_global, p_value
    """
    expected = (a + b) * (a + c) / total if total > 0 else 0
    ratio = a / expected if expected > 0 else 0
    
    table[:] = [[a, b], [c, d]]
    _, p_value = scipy_fisher_exact(table, alternative=alternative)
    
    return {
        "target": target,
        "collocate": candidate,
        "exp_local": expected,
        "obs_local": int(a),
        "ratio_local": ratio,
        "obs_global": int(obs_global),
        "p_value": p_value,
    }


def _build_results_from_cython_window(cython_result, target_words, alternative='greater'):
    """
    Build result list from Cython window-based collocation data.
    
    Args:
        cython_result: Tuple from calculate_collocations_window containing:
            (T_count_total, candidate_counts_total, token_counter_total, 
             total_tokens, word2idx, idx2word, target_indices)
        target_words: List of target words (unused, but kept for consistency)
        alternative: Alternative hypothesis for Fisher's exact test
    
    Returns:
        List of dictionaries with collocation statistics
    """
    T_count_total, candidate_counts_total, token_counter_total, total_tokens, word2idx, idx2word, target_indices = cython_result
    
    if T_count_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []
    vocab_size = len(word2idx)
    table = np.zeros((2, 2), dtype=np.int64)

    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        for candidate_idx in range(vocab_size):
            a = candidate_counts_total[t_idx, candidate_idx]
            if a == 0 or candidate_idx == target_word_idx:
                continue
            
            candidate = idx2word[candidate_idx]
            b = T_count_total[t_idx] - a
            c = token_counter_total[candidate_idx] - a
            d = (total_tokens - token_counter_total[target_word_idx]) - (a + b + c)
            
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, total_tokens,
                token_counter_total[candidate_idx], table, alternative
            ))
    
    return results


def _build_results_from_cython_sentence(cython_result, target_words, alternative='greater'):
    """
    Build result list from Cython sentence-based collocation data.
    
    Args:
        cython_result: Tuple from calculate_collocations_sentence containing:
            (candidate_sentences_total, sentences_with_token_total, 
             total_sentences, word2idx, idx2word, target_indices)
        target_words: List of target words (unused, but kept for consistency)
        alternative: Alternative hypothesis for Fisher's exact test
    
    Returns:
        List of dictionaries with collocation statistics
    """
    candidate_sentences_total, sentences_with_token_total, total_sentences, word2idx, idx2word, target_indices = cython_result
    
    if candidate_sentences_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []
    vocab_size = len(word2idx)
    table = np.zeros((2, 2), dtype=np.int64)

    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        for candidate_idx in range(vocab_size):
            a = candidate_sentences_total[t_idx, candidate_idx]
            if a == 0 or candidate_idx == target_word_idx:
                continue
            
            candidate = idx2word[candidate_idx]
            b = sentences_with_token_total[target_word_idx] - a
            c = sentences_with_token_total[candidate_idx] - a
            d = total_sentences - a - b - c
            
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, total_sentences,
                sentences_with_token_total[candidate_idx], table, alternative
            ))
    
    return results


def _build_results_from_counts(target_words, target_counts, candidate_counts, global_counts, total, alternative='greater'):
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
    
    Returns:
        List of dictionaries with collocation statistics
    """
    table = np.zeros((2, 2), dtype=np.int64)
    results = []
    
    for target in target_words:
        for candidate, a in candidate_counts[target].items():
            if candidate == target:
                continue
            
            b = target_counts[target] - a
            c = global_counts[candidate] - a
            d = (total - global_counts[target]) - (a + b + c)
            
            results.append(_compute_collocation_result(
                target, candidate, a, b, c, d, total,
                global_counts[candidate], table, alternative
            ))
    
    return results


def _calculate_collocations_window_cython(tokenized_sentences, target_words, horizon=5, 
                                         max_sentence_length=256, alternative='greater'):
    """
    Cython implementation of window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        horizon: Window size - int for symmetric (left, right) or tuple (left, right)
        max_sentence_length: Maximum sentence length to consider (default 256)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    # Normalize horizon to (left, right) tuple
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        left_horizon, right_horizon = horizon
    
    cython_result = calculate_collocations_window(
        tokenized_sentences, target_words, left_horizon, right_horizon, max_sentence_length
    )
    
    return _build_results_from_cython_window(cython_result, target_words, alternative)


def _calculate_collocations_window(tokenized_sentences, target_words, horizon=5, alternative='greater'):
    """
    Pure Python window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        horizon: Window size - int for symmetric (left, right) or tuple (left, right)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    # Normalize horizon to (left, right) tuple
    if isinstance(horizon, int):
        left_horizon, right_horizon = horizon, horizon
    else:
        left_horizon, right_horizon = horizon
    
    total_tokens = 0
    T_count = {target: 0 for target in target_words}
    candidate_in_context = {target: Counter() for target in target_words}
    token_counter = Counter()

    for sentence in tqdm(tokenized_sentences):
        for i, token in enumerate(sentence):
            total_tokens += 1
            token_counter[token] += 1

            start = max(0, i - left_horizon)
            end = min(len(sentence), i + right_horizon + 1)
            context = sentence[start:i] + sentence[i+1:end]

            for target in target_words:
                if target in context:
                    T_count[target] += 1
                    candidate_in_context[target][token] += 1

    return _build_results_from_counts(
        target_words, T_count, candidate_in_context, token_counter, total_tokens, alternative
    )


def _calculate_collocations_sentence_cython(tokenized_sentences, target_words, max_sentence_length=256, alternative='greater'):
    """
    Cython implementation of sentence-based collocation calculation.
    
    Pre-converts all sentences to integer arrays and uses lightweight buffers
    for uniqueness checks. All hot loops run with nogil using memoryviews.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        max_sentence_length: Maximum sentence length to consider (default 256)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    cython_result = calculate_collocations_sentence(tokenized_sentences, target_words, max_sentence_length)
    
    return _build_results_from_cython_sentence(cython_result, target_words, alternative)


def _calculate_collocations_sentence(tokenized_sentences, target_words, max_sentence_length=256, alternative='greater'):
    """
    Pure Python sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        max_sentence_length: Maximum sentence length to consider (default 256)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    total_sentences = len(tokenized_sentences)
    candidate_in_sentences = {target: Counter() for target in target_words}
    sentences_with_token = defaultdict(int)

    for sentence in tqdm(tokenized_sentences):
        if max_sentence_length is not None and len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        
        unique_tokens = set(sentence)
        for token in unique_tokens:
            sentences_with_token[token] += 1
        for target in target_words:
            if target in unique_tokens:
                candidate_in_sentences[target].update(unique_tokens)

    return _build_results_from_counts(
        target_words, sentences_with_token, candidate_in_sentences, sentences_with_token, total_sentences, alternative
    )

def find_collocates(
    sentences: List[List[str]], 
    target_words: Union[str, List[str]], 
    method: str = 'window', 
    horizon: Optional[Union[int, tuple]] = None, 
    filters: Optional[FilterOptions] = None, 
    as_dataframe: bool = True,
    max_sentence_length: Optional[int] = 256,
    alternative: str = 'greater'
) -> Union[List[Dict], pd.DataFrame]:
    """
    Find collocates for target words within a corpus of sentences.
    
    Parameters:
    -----------
    sentences : List[List[str]]
        List of tokenized sentences, where each sentence is a list of tokens.
    target_words : Union[str, List[str]]
        Target word(s) to find collocates for.
    method : str, default='window'
        Method to use for calculating collocations. Either 'window' or 'sentence'.
        - 'window': Uses a sliding window of specified horizon around each token
        - 'sentence': Considers whole sentences as context units (horizon not applicable)
    horizon : Optional[Union[int, tuple]], default=None
        Context window size. Only applicable when method='window'. 
        Must be None when method='sentence'.
        - int: Symmetric window (e.g., 5 means 5 words on each side)
        - tuple: Asymmetric window (left, right) (e.g., (0, 5) means only 5 words on the right)
        - None: Uses default of 5 for 'window' method
    filters : Optional[FilterOptions], optional
        Dictionary of filters to apply to results, AFTER computation is done:
        - 'max_p': float - Maximum p-value threshold for statistical significance
        - 'stopwords': List[str] - Words to exclude from results
        - 'min_word_length': int - Minimum character length for collocates
        - 'min_exp_local': float - Minimum expected local frequency
        - 'max_exp_local': float - Maximum expected local frequency
        - 'min_obs_local': int - Minimum observed local frequency
        - 'max_obs_local': int - Maximum observed local frequency
        - 'min_ratio_local': float - Minimum local frequency ratio (obs/exp)
        - 'max_ratio_local': float - Maximum local frequency ratio (obs/exp)
        - 'min_obs_global': int - Minimum global frequency
        - 'max_obs_global': int - Maximum global frequency
    as_dataframe : bool, default=True
        If True, return results as a pandas DataFrame.
    max_sentence_length : Optional[int], default=256
        Maximum sentence length for preprocessing. Used by both 'window' and 'sentence' methods.
        Longer sentences will be truncated to avoid memory bloat from outliers. 
        Set to None for no limit (may use a lot of memory with very long sentences).
    alternative : str, default='greater'
        Alternative hypothesis for Fisher's exact test. Options are:
        - 'greater': Test if observed co-occurrence is greater than expected (default)
        - 'less': Test if observed co-occurrence is less than expected
        - 'two-sided': Test if observed co-occurrence differs from expected
    
    Returns:
    --------
    Union[List[Dict], pd.DataFrame]
        List of dictionaries or DataFrame containing collocation statistics.
    """
    if not sentences:
        raise ValueError("sentences cannot be empty")
    if not all(isinstance(s, list) for s in sentences):
        raise ValueError("sentences must be a list of lists (tokenized sentences)")
    
    # Filter out empty sentences
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("All sentences are empty")
    
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = list(set(target_words))
    
    if not target_words:
        raise ValueError("target_words cannot be empty")
    
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
                sentences, target_words, horizon=horizon, 
                max_sentence_length=max_sentence_length,
                alternative=alternative
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_cython(
                sentences, target_words, max_sentence_length=max_sentence_length,
                alternative=alternative
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")
    else:
        if method == 'window':
            results = _calculate_collocations_window(sentences, target_words, horizon=horizon, 
                                                    alternative=alternative)
        elif method == 'sentence':
            results = _calculate_collocations_sentence(
                sentences, target_words, alternative=alternative
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")

    if filters:
        valid_keys = {
            'max_p', 'stopwords', 'min_word_length', 'min_exp_local', 'max_exp_local',
            'min_obs_local', 'max_obs_local', 'min_ratio_local', 'max_ratio_local',
            'min_obs_global', 'max_obs_global'
        }
        invalid_keys = set(filters.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid filter keys: {invalid_keys}. Valid keys are: {valid_keys}")
        
        if 'max_p' in filters:
            max_p = filters['max_p']
            if not isinstance(max_p, (int, float)) or max_p < 0 or max_p > 1:
                raise ValueError("max_p must be a number between 0 and 1")
            results = [result for result in results if result["p_value"] <= max_p]
        
        if 'stopwords' in filters:
            stopwords = filters['stopwords']
            if not isinstance(stopwords, (list, set)):
                raise ValueError("stopwords must be a list or set of strings")
            stopwords_set = set(stopwords)
            results = [result for result in results if result["collocate"] not in stopwords_set]
        
        if 'min_word_length' in filters:
            min_word_length = filters['min_word_length']
            if not isinstance(min_word_length, int) or min_word_length < 1:
                raise ValueError("min_word_length must be a positive integer")
            results = [result for result in results if len(result["collocate"]) >= min_word_length]
        
        if 'min_exp_local' in filters:
            min_exp = filters['min_exp_local']
            if not isinstance(min_exp, (int, float)) or min_exp < 0:
                raise ValueError("min_exp_local must be a non-negative number")
            results = [result for result in results if result["exp_local"] >= min_exp]
        
        if 'max_exp_local' in filters:
            max_exp = filters['max_exp_local']
            if not isinstance(max_exp, (int, float)) or max_exp < 0:
                raise ValueError("max_exp_local must be a non-negative number")
            results = [result for result in results if result["exp_local"] <= max_exp]
        
        if 'min_obs_local' in filters:
            min_obs = filters['min_obs_local']
            if not isinstance(min_obs, int) or min_obs < 0:
                raise ValueError("min_obs_local must be a non-negative integer")
            results = [result for result in results if result["obs_local"] >= min_obs]
        
        if 'max_obs_local' in filters:
            max_obs = filters['max_obs_local']
            if not isinstance(max_obs, int) or max_obs < 0:
                raise ValueError("max_obs_local must be a non-negative integer")
            results = [result for result in results if result["obs_local"] <= max_obs]
        
        if 'min_ratio_local' in filters:
            min_ratio = filters['min_ratio_local']
            if not isinstance(min_ratio, (int, float)) or min_ratio < 0:
                raise ValueError("min_ratio_local must be a non-negative number")
            results = [result for result in results if result["ratio_local"] >= min_ratio]
        
        if 'max_ratio_local' in filters:
            max_ratio = filters['max_ratio_local']
            if not isinstance(max_ratio, (int, float)) or max_ratio < 0:
                raise ValueError("max_ratio_local must be a non-negative number")
            results = [result for result in results if result["ratio_local"] <= max_ratio]
        
        if 'min_obs_global' in filters:
            min_global = filters['min_obs_global']
            if not isinstance(min_global, int) or min_global < 0:
                raise ValueError("min_obs_global must be a non-negative integer")
            results = [result for result in results if result["obs_global"] >= min_global]
        
        if 'max_obs_global' in filters:
            max_global = filters['max_obs_global']
            if not isinstance(max_global, int) or max_global < 0:
                raise ValueError("max_obs_global must be a non-negative integer")
            results = [result for result in results if result["obs_global"] <= max_global]

    if as_dataframe:
        results = pd.DataFrame(results)
    return results

def cooc_matrix(
    documents: List[List[str]], 
    method: str = 'window', 
    horizon: Optional[Union[int, Tuple[int, int]]] = None, 
    min_abs_count: int = 1, 
    min_doc_count: int = 1, 
    vocab_size: Optional[int] = None, 
    binary: bool = False, 
    as_dataframe: bool = True, 
    vocab: Optional[Union[List[str], set]] = None, 
    use_sparse: bool = False
) -> Union[pd.DataFrame, Tuple[np.ndarray, Dict[str, int]]]:
    """
    Calculate a co-occurrence matrix from a list of documents.
    
    Parameters:
    -----------
    documents : list
        List of tokenized documents, where each document is a list of tokens.
    method : str, default='window'
        Method to use for calculating co-occurrences. Either 'window' or 'document'.
    horizon : Optional[Union[int, tuple]], default=None
        Context window size. Only applicable when method='window'.
        Must be None when method='document'.
        - int: Symmetric window (e.g., 5 means 5 words on each side)
        - tuple: Asymmetric window (left, right) (e.g., (0, 5) means only 5 words on the right)
        - None: Uses default of 5 for 'window' method
    min_abs_count : int, default=1
        Minimum absolute count for a word to be included in the vocabulary.
    min_doc_count : int, default=1
        Minimum number of documents a word must appear in to be included.
    vocab_size : int, optional
        Maximum size of the vocabulary. Words are sorted by frequency.
    binary : bool, default=False
        If True, count co-occurrences as binary (0/1) rather than frequencies.
    as_dataframe : bool, default=True
        If True, return the co-occurrence matrix as a pandas DataFrame.
    vocab : list or set, optional
        Predefined vocabulary to use. Words will still be filtered by min_abs_count and min_doc_count.
        If vocab_size is also provided, only the top vocab_size words will be kept.
    use_sparse : bool, default=False
        If True, use a sparse matrix representation for better memory efficiency with large vocabularies.
        
    Returns:
    --------
    If as_dataframe=True:
        pandas DataFrame with rows and columns labeled by vocabulary
    If as_dataframe=False and use_sparse=False:
        tuple of (numpy array, word_to_index dictionary)
    If as_dataframe=False and use_sparse=True:
        tuple of (scipy sparse matrix, word_to_index dictionary)
    """
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
    
    if use_sparse:
        from scipy import sparse
    
    word_counts = Counter()
    document_counts = Counter()
    for document in documents:
        word_counts.update(document)
        document_counts.update(set(document))
    
    filtered_vocab = {word for word, count in word_counts.items() 
                     if count >= min_abs_count and document_counts[word] >= min_doc_count}
    
    if vocab is not None:
        vocab = set(vocab)
        filtered_vocab = filtered_vocab.intersection(vocab)
    
    if vocab_size and len(filtered_vocab) > vocab_size:
        filtered_vocab = set(sorted(filtered_vocab, 
                                   key=lambda word: word_counts[word], 
                                   reverse=True)[:vocab_size])
    
    vocab_list = sorted(filtered_vocab)
    word_to_index = {word: i for i, word in enumerate(vocab_list)}
    
    filtered_documents = [[word for word in document if word in word_to_index] 
                         for document in documents]
    
    cooc_dict = defaultdict(int)

    def update_cooc(word1_idx, word2_idx, count=1):
        if binary:
            cooc_dict[(word1_idx, word2_idx)] = 1
        else:
            cooc_dict[(word1_idx, word2_idx)] += count

    if method == 'window':
        # Normalize horizon to (left, right) tuple
        if isinstance(horizon, int):
            left_horizon, right_horizon = horizon, horizon
        else:
            left_horizon, right_horizon = horizon
        
        for document in filtered_documents:
            for i, word1 in enumerate(document):
                idx1 = word_to_index[word1]
                start = max(0, i - left_horizon)
                end = min(len(document), i + right_horizon + 1)
                context_words = document[start:i] + document[i+1:end]

                for word2 in context_words:
                    idx2 = word_to_index[word2]
                    update_cooc(idx1, idx2, 1)

    elif method == 'document':
        for document in filtered_documents:
            doc_word_counts = Counter(document)
            unique_words = set(document)
            for word1 in unique_words:
                idx1 = word_to_index[word1]
                for word2 in unique_words:
                    if word2 != word1:
                        idx2 = word_to_index[word2]
                        update_cooc(idx1, idx2, doc_word_counts[word2])

    n = len(vocab_list)

    if use_sparse:
        row_indices, col_indices, data_values = zip(*((i, j, count) for (i, j), count in cooc_dict.items()))
        cooc_matrix_array = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(n, n)).tocsr()
    else:
        cooc_matrix_array = np.zeros((n, n))
        for (i, j), count in cooc_dict.items():
            cooc_matrix_array[i, j] = count
    
    del cooc_dict
    
    if as_dataframe:
        if use_sparse:
            # Warn user about converting sparse to dense
            import warnings
            warnings.warn(
                "Converting sparse matrix to dense DataFrame. This defeats the purpose of "
                "use_sparse=True and may cause memory issues for large vocabularies. "
                "Consider using as_dataframe=False to keep the sparse matrix, or "
                "use_sparse=False if you need a DataFrame.",
                UserWarning
            )
            cooc_matrix_df = pd.DataFrame(
                cooc_matrix_array.toarray(), 
                index=vocab_list, 
                columns=vocab_list
            )
        else:
            cooc_matrix_df = pd.DataFrame(
                cooc_matrix_array, 
                index=vocab_list, 
                columns=vocab_list
            )
        return cooc_matrix_df
    else:
        return cooc_matrix_array, word_to_index

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
    
    Parameters:
    -----------
    collocates : Union[List[Dict], pd.DataFrame]
        Output from find_collocates, either as a list of dictionaries or DataFrame.
    x_col : str, default='ratio_local'
        Column name to plot on x-axis. Common choices: 'ratio_local', 'obs_local',
        'exp_local', 'obs_global'.
    y_col : str, default='p_value'
        Column name to plot on y-axis. Common choices: 'p_value', 'obs_local',
        'ratio_local', 'obs_global'.
    x_scale : str, default='log'
        Scale for x-axis. Options: 'log', 'linear', 'symlog', 'logit'.
        For ratio_local, 'log' makes the scale symmetric around 1.
    y_scale : str, default='log'
        Scale for y-axis. Options: 'log', 'linear', 'symlog', 'logit'.
        For p_value, 'log' is recommended to visualize small values.
    color : Optional[Union[str, List[str]]], default=None
        Color(s) for the points. Can be a single color string, list of colors,
        or None to use default.
    colormap : str, default='viridis'
        Matplotlib colormap to use when color_by is specified.
    color_by : Optional[str], default=None
        Column name to use for coloring points (e.g., 'obs_local', 'obs_global').
    title : Optional[str], default=None
        Title for the plot.
    figsize : tuple, default=(10, 8)
        Figure size as (width, height) in inches.
    fontsize : int, default=10
        Base font size for labels.
    show_labels : bool, default=False
        Whether to show collocate text labels next to points.
    label_top_n : Optional[int], default=None
        If specified, only label the top N points. When color_by is set, ranks by that
        column; otherwise ranks by y-axis values. For p_value, labels smallest (most 
        significant) values; for other metrics, labels largest values.
    alpha : float, default=0.6
        Transparency of points (0.0 to 1.0).
    marker_size : int, default=50
        Size of markers.
    show_diagonal : bool, default=False
        Whether to draw a diagonal reference line (y=x). Useful for observed vs
        expected plots to show where values match perfectly.
    diagonal_color : str, default='red'
        Color of the diagonal reference line.
    filename : Optional[str], default=None
        If provided, saves the figure to the specified file path.
    xlabel : Optional[str], default=None
        Label for x-axis. If None, auto-generated from x_col and x_scale.
    ylabel : Optional[str], default=None
        Label for y-axis. If None, auto-generated from y_col and y_scale.
    
    Returns:
    --------
    None
        Displays the plot using matplotlib. To further customize, use plt.gca() 
        to get the current axes object after calling this function.
    
    Examples:
    ---------
    >>> # Basic usage: ratio vs p-value with log scales (default)
    >>> collocates = find_collocates(sentences, ['天'])
    >>> plot_collocates(collocates)
    
    >>> # Plot observed vs expected frequency
    >>> plot_collocates(collocates, x_col='exp_local', y_col='obs_local',
    ...                 x_scale='linear', y_scale='linear')
    
    >>> # Plot global frequency vs ratio with custom scales
    >>> plot_collocates(collocates, x_col='obs_global', y_col='ratio_local',
    ...                 x_scale='log', y_scale='log')
    
    >>> # With labels and custom styling
    >>> plot_collocates(collocates, show_labels=True, label_top_n=20,
    ...                 color='red', title='Collocates of 天')
    
    >>> # Color by a column
    >>> plot_collocates(collocates, color_by='obs_local', colormap='plasma')
    """
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