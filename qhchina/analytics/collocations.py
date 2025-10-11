from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, TypedDict
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact as scipy_fisher_exact

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
    min_length: int
    min_exp_local: float
    max_exp_local: float
    min_obs_local: int
    max_obs_local: int
    min_ratio_local: float
    max_ratio_local: float
    min_obs_global: int
    max_obs_global: int

def _calculate_collocations_window_cython(tokenized_sentences, target_words, horizon=5, 
                                         max_sentence_length=256, alternative='greater'):
    """
    Cython implementation of window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        horizon: Window size on each side
        max_sentence_length: Maximum sentence length to consider (default 256)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    result = calculate_collocations_window(tokenized_sentences, target_words, horizon, 
                                           max_sentence_length)
    
    T_count_total, candidate_counts_total, token_counter_total, total_tokens, word2idx, idx2word, target_indices = result
    
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
            
            expected = (a + b) * (a + c) / total_tokens if total_tokens > 0 else 0
            ratio = a / expected if expected > 0 else 0
            
            table[:] = [[a, b], [c, d]]
            _, p_value = scipy_fisher_exact(table, alternative=alternative)

            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": int(a),
                "ratio_local": ratio,
                "obs_global": int(token_counter_total[candidate_idx]),
                "p_value": p_value,
            })
    
    return results

def _calculate_collocations_window(tokenized_sentences, target_words, horizon=5, alternative='greater'):
    """Window-based collocation calculation."""
    total_tokens = 0
    T_count = {target: 0 for target in target_words}
    candidate_in_context = {target: Counter() for target in target_words}
    token_counter = Counter()

    for sentence in tqdm(tokenized_sentences):
        for i, token in enumerate(sentence):
            total_tokens += 1
            token_counter[token] += 1

            start = max(0, i - horizon)
            end = min(len(sentence), i + horizon + 1)
            context = sentence[start:i] + sentence[i+1:end]

            for target in target_words:
                if target in context:
                    T_count[target] += 1
                    candidate_in_context[target][token] += 1

    results = []
    
    table = np.zeros((2, 2), dtype=np.int64)

    for target in target_words:
        for candidate, a in candidate_in_context[target].items():
            if candidate == target:
                continue
                
            b = T_count[target] - a
            c = token_counter[candidate] - a
            d = (total_tokens - token_counter[target]) - (a + b + c)

            expected = (a + b) * (a + c) / total_tokens if total_tokens > 0 else 0
            ratio = a / expected if expected > 0 else 0

            # Reuse array - update in-place
            table[:] = [[a, b], [c, d]]
            _, p_value = scipy_fisher_exact(table, alternative=alternative)

            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": a,
                "ratio_local": ratio,
                "obs_global": token_counter[candidate],
                "p_value": p_value,
            })

    return results

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
    result = calculate_collocations_sentence(tokenized_sentences, target_words, max_sentence_length)
    
    candidate_sentences_total, sentences_with_token_total, total_sentences, word2idx, idx2word, target_indices = result
    
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
            
            expected = (a + b) * (a + c) / total_sentences if total_sentences > 0 else 0
            ratio = a / expected if expected > 0 else 0
            
            table[:] = [[a, b], [c, d]]
            _, p_value = scipy_fisher_exact(table, alternative=alternative)

            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": int(a),
                "ratio_local": ratio,
                "obs_global": int(sentences_with_token_total[candidate_idx]),
                "p_value": p_value,
            })
    
    return results

def _calculate_collocations_sentence(tokenized_sentences, target_words, max_sentence_length=256, alternative='greater'):
    """Sentence-based collocation calculation."""
    total_sentences = len(tokenized_sentences)
    results = []
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

    # Create reusable table array (avoid creating new array in each iteration)
    table = np.zeros((2, 2), dtype=np.int64)

    for target in target_words:
        for candidate, a in candidate_in_sentences[target].items():
            if candidate == target:
                continue
            b = sentences_with_token[target] - a
            c = sentences_with_token[candidate] - a
            d = total_sentences - a - b - c

            expected = (a + b) * (a + c) / total_sentences if total_sentences > 0 else 0
            ratio = a / expected if expected > 0 else 0

            # Reuse array - update in-place
            table[:] = [[a, b], [c, d]]
            _, p_value = scipy_fisher_exact(table, alternative=alternative)

            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": a,
                "ratio_local": ratio,
                "obs_global": sentences_with_token[candidate],
                "p_value": p_value,
            })

    return results

def find_collocates(
    sentences: List[List[str]], 
    target_words: Union[str, List[str]], 
    method: str = 'window', 
    horizon: int = 5, 
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
        - 'sentence': Considers whole sentences as context units
    horizon : int, default=5
        Size of the context window on each side (only used if method='window').
    filters : Optional[FilterOptions], optional
        Dictionary of filters to apply to results, AFTER computation is done:
        - 'max_p': float - Maximum p-value threshold for statistical significance
        - 'stopwords': List[str] - Words to exclude from results
        - 'min_length': int - Minimum character length for collocates
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
    
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = list(set(target_words))
    
    if not target_words:
        raise ValueError("target_words cannot be empty")

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
            'max_p', 'stopwords', 'min_length', 'min_exp_local', 'max_exp_local',
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
        
        if 'min_length' in filters:
            min_length = filters['min_length']
            if not isinstance(min_length, int) or min_length < 1:
                raise ValueError("min_length must be a positive integer")
            results = [result for result in results if len(result["collocate"]) >= min_length]
        
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

def cooc_matrix(documents, method='window', horizon=5, min_abs_count=1, min_doc_count=1, 
                vocab_size=None, binary=False, as_dataframe=True, vocab=None, use_sparse=False):
    """
    Calculate a co-occurrence matrix from a list of documents.
    
    Parameters:
    -----------
    documents : list
        List of tokenized documents, where each document is a list of tokens.
    method : str, default='window'
        Method to use for calculating co-occurrences. Either 'window' or 'document'.
    horizon : int, default=5
        Size of the context window (only used if method='window').
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
        for document in filtered_documents:
            for i, word1 in enumerate(document):
                idx1 = word_to_index[word1]
                start = max(0, i - horizon)
                end = min(len(document), i + horizon + 1)
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
            # Note: Converting sparse to dense could be memory-intensive for large matrices
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
            print(f"Warning: {zero_or_neg_x} values in {x_col} are ≤ 0. Replacing with 1e-300 for log scale.")
            x = np.where(x <= 0, 1e-300, x)
    
    if y_scale == 'log':
        zero_or_neg_y = (y <= 0).sum()
        if zero_or_neg_y > 0:
            print(f"Warning: {zero_or_neg_y} values in {y_col} are ≤ 0. Replacing with 1e-300 for log scale.")
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