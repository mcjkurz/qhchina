from collections import Counter, defaultdict
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, TypedDict

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
                                         max_sentence_length=256, batch_size=10000, alternative='greater'):
    """
    Cython implementation of window-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        horizon: Window size on each side
        max_sentence_length: Maximum sentence length to consider (default 256)
        batch_size: Number of sentences to process per batch (default 10000)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    result = calculate_collocations_window(tokenized_sentences, target_words, horizon, 
                                           max_sentence_length, batch_size)
    
    T_count_total, candidate_counts_total, token_counter_total, total_tokens, word2idx, idx2word, target_indices = result
    
    if T_count_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []
    vocab_size = len(word2idx)
    
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
            
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative=alternative)[1]
            
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
    """Pure Python implementation of window-based collocation calculation."""
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

    for target in target_words:
        for candidate, a in candidate_in_context[target].items():
            if candidate == target:
                continue
                
            b = T_count[target] - a
            c = token_counter[candidate] - a
            d = (total_tokens - token_counter[target]) - (a + b + c)

            expected = (a + b) * (a + c) / total_tokens if total_tokens > 0 else 0
            ratio = a / expected if expected > 0 else 0

            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative=alternative)[1]

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

def _calculate_collocations_sentence_cython(tokenized_sentences, target_words, 
                                           max_sentence_length=256, batch_size=10000, alternative='greater'):
    """
    Cython implementation of sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: List or set of target words
        max_sentence_length: Maximum sentence length to consider (default 256)
        batch_size: Number of sentences to process per batch (default 10000)
        alternative: Alternative hypothesis for Fisher's exact test (default 'greater')
    
    Returns:
        List of dictionaries with collocation statistics
    """
    result = calculate_collocations_sentence(tokenized_sentences, target_words, 
                                             max_sentence_length, batch_size)
    
    candidate_sentences_total, sentences_with_token_total, total_sentences, word2idx, idx2word, target_indices = result
    
    if candidate_sentences_total is None:
        return []
    
    target_words_filtered = [idx2word[int(idx)] for idx in target_indices] if len(target_indices) > 0 else []
    vocab_size = len(word2idx)
    
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
            
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative=alternative)[1]
            
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
    """
    Pure Python implementation of sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: Set or list of target words
        max_sentence_length: Maximum sentence length to consider (default 256)
    """
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

    for target in target_words:
        for candidate, a in candidate_in_sentences[target].items():
            if candidate == target:
                continue
            b = sentences_with_token[target] - a
            c = sentences_with_token[candidate] - a
            d = total_sentences - a - b - c

            expected = (a + b) * (a + c) / total_sentences if total_sentences > 0 else 0
            ratio = a / expected if expected > 0 else 0

            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative=alternative)[1]

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
    batch_size: int = 10000,
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
        Maximum sentence length for preprocessing. Longer sentences will be 
        truncated to avoid memory bloat from outliers. Set to None for no limit 
        (may use a lot of memory if you have very long sentences).
    batch_size : int, default=10000
        Number of sentences to process in each batch when using Cython 
        implementation. Larger batches are faster but use more memory. 
        Adjust based on available RAM and corpus characteristics.
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
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = list(set(target_words))

    # Use Cython implementation if available, otherwise fall back to pure Python
    if CYTHON_AVAILABLE:
        if method == 'window':
            results = _calculate_collocations_window_cython(
                sentences, target_words, horizon=horizon, 
                max_sentence_length=max_sentence_length, batch_size=batch_size,
                alternative=alternative
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_cython(
                sentences, target_words, max_sentence_length=max_sentence_length,
                batch_size=batch_size, alternative=alternative
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")
    else:
        if method == 'window':
            results = _calculate_collocations_window(sentences, target_words, horizon=horizon, 
                                                    alternative=alternative)
        elif method == 'sentence':
            results = _calculate_collocations_sentence(
                sentences, target_words, max_sentence_length=max_sentence_length,
                alternative=alternative
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")

    # Apply filters if specified
    if filters:
        # Validate filter keys
        valid_keys = {
            'max_p', 'stopwords', 'min_length', 'min_exp_local', 'max_exp_local',
            'min_obs_local', 'max_obs_local', 'min_ratio_local', 'max_ratio_local',
            'min_obs_global', 'max_obs_global'
        }
        invalid_keys = set(filters.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid filter keys: {invalid_keys}. Valid keys are: {valid_keys}")
        
        # Filter by p-value threshold
        if 'max_p' in filters:
            max_p = filters['max_p']
            if not isinstance(max_p, (int, float)) or max_p < 0 or max_p > 1:
                raise ValueError("max_p must be a number between 0 and 1")
            results = [result for result in results if result["p_value"] <= max_p]
        
        # Filter out stopwords
        if 'stopwords' in filters:
            stopwords = filters['stopwords']
            if not isinstance(stopwords, (list, set)):
                raise ValueError("stopwords must be a list or set of strings")
            stopwords_set = set(stopwords)
            results = [result for result in results if result["collocate"] not in stopwords_set]
        
        # Filter by minimum length
        if 'min_length' in filters:
            min_length = filters['min_length']
            if not isinstance(min_length, int) or min_length < 1:
                raise ValueError("min_length must be a positive integer")
            results = [result for result in results if len(result["collocate"]) >= min_length]
        
        # Filter by expected local frequency
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
        
        # Filter by observed local frequency
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
        
        # Filter by local frequency ratio
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
        
        # Filter by global frequency
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
    if method not in ('window', 'document'):
        raise ValueError("method must be 'window' or 'document'")
    
    # Import scipy sparse matrix if needed
    if use_sparse:
        from scipy import sparse
    
    # Calculate word counts for all documents
    word_counts = Counter()
    document_counts = Counter()
    for document in documents:
        word_counts.update(document)
        document_counts.update(set(document))
    
    # Filter words by minimum counts
    filtered_vocab = {word for word, count in word_counts.items() 
                     if count >= min_abs_count and document_counts[word] >= min_doc_count}
    
    # If vocab is provided, intersect with filtered_vocab
    if vocab is not None:
        vocab = set(vocab)
        filtered_vocab = filtered_vocab.intersection(vocab)
    
    # If vocab_size is provided, select the top vocab_size words
    if vocab_size and len(filtered_vocab) > vocab_size:
        filtered_vocab = set(sorted(filtered_vocab, 
                                   key=lambda word: word_counts[word], 
                                   reverse=True)[:vocab_size])
    
    # Create vocabulary list and mapping
    vocab_list = sorted(filtered_vocab)
    word_to_index = {word: i for i, word in enumerate(vocab_list)}
    
    # Filter documents to only include words in the final vocabulary
    filtered_documents = [[word for word in document if word in word_to_index] 
                         for document in documents]
    
    # Initialize co-occurrence dictionary for sparse matrix
    cooc_dict = defaultdict(int)

    # Function to update co-occurrence counts
    def update_cooc(word1_idx, word2_idx, count=1):
        if binary:
            cooc_dict[(word1_idx, word2_idx)] = 1
        else:
            cooc_dict[(word1_idx, word2_idx)] += count

    # Calculate co-occurrences
    if method == 'window':
        for document in filtered_documents:
            for i, word1 in enumerate(document):
                idx1 = word_to_index[word1]
                start = max(0, i - horizon)
                end = min(len(document), i + horizon + 1)

                # Get context words (excluding the word itself)
                context_words = document[start:i] + document[i+1:end]

                # Update co-occurrence counts for each context word
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

    # Define the size of the vocabulary
    n = len(vocab_list)

    # Convert co-occurrence dictionary to sparse matrix
    if use_sparse:
        row_indices, col_indices, data_values = zip(*((i, j, count) for (i, j), count in cooc_dict.items()))
        cooc_matrix_array = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(n, n)).tocsr()
    else:
        # Create a dense matrix for non-sparse case
        cooc_matrix_array = np.zeros((n, n))
        for (i, j), count in cooc_dict.items():
            cooc_matrix_array[i, j] = count
    
    del cooc_dict
    
    # Return results based on parameters
    if as_dataframe:
        if use_sparse:
            # Convert sparse matrix to dense for DataFrame
            # Note: This could be memory-intensive for very large matrices
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