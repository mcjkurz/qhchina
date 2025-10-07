from collections import Counter, defaultdict
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, TypedDict

# Try to import Cython extensions for faster computation
try:
    from .cython_ext.collocations import calculate_window_counts, calculate_sentence_counts
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

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

def _build_vocabulary(tokenized_sentences):
    """Build vocabulary mapping for Cython implementation."""
    word2idx = {}
    idx = 0
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word

def _convert_sentences_to_indices(tokenized_sentences, word2idx, max_sentence_length=256):
    """
    Convert tokenized sentences to padded integer arrays for Cython.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        word2idx: Vocabulary mapping
        max_sentence_length: Maximum sentence length to pad to (default 256).
                           Longer sentences will be truncated to avoid memory bloat.
                           Set to None for no limit (may use a lot of memory).
    
    Returns:
        tuple: (sentences_indices, sentence_lengths)
    """
    n_sentences = len(tokenized_sentences)
    max_length = max(len(sent) for sent in tokenized_sentences) if tokenized_sentences else 0
    
    # Cap max_length to avoid memory bloat from outlier long sentences
    if max_sentence_length is not None:
        max_length = min(max_length, max_sentence_length)
    
    # Create padded array
    sentences_indices = np.zeros((n_sentences, max_length), dtype=np.int32)
    sentence_lengths = np.zeros(n_sentences, dtype=np.int32)
    
    for i, sentence in enumerate(tokenized_sentences):
        # Truncate if necessary
        actual_len = min(len(sentence), max_length)
        sentence_lengths[i] = actual_len
        for j in range(actual_len):
            sentences_indices[i, j] = word2idx[sentence[j]]
    
    return sentences_indices, sentence_lengths

def _calculate_collocations_window_fast(tokenized_sentences, target_words, horizon=5):
    """Fast Cython implementation of window-based collocation calculation."""
    # Build vocabulary
    word2idx, idx2word = _build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    
    # Convert sentences to indices (no length limit - window method processes locally)
    sentences_indices, sentence_lengths = _convert_sentences_to_indices(
        tokenized_sentences, word2idx, max_sentence_length=None
    )
    
    # Convert target words to indices (filter out targets not in vocabulary)
    target_indices = np.array([word2idx[w] for w in target_words if w in word2idx], dtype=np.int32)
    target_words_filtered = [w for w in target_words if w in word2idx]
    
    if len(target_indices) == 0:
        return []
    
    # Call Cython function
    T_count, candidate_counts, token_counter, total_tokens = calculate_window_counts(
        sentences_indices, sentence_lengths, target_indices, horizon, vocab_size
    )
    
    # Build results from the count matrices
    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        for candidate_idx in range(vocab_size):
            a = candidate_counts[t_idx, candidate_idx]
            if a == 0 or candidate_idx == target_word_idx:
                continue
            
            candidate = idx2word[candidate_idx]
            
            # Calculate contingency table
            b = T_count[t_idx] - a
            c = token_counter[candidate_idx] - a
            d = (total_tokens - token_counter[target_word_idx]) - (a + b + c)
            
            # Calculate expected and ratio
            expected = (a + b) * (a + c) / total_tokens if total_tokens > 0 else 0
            ratio = a / expected if expected > 0 else 0
            
            # Compute Fisher's exact test
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative='greater')[1]
            
            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": int(a),
                "ratio_local": ratio,
                "obs_global": int(token_counter[candidate_idx]),
                "p_value": p_value,
            })
    
    return results

def _calculate_collocations_window(tokenized_sentences, target_words, horizon=5):
    total_tokens = 0  # Total number of token positions in the corpus

    # For each target word:
    # T_count[target] counts how many token positions have the target in their context.
    T_count = {target: 0 for target in target_words}
    # candidate_in_context[target] counts, for each candidate word,
    # how many times it occurs in a token position whose window contains the target.
    candidate_in_context = {target: Counter() for target in target_words}
    # Global count: for each token, how many times does it occur (across all token positions).
    token_counter = Counter()

    # Loop over all sentences and token positions.
    for sentence in tqdm(tokenized_sentences):
        for i, token in enumerate(sentence):
            total_tokens += 1
            token_counter[token] += 1  # global count for this token

            # Define the window (context) for this token.
            start = max(0, i - horizon)
            end = min(len(sentence), i + horizon + 1)
            # Exclude the token itself.
            context = sentence[start:i] + sentence[i+1:end]

            # For each target, check if it is in this context.
            for target in target_words:
                if target in context:
                    T_count[target] += 1
                    candidate_in_context[target][token] += 1

    results = []

    # Now, for each target and for each candidate word that appeared
    # in positions where the target was in the context:
    for target in target_words:
        for candidate, a in candidate_in_context[target].items():
            if candidate == target:
                continue  # Skip if the candidate is the target itself.
                
            # the contingency table is:
            #               candidate | ~candidate
            # near (target)     [a,        b]
            # ~near (target)    [c,        d]

            # a: candidate appears in a token position whose context includes target.
            # this we already have from candidate_in_context[target][candidate]
            # b: positions with target in context where candidate did not appear.
            b = T_count[target] - a
            # c: candidate appears in a token position whose context does NOT include target.
            c = token_counter[candidate] - a
            # d: all other positions. We need to remove token_counter[target] from total_tokens
            # because it has never been included in the previous calculations.
            # a + b are all the positions where the target is in the context
            # or in other words, all positions surrounding the target (already excluded)
            # c is the number of times the candidate appears without the target in the context  
            # so d are the remaining positions
            d = (total_tokens - token_counter[target]) - (a + b + c)        
            # this is equivalent to:
            # d = total_tokens - T_count[target] - c - token_counter[target]

            # Calculate the expected frequency (if independent) and ratio.
            expected = (a + b) * (a + c) / total_tokens if total_tokens > 0 else 0
            ratio = a / expected if expected > 0 else 0

            # Compute Fisher's exact test.
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative='greater')[1]

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

def _calculate_collocations_sentence_fast(tokenized_sentences, target_words, max_sentence_length=256):
    """Fast Cython implementation of sentence-based collocation calculation."""
    # Build vocabulary
    word2idx, idx2word = _build_vocabulary(tokenized_sentences)
    vocab_size = len(word2idx)
    
    # Convert sentences to indices
    sentences_indices, sentence_lengths = _convert_sentences_to_indices(
        tokenized_sentences, word2idx, max_sentence_length
    )
    
    # Convert target words to indices
    target_indices = np.array([word2idx[w] for w in target_words if w in word2idx], dtype=np.int32)
    target_words_filtered = [w for w in target_words if w in word2idx]
    
    if len(target_indices) == 0:
        return []
    
    # Call Cython function
    candidate_sentences, sentences_with_token, total_sentences = calculate_sentence_counts(
        sentences_indices, sentence_lengths, target_indices, vocab_size
    )
    
    # Build results from the count matrices
    results = []
    for t_idx, target in enumerate(target_words_filtered):
        target_word_idx = target_indices[t_idx]
        for candidate_idx in range(vocab_size):
            a = candidate_sentences[t_idx, candidate_idx]
            if a == 0 or candidate_idx == target_word_idx:
                continue
            
            candidate = idx2word[candidate_idx]
            
            # Calculate contingency table
            b = sentences_with_token[target_word_idx] - a
            c = sentences_with_token[candidate_idx] - a
            d = total_sentences - a - b - c
            
            # Calculate expected and ratio
            expected = (a + b) * (a + c) / total_sentences if total_sentences > 0 else 0
            ratio = a / expected if expected > 0 else 0
            
            # Compute Fisher's exact test
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative='greater')[1]
            
            results.append({
                "target": target,
                "collocate": candidate,
                "exp_local": expected,
                "obs_local": int(a),
                "ratio_local": ratio,
                "obs_global": int(sentences_with_token[candidate_idx]),
                "p_value": p_value,
            })
    
    return results

def _calculate_collocations_sentence(tokenized_sentences, target_words, max_sentence_length=256):
    """
    Pure Python implementation of sentence-based collocation calculation.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        target_words: Set or list of target words
        max_sentence_length: Maximum sentence length to consider (default 256).
                           Longer sentences will be truncated for consistency with Cython.
                           Set to None for no limit.
    """
    total_sentences = len(tokenized_sentences)
    results = []
    candidate_in_sentences = {target: Counter() for target in target_words}
    sentences_with_token = defaultdict(int)

    for sentence in tqdm(tokenized_sentences):
        # Truncate sentence if needed for consistency with Cython implementation
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

            # Calculate the expected frequency (if independent) and ratio.
            expected = (a + b) * (a + c) / total_sentences if total_sentences > 0 else 0
            ratio = a / expected if expected > 0 else 0

            # Compute Fisher's exact test.
            table = np.array([[a, b], [c, d]])
            p_value = fisher_exact(table, alternative='greater')[1]

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
    max_sentence_length: Optional[int] = 256
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
        Maximum sentence length for preprocessing (only affects 'sentence' method).
        Longer sentences will be truncated to avoid memory bloat from outliers. 
        Set to None for no limit (may use a lot of memory if you have very long 
        sentences). The 'window' method processes tokens locally and does not use
        this parameter.
    
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
            # Window method doesn't use max_sentence_length (processes token-by-token)
            results = _calculate_collocations_window_fast(
                sentences, target_words, horizon=horizon
            )
        elif method == 'sentence':
            results = _calculate_collocations_sentence_fast(
                sentences, target_words, max_sentence_length=max_sentence_length
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")
    else:
        if method == 'window':
            results = _calculate_collocations_window(sentences, target_words, horizon=horizon)
        elif method == 'sentence':
            results = _calculate_collocations_sentence(
                sentences, target_words, max_sentence_length=max_sentence_length
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
    
    del cooc_dict # free memory
    
    # Return results based on parametersi
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