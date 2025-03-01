from collections import Counter, defaultdict
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd

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
    for sentence in tokenized_sentences:
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

            # a: candidate appears in a token position whose context includes target.
            # c: candidate appears in a token position whose context does NOT include target.
            c = token_counter[candidate] - a
            # b: positions with target in context where candidate did not appear.
            b = T_count[target] - a
            # d: all other positions.
            d = total_tokens - T_count[target] - c - token_counter[target]

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
                "ratio_local": round(ratio, 2),
                "obs_global": token_counter[candidate],
                "p_value": p_value,
            })

    return results

def _calculate_collocations_sentence(tokenized_sentences, target_words):
    total_sentences = len(tokenized_sentences)
    results = []
    candidate_in_sentences = {target: Counter() for target in target_words}
    sentences_with_token = defaultdict(int)

    for sentence in tokenized_sentences:
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
                "ratio_local": round(ratio, 2),
                "obs_global": sentences_with_token[candidate],
                "p_value": p_value,
            })

    return results

def find_collocates(sentences, target_words, method='window', horizon=5, stopwords=None, as_dataframe=False):
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = set(target_words)

    if method == 'window':
        results = _calculate_collocations_window(sentences, target_words, horizon=horizon)
    elif method == 'sentence':
        results = _calculate_collocations_sentence(sentences, target_words)
    else:
        raise NotImplementedError(f"The method {method} is not implemented.")

    if stopwords:
        results = [result for result in results if result["collocate"] not in stopwords]

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