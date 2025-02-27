from collections import Counter, defaultdict
from scipy.stats import fisher_exact
import numpy as np

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

def find_collocations(tokenized_sentences, target_words, method='window', horizon=5, stopwords=None, as_dataframe=False):
    if not isinstance(target_words, list):
        target_words = [target_words]
    target_words = set(target_words)

    if method == 'window':
        results = _calculate_collocations_window(tokenized_sentences, target_words, horizon=horizon)
    elif method == 'sentence':
        results = _calculate_collocations_sentence(tokenized_sentences, target_words)
    else:
        raise NotImplementedError(f"The method {method} is not implemented.")

    if stopwords:
        results = [result for result in results if result["collocate"] not in stopwords]

    if as_dataframe:
        import pandas
        results = pandas.DataFrame(results)
    return results

def cooc_matrix(documents, method='window', horizon=5, min_abs_count=1, min_doc_count=1, vocab_size=None, binary=False, as_dataframe=False):
    if method not in ('window', 'document'):
        raise ValueError("method must be 'window' or 'document'")

    word_counts = Counter()
    document_counts = Counter()
    for document in documents:
        word_counts.update(document)
        document_counts.update(set(document))

    # Create the list of words to keep.
    keep_words = {word for word, count in word_counts.items() if count >= min_abs_count and document_counts[word] >= min_doc_count}
    if vocab_size:
        keep_words = set(sorted(keep_words, key=lambda word: word_counts[word], reverse=True)[:vocab_size])
            
    # Filter each sentence
    filtered_documents = [[word for word in document if word in keep_words] for document in documents]
    
    # 2.  Co-occurrence Calculation
    cooc_counts = defaultdict(lambda: defaultdict(int))

    if method == 'window':
        for document in filtered_documents:
            for i, word1 in enumerate(document):
                start = max(0, i - horizon)
                end = min(len(document), i + horizon + 1)
                for word2 in document[start:i] + document[i+1:end]:
                    if binary:
                        cooc_counts[word1][word2] = 1
                    else:
                        cooc_counts[word1][word2] += 1
                    
    elif method == 'document':
        for document in filtered_documents:
            for i in range(len(document)):
                word1 = document[i]
                context = document[:i] + document[i+1:]
                for word2 in context:
                    if binary:
                        cooc_counts[word1][word2] = 1
                    else:
                        cooc_counts[word1][word2] += 1
    
    all_words = sorted(cooc_counts.keys())
    word_to_index = {word:i for i,word in enumerate(all_words)}
    n = len(all_words)

    cooc_matrix = np.zeros((n,n),dtype=int)
    for word1, inner_dict in cooc_counts.items():
        for word2, count in inner_dict.items():
            cooc_matrix[word_to_index[word1],word_to_index[word2]] = count

    if as_dataframe:
        import pandas
        cooc_matrix_df = pandas.DataFrame(cooc_matrix, index=all_words, columns=all_words)
        return cooc_matrix_df
    else:
        return cooc_matrix, word_to_index