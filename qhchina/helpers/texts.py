def load_texts(filenames):
    """
    Loads text from a file or a list of files.

    Parameters:
    filenames (str or list): The filename or list of filenames to load text from.

    Returns:
    str or list: The text content of the file or a list of text contents if multiple files are provided.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    texts = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def sample_sentences_to_token_count(corpus, target_tokens):
    """
    Samples sentences from a corpus until the target token count is reached.
    
    This function randomly selects sentences from the corpus until the total number
    of tokens reaches or slightly exceeds the target count. This is useful for balancing
    corpus sizes when comparing different time periods or domains.
    
    Parameters:
    -----------
    corpus : List[List[str]]
        A list of sentences, where each sentence is a list of tokens
    target_tokens : int
        The target number of tokens to sample
        
    Returns:
    --------
    List[List[str]]
        A list of sampled sentences with token count close to target_tokens
    """
    import random
    
    sampled_sentences = []
    current_tokens = 0
    sentence_indices = list(range(len(corpus)))
    random.shuffle(sentence_indices)
    
    for idx in sentence_indices:
        sentence = corpus[idx]
        if current_tokens + len(sentence) <= target_tokens:
            sampled_sentences.append(sentence)
            current_tokens += len(sentence)
        if current_tokens >= target_tokens:
            break
    return sampled_sentences

def add_corpus_tags(corpora, labels, target_words):
    """
    Add corpus-specific tags to target words in all corpora at once.
    
    Args:
        corpora: List of corpora (each corpus is list of tokenized sentences)
        labels: List of corpus labels
        target_words: List of words to tag
    
    Returns:
        List of processed corpora where target words have been tagged with their corpus label
    """
    processed_corpora = []
    target_words_set = set(target_words)
    
    for corpus, label in zip(corpora, labels):
        processed_corpus = []
        for sentence in corpus:
            processed_sentence = []
            for token in sentence:
                if token in target_words_set:
                    processed_sentence.append(f"{token}_{label}")
                else:
                    processed_sentence.append(token)
            processed_corpus.append(processed_sentence)
        processed_corpora.append(processed_corpus)
    
    return processed_corpora

def load_stopwords(language: str = "zh_sim") -> set:
    """
    Load stopwords from a file for the specified language.
    
    Args:
        language: Language code (default: "zh_sim" for simplified Chinese)
    
    Returns:
        Set of stopwords
    """
    import os
    
    # Get the current file's directory (helpers)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the qhchina package root and construct the path to stopwords
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    stopwords_path = os.path.join(package_root, 'data', 'stopwords', f'{language}.txt')
    
    # Load stopwords from file
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        return stopwords
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found for language '{language}' at path {stopwords_path}")
        return set()