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
    
def split_into_chunks(sequence, chunk_size, overlap=0.0):
    """
    Splits text or a list of tokens into chunks with optional overlap between consecutive chunks.
    
    Parameters:
    sequence (str or list): The text string or list of tokens to be split.
    chunk_size (int): The size of each chunk (characters for text, items for lists).
    overlap (float): The fraction of overlap between consecutive chunks (0.0 to 1.0).
                    Default is 0.0 (no overlap).
    
    Returns:
    list: A list of chunks. If input is a string, each chunk is a string.
         If input is a list, each chunk is a list of tokens.
    
    Raises:
    ValueError: If overlap is not between 0 and 1.
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    if not sequence:
        return []
        
    overlap_size = int(chunk_size * overlap)
    stride = chunk_size - overlap_size
    
    chunks = []
    for i in range(0, len(sequence) - chunk_size + 1, stride):
        chunks.append(sequence[i:i + chunk_size])
    
    # Handle the last chunk if there are remaining tokens/characters
    if i + chunk_size < len(sequence):
        chunks.append(sequence[-chunk_size:])
        
    return chunks
