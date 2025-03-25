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