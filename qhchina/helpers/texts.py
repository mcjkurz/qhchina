def load_text(filename, encoding="utf-8"):
    """
    Loads text from a file.

    Parameters:
    filename (str): The filename to load text from.
    encoding (str): The encoding of the file. Default is "utf-8".
    
    Returns:
    str: The text content of the file.
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    with open(filename, 'r', encoding=encoding) as file:
        return file.read()

def load_texts(filenames, encoding="utf-8"):
    """
    Loads text from multiple files.

    Parameters:
    filenames (list): A list of filenames to load text from.
    encoding (str): The encoding of the files. Default is "utf-8".
    
    Returns:
    list: A list of text contents from the files.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    
    texts = []
    for filename in filenames:
        texts.append(load_text(filename, encoding))
    return texts

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

def get_stopword_languages() -> list:
    """
    Get all available stopword language codes.
    
    Returns:
        List of available language codes (e.g., ['zh_sim', 'zh_cl_sim', 'zh_cl_tr'])
    """
    import os
    
    # Get the current file's directory (helpers)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the qhchina package root and construct the path to stopwords
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    stopwords_dir = os.path.join(package_root, 'data', 'stopwords')
    
    # List all .txt files in the stopwords directory
    try:
        files = os.listdir(stopwords_dir)
        # Filter for .txt files and remove the extension
        stopword_lists = [f[:-4] for f in files if f.endswith('.txt')]
        return sorted(stopword_lists)
    except FileNotFoundError:
        print(f"Warning: Stopwords directory not found at path {stopwords_dir}")
        return []
    
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
