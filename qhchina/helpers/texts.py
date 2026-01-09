import logging

logger = logging.getLogger("qhchina.helpers.texts")


def detect_encoding(filename, num_bytes=10000):
    """
    Detects the encoding of a file.
    
    Parameters:
    filename (str): The path to the file.
    num_bytes (int): Number of bytes to read for detection. Default is 10000.
                     Larger values may be more accurate but slower.
    
    Returns:
    str: The detected encoding (e.g., 'utf-8', 'gb2312', 'gbk', 'big5').
    
    Raises:
    ImportError: If chardet is not installed.
    """
    try:
        import chardet
    except ImportError:
        raise ImportError(
            "The 'chardet' package is required for automatic encoding detection. "
            "Install it with: pip install chardet"
        )
    
    with open(filename, 'rb') as file:
        raw_data = file.read(num_bytes)
    
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    
    # Handle common encoding aliases and edge cases
    if encoding is None:
        return 'utf-8'  # Fallback to UTF-8
    
    # Normalize some common Chinese encoding names
    encoding_lower = encoding.lower()
    if encoding_lower in ('gb2312', 'gbk', 'gb18030'):
        # Use gb18030 as it's a superset of GB2312 and GBK
        return 'gb18030'
    
    return encoding


def load_text(filename, encoding="utf-8"):
    """
    Loads text from a file.

    Parameters:
    filename (str): The filename to load text from.
    encoding (str): The encoding of the file. Default is "utf-8".
                    Use "auto" to automatically detect the encoding.
    
    Returns:
    str: The text content of the file.
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    if encoding == "auto":
        encoding = detect_encoding(filename)
    
    with open(filename, 'r', encoding=encoding) as file:
        return file.read()

def load_texts(filenames, encoding="utf-8"):
    """
    Loads text from multiple files.

    Parameters:
    filenames (list): A list of filenames to load text from.
    encoding (str): The encoding of the files. Default is "utf-8".
                    Use "auto" to automatically detect encoding for each file.
    
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
        language: Language code (default: "zh_sim" for simplified Chinese).
                  Use get_stopword_languages() to see available options.
    
    Returns:
        Set of stopwords
    
    Raises:
        ValueError: If the specified language is not available.
    """
    import os
    
    # Get the current file's directory (helpers)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the qhchina package root and construct the path to stopwords
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    stopwords_dir = os.path.join(package_root, 'data', 'stopwords')
    stopwords_path = os.path.join(stopwords_dir, f'{language}.txt')
    
    # Load stopwords from file
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        return stopwords
    except FileNotFoundError:
        # Get available stopword languages
        available = []
        try:
            files = os.listdir(stopwords_dir)
            available = sorted([f[:-4] for f in files if f.endswith('.txt')])
        except FileNotFoundError:
            pass
        
        raise ValueError(
            f"Stopwords file not found for language '{language}'. "
            f"Available options: {available}. "
            f"Note: Do not include the file extension (use 'zh_sim' not 'zh_sim.txt')."
        )

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
        logger.warning(f"Stopwords directory not found at path {stopwords_dir}")
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
         Note: The last chunk may be smaller than chunk_size if the sequence
         doesn't divide evenly.
    
    Raises:
    ValueError: If overlap is not between 0 and 1.
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    if not sequence:
        return []
    
    # Handle case where sequence is shorter than or equal to chunk_size
    if len(sequence) <= chunk_size:
        return [sequence]
    
    overlap_size = int(chunk_size * overlap)
    stride = max(1, chunk_size - overlap_size)  # Ensure stride is at least 1
    
    chunks = []
    i = 0
    while i < len(sequence):
        end = i + chunk_size
        if end >= len(sequence):
            # Last chunk - include all remaining elements
            chunks.append(sequence[i:])
            break
        else:
            chunks.append(sequence[i:end])
            i += stride
        
    return chunks
