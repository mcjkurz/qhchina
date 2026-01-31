import logging

logger = logging.getLogger("qhchina.helpers.texts")


__all__ = [
    'detect_encoding',
    'load_text',
    'load_texts',
    'load_stopwords',
    'get_stopword_languages',
    'split_into_chunks',
]


def detect_encoding(filename, num_bytes=10000):
    """
    Detect the encoding of a text file automatically.
    
    Uses the chardet library to detect the character encoding of a file
    by analyzing a sample of bytes from the beginning.
    
    Args:
        filename (str): Path to the file to analyze.
        num_bytes (int): Number of bytes to read for detection (default: 10000).
            Larger values improve accuracy but slow down detection.
    
    Returns:
        str: The detected encoding (e.g., 'utf-8', 'gb18030', 'big5').
            Returns 'gb18030' for GB2312/GBK files as it's a superset.
    
    Raises:
        ImportError: If chardet is not installed.
    
    Example:
        >>> from qhchina.helpers import detect_encoding
        >>> encoding = detect_encoding("chinese_text.txt")
        >>> print(encoding)
        'utf-8'
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
    Load text content from a single file.
    
    Args:
        filename (str): Path to the text file.
        encoding (str): File encoding (default: "utf-8").
            Use "auto" to automatically detect the encoding using chardet.
    
    Returns:
        str: The complete text content of the file.
    
    Raises:
        ValueError: If filename is not a string.
        FileNotFoundError: If the file does not exist.
    
    Example:
        >>> from qhchina.helpers import load_text
        >>> text = load_text("novel.txt", encoding="utf-8")
        >>> # Or with auto-detection for unknown encodings
        >>> text = load_text("old_text.txt", encoding="auto")
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    if encoding == "auto":
        encoding = detect_encoding(filename)
    
    with open(filename, 'r', encoding=encoding) as file:
        return file.read()

def load_texts(filenames, encoding="utf-8"):
    """
    Load text content from multiple files.
    
    Args:
        filenames (list): List of file paths to load.
            Can also pass a single string for one file.
        encoding (str): File encoding (default: "utf-8").
            Use "auto" to detect encoding for each file individually.
    
    Returns:
        list: List of text strings, one per file, in the same order as filenames.
    
    Example:
        >>> from qhchina.helpers import load_texts
        >>> import glob
        >>> files = glob.glob("corpus/*.txt")
        >>> texts = load_texts(files)
        >>> print(f"Loaded {len(texts)} documents")
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    
    texts = []
    for filename in filenames:
        texts.append(load_text(filename, encoding))
    return texts

def load_stopwords(language: str = "zh_sim") -> set:
    """
    Load a stopword list for Chinese text processing.
    
    Provides pre-built stopword lists for different Chinese variants.
    These can be used with segmenters and other text processing tools.
    
    Args:
        language (str): Stopword list identifier. Available options:
            - 'zh_sim': Modern simplified Chinese (default)
            - 'zh_tr': Modern traditional Chinese
            - 'zh_cl_sim': Classical Chinese in simplified characters
            - 'zh_cl_tr': Classical Chinese in traditional characters
            Use get_stopword_languages() to see all available options.
    
    Returns:
        set: A set of stopword strings.
    
    Raises:
        ValueError: If the specified language is not available.
    
    Example:
        >>> from qhchina.helpers import load_stopwords
        >>> from qhchina.preprocessing import create_segmenter
        >>> 
        >>> stopwords = load_stopwords("zh_sim")
        >>> segmenter = create_segmenter(
        ...     backend="jieba",
        ...     filters={"stopwords": stopwords}
        ... )
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
    List all available stopword language codes.
    
    Returns:
        list: Sorted list of available language codes.
            Typical values include: 'zh_sim', 'zh_tr', 'zh_cl_sim', 'zh_cl_tr'
    
    Example:
        >>> from qhchina.helpers import get_stopword_languages
        >>> print(get_stopword_languages())
        ['zh_cl_sim', 'zh_cl_tr', 'zh_sim', 'zh_tr']
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
    Split a sequence into fixed-size chunks with optional overlap.
    
    Works with both strings (splits by character) and lists (splits by item).
    Useful for processing long texts that exceed model limits.
    
    Args:
        sequence (str or list): The text or token list to split.
        chunk_size (int): Maximum size of each chunk.
            For strings: number of characters.
            For lists: number of items.
        overlap (float): Fraction of overlap between consecutive chunks (0.0 to 1.0).
            Default is 0.0 (no overlap). Use overlap for context preservation
            when processing with models that need surrounding context.
    
    Returns:
        list: List of chunks (strings if input was string, lists if input was list).
            The last chunk may be smaller than chunk_size.
    
    Raises:
        ValueError: If overlap is not between 0.0 and 1.0.
    
    Example:
        >>> from qhchina.helpers import split_into_chunks
        >>> # Split text into 100-character chunks with 10% overlap
        >>> text = "这是一段很长的中文文本..." * 50
        >>> chunks = split_into_chunks(text, chunk_size=100, overlap=0.1)
        >>> 
        >>> # Split a token list
        >>> tokens = ["word1", "word2", "word3", "word4", "word5"]
        >>> chunks = split_into_chunks(tokens, chunk_size=2)
        >>> print(chunks)
        [['word1', 'word2'], ['word3', 'word4'], ['word5']]
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
