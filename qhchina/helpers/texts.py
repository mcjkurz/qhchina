import logging
from pathlib import Path

logger = logging.getLogger("qhchina.helpers.texts")


__all__ = [
    'detect_encoding',
    'load_text',
    'load_texts',
    'load_stopwords',
    'get_stopword_languages',
    'split_into_chunks',
    'download_corpus',
    'download_file',
    'list_remote_corpora',
]


def detect_encoding(filename, num_bytes=10000):
    """
    Detects the encoding of a file.
    
    Args:
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

    Args:
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

    Args:
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
    
    Supports prefix matching: if the language code doesn't match an exact file,
    all files starting with that prefix will be loaded and combined.
    
    Args:
        language: Language code or prefix (default: "zh_sim" for simplified Chinese).
                  - Exact match: "zh_sim" loads zh_sim.txt only
                  - Prefix match: "zh" loads all files starting with "zh" (zh_sim, zh_tr, zh_cl_sim, zh_cl_tr)
                  - Prefix match: "zh_cl" loads zh_cl_sim.txt and zh_cl_tr.txt
                  Use get_stopword_languages() to see available options.
    
    Returns:
        Set of stopwords (combined from all matching files)
    
    Raises:
        ValueError: If no matching stopwords files are found.
    """
    import os
    
    # Get the current file's directory (helpers)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the qhchina package root and construct the path to stopwords
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    stopwords_dir = os.path.join(package_root, 'data', 'stopwords')
    
    # Get all available stopword files
    try:
        all_files = [f for f in os.listdir(stopwords_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        raise ValueError(f"Stopwords directory not found: {stopwords_dir}")
    
    # Check for exact match first
    exact_match = f'{language}.txt'
    if exact_match in all_files:
        matching_files = [exact_match]
    else:
        # Prefix matching: find all files starting with the language prefix
        # Use underscore or end-of-name to ensure proper prefix matching
        # e.g., "zh" matches "zh_sim.txt" but "zh_c" matches "zh_cl_sim.txt"
        matching_files = [
            f for f in all_files 
            if f.startswith(language) and (
                f == f'{language}.txt' or 
                f[len(language)] == '_'
            )
        ]
    
    if not matching_files:
        available = sorted([f[:-4] for f in all_files])
        raise ValueError(
            f"No stopwords files found matching '{language}'. "
            f"Available options: {available}. "
            f"Note: Use exact names like 'zh_sim' or prefixes like 'zh' to load multiple files."
        )
    
    # Load and combine stopwords from all matching files
    stopwords = set()
    for filename in matching_files:
        filepath = os.path.join(stopwords_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords.update(line.strip() for line in f if line.strip())
    
    return stopwords

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
    
    Args:
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
        ValueError: If overlap is not between 0 and 1, or if chunk_size is not positive.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
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

def _get_requests():
    """Lazy import of requests module."""
    try:
        import requests
        return requests
    except ImportError as e:
        raise ImportError(
            "requests is required for downloading from GitHub. "
            "Install it with: pip install requests"
        ) from e


def download_corpus(name: str, parent_dir: str | None = None) -> Path:
    """
    Download a corpus folder from the qhchina-data GitHub repository.
    
    Downloads all .txt files from the specified corpus folder and saves them
    to a local directory.
    
    Args:
        name: Corpus name (e.g., "张爱玲", "songshi"). This corresponds to a 
            folder name under ``corpora/`` in the qhchina-data repository.
        parent_dir: Parent directory where the corpus folder will be created.
            If None (default), uses the current working directory.
            
    Returns:
        Path to the downloaded corpus folder.
        
    Raises:
        ImportError: If requests is not installed.
        ValueError: If the corpus is not found or contains no .txt files.
        requests.RequestException: If the download fails.
        
    Example:
        >>> from qhchina import download_corpus
        >>> 
        >>> # Download to current directory
        >>> path = download_corpus("张爱玲")
        >>> # Creates ./张爱玲/张爱玲_倾城之恋.txt, ./张爱玲/张爱玲_金锁记.txt, ...
        >>> 
        >>> # Download to a specific parent directory
        >>> path = download_corpus("张爱玲", parent_dir="corpora")
        >>> # Creates ./corpora/张爱玲/...
    """
    from .github import query_github_api, download_file as _download_file
    
    # Determine output directory
    if parent_dir is None:
        output_dir = Path.cwd() / name
    else:
        output_dir = Path(parent_dir) / name
    
    # Query GitHub API for corpus contents
    api_path = f"corpora/{name}"
    try:
        contents = query_github_api(api_path)
    except Exception as e:
        available = list_remote_corpora()
        raise ValueError(
            f"Corpus '{name}' not found. Available corpora: {available}"
        ) from e
    
    # Filter for .txt files
    txt_files = [
        item for item in contents 
        if item['type'] == 'file' and item['name'].endswith('.txt')
    ]
    
    if not txt_files:
        raise ValueError(
            f"No .txt files found in corpus '{name}'. "
            f"Check available corpora with list_remote_corpora()."
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each file
    total_size = 0
    for file_info in txt_files:
        dest_path = output_dir / file_info['name']
        _download_file(file_info['download_url'], dest_path)
        total_size += dest_path.stat().st_size
    
    # Print summary
    size_kb = total_size / 1024
    if size_kb > 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"
    print(f"Downloaded {len(txt_files)} file(s) ({size_str}) to {output_dir}/")
    
    return output_dir


def download_file(path: str, output_dir: str | None = None) -> Path:
    """
    Download a single file from the qhchina-data GitHub repository.
    
    Args:
        path: Path to the file in the repository (e.g., "corpora/莫言/莫言_丰乳肥臀.txt",
            "fonts/NotoSerifSC-Regular.otf"). The path is relative to the repository root.
        output_dir: Directory where the file will be saved. If None (default),
            uses the current working directory.
            
    Returns:
        Path to the downloaded file.
        
    Raises:
        ImportError: If requests is not installed.
        ValueError: If the file is not found.
        requests.RequestException: If the download fails.
        
    Example:
        >>> from qhchina import download_file
        >>> 
        >>> # Download to current directory
        >>> path = download_file("corpora/莫言/莫言_丰乳肥臀.txt")
        >>> # Creates ./莫言_丰乳肥臀.txt
        >>> 
        >>> # Download to a specific directory
        >>> path = download_file("corpora/莫言/莫言_丰乳肥臀.txt", output_dir="texts")
        >>> # Creates ./texts/莫言_丰乳肥臀.txt
    """
    from .github import download_file as _download_file, GITHUB_API_BASE
    
    requests = _get_requests()
    
    # Normalize path (remove leading slash if present)
    path = path.lstrip('/')
    
    # Get file info from GitHub API
    # We need to query the parent directory to get the download URL
    path_parts = path.rsplit('/', 1)
    if len(path_parts) == 2:
        parent_path, filename = path_parts
    else:
        parent_path, filename = '', path_parts[0]
    
    api_url = f"{GITHUB_API_BASE}/{parent_path}" if parent_path else GITHUB_API_BASE
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        contents = response.json()
    except Exception as e:
        raise ValueError(f"Could not access path '{parent_path}': {e}") from e
    
    # Find the file
    file_info = None
    for item in contents:
        if item['type'] == 'file' and item['name'] == filename:
            file_info = item
            break
    
    if file_info is None:
        raise ValueError(f"File '{filename}' not found in '{parent_path}'")
    
    # Determine output path
    if output_dir is None:
        dest_path = Path.cwd() / filename
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dest_path = output_path / filename
    
    # Download the file
    _download_file(file_info['download_url'], dest_path)
    
    # Print summary
    size_bytes = dest_path.stat().st_size
    size_kb = size_bytes / 1024
    if size_kb > 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"
    print(f"Downloaded {filename} ({size_str})")
    
    return dest_path


def list_remote_corpora() -> list[str]:
    """
    List available corpora in the qhchina-data GitHub repository.
    
    Returns:
        List of corpus names (folder names under ``corpora/``).
        
    Raises:
        ImportError: If requests is not installed.
        requests.RequestException: If the API request fails.
        
    Example:
        >>> from qhchina import list_remote_corpora
        >>> corpora = list_remote_corpora()
        >>> print(corpora)
        ['张爱玲', '沈从文', '莫言', ...]
    """
    from .github import query_github_api
    
    contents = query_github_api('corpora')
    
    corpora = [
        item['name'] 
        for item in contents 
        if item['type'] == 'dir'
    ]
    
    return sorted(corpora)
