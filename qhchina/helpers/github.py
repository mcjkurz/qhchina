"""
GitHub API utilities for downloading assets from qhchina-data repository.

This module provides shared functionality for downloading fonts, corpora,
and other assets from the qhchina-data GitHub repository.
"""

import logging
from pathlib import Path

logger = logging.getLogger("qhchina.helpers.github")


__all__ = [
    'GITHUB_REPO',
    'GITHUB_API_BASE',
    'CACHE_BASE',
    'ensure_cache_dir',
    'download_file',
    'query_github_api',
]


# Configuration
GITHUB_REPO = "mcjkurz/qhchina-data"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
CACHE_BASE = Path.home() / '.cache' / 'qhchina'


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


def ensure_cache_dir(subdir: str) -> Path:
    """
    Create and return cache directory for the given subdirectory.
    
    Args:
        subdir: Subdirectory name (e.g., 'fonts', 'corpora')
    
    Returns:
        Path to the cache directory (created if it doesn't exist)
    """
    cache_dir = CACHE_BASE / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        dest: Destination path to save the file
        timeout: Request timeout in seconds (default: 120)
    
    Raises:
        ImportError: If requests is not installed
        requests.RequestException: If download fails
    """
    requests = _get_requests()
    
    response = requests.get(url, timeout=timeout, stream=True)
    if response.status_code == 404:
        raise ValueError(f"Resource not found at URL: {url}")
    response.raise_for_status()
    
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)


def query_github_api(path: str, timeout: int = 30) -> list[dict]:
    """
    Query GitHub API for contents at the given path.
    
    Args:
        path: Path relative to repository root (e.g., 'corpora/songshi', 'fonts')
    
    Returns:
        List of file/directory info dicts from GitHub API
    
    Raises:
        ImportError: If requests is not installed
        requests.RequestException: If API request fails
    """
    requests = _get_requests()
    
    url = f"{GITHUB_API_BASE}/{path}"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        raise ValueError(f"Repository path not found: '{path}'")
    response.raise_for_status()
    return response.json()
