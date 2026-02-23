"""
Corpus management for qhchina analytics.

The Corpus class provides a unified data structure for managing tokenized documents
with metadata, while maintaining full compatibility with existing analytics modules.

Corpora can be downloaded from the qhchina-data GitHub repository:
    >>> corpus = Corpus.download('songshi')  # Download all files from corpora/songshi/

Example:
    >>> from qhchina import Corpus
    >>> 
    >>> corpus = Corpus()
    >>> corpus.add(['没有', '吃', '过', '人', '的', '孩子'], author='鲁迅')
    >>> corpus.add(['太阳', '刚刚', '下', '了', '地平线'], author='茅盾')
    >>> 
    >>> # Filter by metadata, then analyze the subset
    >>> luxun = corpus.filter(author='鲁迅')
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload, TYPE_CHECKING

import numpy as np

from qhchina.config import get_rng, resolve_seed

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("qhchina.corpus")


# =============================================================================
# Remote corpus download configuration
# =============================================================================

from qhchina.helpers.github import (
    ensure_cache_dir as _ensure_cache_dir,
    download_file as _download_file,
    query_github_api as _query_github_api,
    CACHE_BASE as _CACHE_BASE,
)

_CORPUS_CACHE_DIR = _CACHE_BASE / 'corpora'


def _ensure_corpus_cache_dir() -> Path:
    """Create corpus cache directory if it doesn't exist."""
    return _ensure_cache_dir('corpora')


def _get_corpus_files(corpus_path: str) -> list[dict]:
    """
    Get list of .txt files for a corpus path.
    
    Args:
        corpus_path: Either a corpus name (e.g., 'songshi') or a full path
                     (e.g., 'corpora/songshi/file.txt')
    
    Returns:
        List of dicts with 'name', 'download_url', 'size' for each .txt file
    """
    # Determine if this is a full path or just a corpus name
    if corpus_path.endswith('.txt'):
        # Full path to a specific file
        if not corpus_path.startswith('corpora/'):
            corpus_path = f'corpora/{corpus_path}'
        
        # Query the parent directory to get file info
        parent_path = '/'.join(corpus_path.split('/')[:-1])
        filename = corpus_path.split('/')[-1]
        
        contents = _query_github_api(parent_path)
        
        for item in contents:
            if item['type'] == 'file' and item['name'] == filename:
                return [{
                    'name': item['name'],
                    'download_url': item['download_url'],
                    'size': item['size'],
                    'path': corpus_path,
                }]
        
        raise ValueError(f"File '{filename}' not found in '{parent_path}'")
    else:
        # Corpus name - get all .txt files from directory
        api_path = f"corpora/{corpus_path}"
        contents = _query_github_api(api_path)
        
        files = []
        for item in contents:
            if item['type'] == 'file' and item['name'].endswith('.txt'):
                files.append({
                    'name': item['name'],
                    'download_url': item['download_url'],
                    'size': item['size'],
                    'path': f"corpora/{corpus_path}/{item['name']}",
                })
        
        if not files:
            raise ValueError(
                f"No .txt files found in corpus '{corpus_path}'. "
                f"Check available corpora with Corpus.list_remote()."
            )
        
        return files


def _ensure_corpus_file_cached(
    file_info: dict,
    corpus_name: str,
    force_download: bool = False
) -> tuple[Path, bool]:
    """
    Ensure a corpus file is cached locally, downloading if necessary.
    
    Args:
        file_info: Dict with 'name', 'download_url', 'path'
        corpus_name: Name of the corpus (for cache subdirectory)
        force_download: Re-download even if cached
    
    Returns:
        Tuple of (path to cached file, whether it was downloaded)
    """
    _ensure_corpus_cache_dir()
    
    # Create corpus subdirectory in cache
    corpus_cache = _CORPUS_CACHE_DIR / corpus_name
    corpus_cache.mkdir(parents=True, exist_ok=True)
    
    cached_path = corpus_cache / file_info['name']
    
    if cached_path.exists() and not force_download:
        return cached_path, False
    
    _download_file(file_info['download_url'], cached_path)
    
    return cached_path, True


__all__ = [
    'Corpus',
    'Document',
]


@dataclass(slots=True)
class Document:
    """
    A single document with tokens and metadata.
    
    Supports flexible indexing:
    - ``doc[0]`` → first token
    - ``doc["author"]`` → metadata value
    - ``doc.get("year", 1900)`` → metadata with default
    
    Attributes:
        tokens: List of string tokens (the segmented text).
        metadata: Dictionary of arbitrary metadata (author, date, source, etc.).
        doc_id: Unique identifier for the document.
    """
    tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    
    def __len__(self) -> int:
        """Return the number of tokens in the document."""
        return len(self.tokens)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over tokens."""
        return iter(self.tokens)
    
    def __getitem__(self, key: int | str) -> str | Any:
        """Access tokens by int index, metadata by string key."""
        if isinstance(key, int):
            return self.tokens[key]
        return self.metadata[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)


class Corpus:
    """
    A collection of tokenized documents with metadata.
    
    Works directly with all qhchina analytics modules:
    
    - ``lda.fit(corpus)`` - topic modeling
    - ``find_collocates(corpus, ...)`` - collocation analysis
    - ``Word2Vec(corpus)`` - word embeddings
    - ``stylo.fit_transform(corpus.groupby('author'))`` - stylometry
    
    Args:
        documents: List of token lists, Document objects, or another Corpus.
        metadata: Default metadata applied to all added documents.
    
    Example:
        >>> corpus = Corpus()
        >>> corpus.add(['没有', '吃', '过', '人', '的', '孩子'], author='鲁迅')
        >>> corpus.add(['太阳', '刚刚', '下', '了', '地平线'], author='茅盾')
        >>> 
        >>> for tokens in corpus.filter(author='鲁迅'):
        ...     print(tokens)
    """
    
    __slots__ = (
        '_documents', '_default_metadata', '_doc_id_counter',
        '_id_to_index', '_token_count_cache', '_word_counts_cache'
    )
    
    def __init__(
        self,
        documents: list[list[str]] | list[Document] | 'Corpus' | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._documents: list[Document] = []
        self._default_metadata: dict[str, Any] = metadata.copy() if metadata else {}
        self._doc_id_counter: int = 0
        self._id_to_index: dict[str, int] = {}
        
        # Cached statistics (invalidated on modification)
        self._token_count_cache: int | None = None
        self._word_counts_cache: Counter | None = None
        
        if documents is not None:
            if isinstance(documents, Corpus):
                # Copy from another corpus
                for doc in documents._documents:
                    self.add(
                        doc.tokens,
                        doc_id=None,  # Generate new IDs
                        **doc.metadata
                    )
            else:
                # List of token lists or Document objects
                for doc in documents:
                    if isinstance(doc, Document):
                        self.add(doc.tokens, doc_id=None, **doc.metadata)
                    else:
                        self.add(doc)
    
    # =========================================================================
    # Iteration - Zero-overhead interface for analytics modules
    # =========================================================================
    
    def __iter__(self) -> Iterator[list[str]]:
        """
        Iterate over token lists directly.
        
        This is the primary interface for analytics modules. Yields raw token
        lists with no wrapper overhead, enabling direct use with Word2Vec,
        LDA, collocations, etc.
        
        Yields:
            Token lists (list[str]) for each document.
        
        Example:
            >>> for tokens in corpus:
            ...     print(len(tokens), "tokens")
        """
        for doc in self._documents:
            yield doc.tokens
    
    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self._documents)
    
    def __bool__(self) -> bool:
        """Return True if corpus is non-empty."""
        return len(self._documents) > 0
    
    @overload
    def __getitem__(self, key: int) -> Document: ...
    @overload
    def __getitem__(self, key: str) -> Document: ...
    @overload
    def __getitem__(self, key: slice) -> 'Corpus': ...
    
    def __getitem__(self, key: int | str | slice) -> Document | 'Corpus':
        """
        Get document(s) by index, ID, or slice.
        
        Args:
            key: Can be:
                - int: Return Document at index
                - str: Return Document with matching doc_id
                - slice: Return new Corpus with sliced documents
        
        Returns:
            Document for int/str keys, Corpus for slice.
        
        Example:
            >>> doc = corpus[0]           # First document
            >>> doc = corpus['doc_5']     # Document by ID
            >>> subset = corpus[10:20]    # Slice returns new Corpus
        """
        if isinstance(key, str):
            return self.get(key)
        elif isinstance(key, slice):
            result = Corpus.__new__(Corpus)
            result._documents = self._documents[key]
            result._default_metadata = self._default_metadata.copy()
            result._doc_id_counter = 0
            result._token_count_cache = None
            result._word_counts_cache = None
            result._rebuild_index()
            return result
        else:
            return self._documents[key]
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID exists in the corpus."""
        return doc_id in self._id_to_index
    
    def __repr__(self) -> str:
        return f"Corpus(documents={len(self._documents)}, tokens={self.token_count})"
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    def add(
        self,
        content: list[str] | str,
        doc_id: str | None = None,
        **metadata: Any
    ) -> str:
        """
        Add a document to the corpus.
        
        Args:
            content: Either a list of tokens (already segmented) or a string
                (raw text to be tokenized later with ``.tokenize()``).
            doc_id: Optional document ID. If not provided, one is auto-generated.
            **metadata: Metadata key-value pairs (author, date, source, etc.).
        
        Returns:
            The document ID (generated or provided).
        
        Raises:
            TypeError: If content is neither a list nor a string.
            ValueError: If doc_id already exists in corpus.
        
        Example:
            >>> # Add tokenized document
            >>> corpus.add(['没有', '吃', '过', '人', '的', '孩子'], author='鲁迅')
            'doc_0'
            
            >>> # Add raw text (call .tokenize() later)
            >>> corpus.add('原始文本，需要分词。', doc_id='raw_doc', author='作者')
            'raw_doc'
            >>> corpus.tokenize()  # Tokenize all documents with raw_text
            
            >>> corpus['边城'].tokens  # Access by doc_id
            ['小溪', '流', '下去']
        """
        if isinstance(content, str):
            # Raw text: store in metadata, tokens empty
            tokens = []
            metadata['raw_text'] = content
        elif isinstance(content, list):
            tokens = content
        else:
            raise TypeError(
                f"content must be a list of tokens or a string, got {type(content).__name__}"
            )
        
        self._invalidate_cache()
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = f"doc_{self._doc_id_counter}"
            self._doc_id_counter += 1
        
        # Check for duplicate doc_id
        if doc_id in self._id_to_index:
            raise ValueError(f"Document ID '{doc_id}' already exists in corpus")
        
        # Merge default metadata with provided metadata
        merged_metadata = {**self._default_metadata, **metadata}
        
        # Create and store document
        doc = Document(tokens=tokens, metadata=merged_metadata, doc_id=doc_id)
        self._id_to_index[doc_id] = len(self._documents)
        self._documents.append(doc)
        
        return doc_id
    
    def add_many(
        self,
        documents: list[list[str]] | list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        **shared_metadata: Any
    ) -> list[str]:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: Either a list of token lists (already segmented) or a list
                of strings (raw texts to be tokenized later with ``.tokenize()``).
            metadata_list: Optional per-document metadata. Must match length of documents.
            **shared_metadata: Metadata applied to all documents.
        
        Returns:
            List of document IDs.
        
        Raises:
            ValueError: If metadata_list length doesn't match documents length.
        
        Example:
            >>> # Add tokenized documents
            >>> docs = [['word1'], ['word2', 'word3']]
            >>> corpus.add_many(docs, period='民国')
            ['doc_0', 'doc_1']
            
            >>> # Add raw texts
            >>> texts = ['第一篇文章的内容。', '第二篇文章的内容。']
            >>> corpus.add_many(texts, source='测试')
            ['doc_2', 'doc_3']
            >>> corpus.tokenize()  # Tokenize all
        """
        if metadata_list is not None:
            if len(metadata_list) != len(documents):
                raise ValueError(
                    f"metadata_list length ({len(metadata_list)}) must match "
                    f"documents length ({len(documents)})"
                )
        
        doc_ids = []
        for i, content in enumerate(documents):
            per_doc_meta = metadata_list[i] if metadata_list else {}
            merged = {**shared_metadata, **per_doc_meta}
            doc_id = self.add(content, **merged)
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def get(self, doc_id: str) -> Document:
        """
        Get a document by its ID. Use ``corpus[index]`` for integer access.
        
        Args:
            doc_id: The document ID to look up.
        
        Returns:
            The Document object.
        
        Raises:
            KeyError: If doc_id is not found.
        
        Example:
            >>> corpus['边城'].tokens   # By doc_id
            >>> corpus[0].tokens        # By index
        """
        if doc_id not in self._id_to_index:
            raise KeyError(f"Document '{doc_id}' not found in corpus")
        return self._documents[self._id_to_index[doc_id]]
    
    def remove(self, doc_id: str) -> Document:
        """
        Remove and return a document by its ID.
        
        Args:
            doc_id: The document ID to remove.
        
        Returns:
            The removed Document object.
        
        Raises:
            KeyError: If doc_id is not found.
        
        Example:
            >>> removed = corpus.remove('doc_0')
            >>> print(f"Removed document with {len(removed.tokens)} tokens")
        """
        if doc_id not in self._id_to_index:
            raise KeyError(f"Document '{doc_id}' not found in corpus")
        
        self._invalidate_cache()
        
        idx = self._id_to_index.pop(doc_id)
        doc = self._documents.pop(idx)
        
        # Rebuild index since indices shifted
        self._rebuild_index()
        
        return doc
    
    # =========================================================================
    # Filtering
    # =========================================================================
    
    def filter(
        self,
        predicate: Callable[[Document], bool] | None = None,
        **metadata_filters: Any
    ) -> 'Corpus':
        """
        Return a new Corpus containing only matching documents.
        
        Args:
            predicate: Function that takes a Document and returns bool.
            **metadata_filters: Filter by exact metadata value match.
        
        Returns:
            New Corpus (shares document references, memory-efficient).
        
        Example:
            >>> luxun = corpus.filter(author='鲁迅')
            >>> pre_1930 = corpus.filter(lambda d: d["year"] < 1930)
        """
        result = Corpus.__new__(Corpus)
        result._documents = []
        result._default_metadata = self._default_metadata.copy()
        result._doc_id_counter = 0
        result._token_count_cache = None
        result._word_counts_cache = None
        result._id_to_index = {}
        
        for doc in self._documents:
            # Check metadata filters
            match = True
            for key, value in metadata_filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            
            # Check predicate if metadata matched
            if match and predicate is not None:
                match = predicate(doc)
            
            if match:
                # Share reference to document (no copy)
                result._documents.append(doc)
        
        result._rebuild_index()
        return result
    
    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle documents in-place.
        
        Args:
            seed: Random seed for reproducibility. If None, uses the global
                random seed from ``qhchina.set_random_seed()``.
        
        Example:
            >>> corpus = Corpus([['a', 'b'], ['c', 'd'], ['e', 'f']])
            >>> corpus.shuffle(seed=42)
            >>> list(corpus)  # Documents in random order
        """
        rng = get_rng(resolve_seed(seed))
        rng.shuffle(self._documents)
        self._rebuild_index()
    
    # =========================================================================
    # Grouping
    # =========================================================================
    
    def groupby(self, key: str) -> dict[str, list[list[str]]]:
        """
        Group documents by a metadata key.
        
        Returns a dictionary mapping metadata values to lists of token lists.
        This is the format expected by ``Stylometry.fit_transform()``.
        
        Documents without the specified metadata key are silently skipped.
        
        Args:
            key: Metadata key to group by (e.g., 'author', 'period').
        
        Returns:
            Dictionary mapping metadata values to lists of token lists.
        
        Raises:
            TypeError: If key is not a string.
        
        Example:
            >>> corpus.add(['没有', '吃', '过', '人'], author='鲁迅')
            >>> corpus.add(['救救', '孩子'], author='鲁迅')
            >>> corpus.add(['太阳', '下', '了', '地平线'], author='茅盾')
            >>> 
            >>> grouped = corpus.groupby('author')
            >>> # {'鲁迅': [['没有', '吃', ...], ['救救', '孩子']], '茅盾': [[...]]}
            >>> 
            >>> # Use with Stylometry
            >>> stylo.fit_transform(corpus.groupby('author'))
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        
        result: dict[str, list[list[str]]] = {}
        
        for doc in self._documents:
            group_value = doc.metadata.get(key)
            if group_value is None:
                continue
            
            if group_value not in result:
                result[group_value] = []
            
            # Append reference to token list (no copy)
            result[group_value].append(doc.tokens)
        
        return result
    
    # =========================================================================
    # Splitting
    # =========================================================================
    
    def split(
        self,
        train_ratio: float = 0.8,
        stratify_by: str | None = None,
        seed: int | None = None
    ) -> tuple['Corpus', 'Corpus']:
        """
        Split corpus into train and test sets.
        
        Args:
            train_ratio: Proportion of documents for training (0.0 to 1.0).
            stratify_by: Optional metadata key for stratified splitting.
                If provided, maintains the proportion of each group in both sets.
            seed: Random seed for reproducibility.
        
        Returns:
            Tuple of (train_corpus, test_corpus).
        
        Raises:
            ValueError: If train_ratio is not between 0 and 1.
        
        Example:
            >>> train, test = corpus.split(0.8, seed=42)
            >>> train_stratified, test_stratified = corpus.split(
            ...     0.8, stratify_by='author', seed=42
            ... )
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        
        rng = np.random.default_rng(seed)
        
        if stratify_by is None:
            # Simple random split
            indices = list(range(len(self._documents)))
            rng.shuffle(indices)
            split_point = int(len(indices) * train_ratio)
            train_indices = set(indices[:split_point])
        else:
            # Stratified split: maintain group proportions
            train_indices = set()
            
            # Group document indices by stratify_by value
            groups: dict[Any, list[int]] = {}
            for i, doc in enumerate(self._documents):
                group_value = doc.metadata.get(stratify_by)
                if group_value not in groups:
                    groups[group_value] = []
                groups[group_value].append(i)
            
            # Split each group proportionally
            for group_indices in groups.values():
                group_indices_array = np.array(group_indices)
                rng.shuffle(group_indices_array)
                n_train = max(1, int(len(group_indices_array) * train_ratio))
                train_indices.update(group_indices_array[:n_train])
        
        # Build train and test corpora
        train_corpus = Corpus.__new__(Corpus)
        train_corpus._documents = []
        train_corpus._default_metadata = self._default_metadata.copy()
        train_corpus._doc_id_counter = 0
        train_corpus._token_count_cache = None
        train_corpus._word_counts_cache = None
        train_corpus._id_to_index = {}
        
        test_corpus = Corpus.__new__(Corpus)
        test_corpus._documents = []
        test_corpus._default_metadata = self._default_metadata.copy()
        test_corpus._doc_id_counter = 0
        test_corpus._token_count_cache = None
        test_corpus._word_counts_cache = None
        test_corpus._id_to_index = {}
        
        for i, doc in enumerate(self._documents):
            if i in train_indices:
                train_corpus._documents.append(doc)
            else:
                test_corpus._documents.append(doc)
        
        train_corpus._rebuild_index()
        test_corpus._rebuild_index()
        
        return train_corpus, test_corpus
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    @property
    def token_count(self) -> int:
        """
        Total number of tokens across all documents.
        
        This value is cached and recomputed only when the corpus is modified.
        """
        if self._token_count_cache is None:
            self._token_count_cache = sum(len(doc.tokens) for doc in self._documents)
        return self._token_count_cache
    
    @property
    def word_counts(self) -> Counter:
        """
        Word frequencies across all documents.
        
        Returns a Counter mapping each unique token to its total count
        across all documents. This value is cached.
        
        Example:
            >>> corpus.word_counts.most_common(10)
            [('的', 1523), ('是', 892), ...]
            >>> corpus.word_counts['的']
            1523
        """
        if self._word_counts_cache is None:
            self._word_counts_cache = Counter()
            for doc in self._documents:
                self._word_counts_cache.update(doc.tokens)
        return self._word_counts_cache
    
    @property
    def vocab(self) -> set[str]:
        """
        Set of unique words in the corpus.
        
        Example:
            >>> '的' in corpus.vocab
            True
            >>> len(corpus.vocab)
            1205
        """
        return set(self.word_counts.keys())
    
    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the corpus."""
        return len(self.word_counts)
    
    @property
    def metadata_keys(self) -> set[str]:
        """Set of all metadata keys present in any document."""
        keys: set[str] = set()
        for doc in self._documents:
            keys.update(doc.metadata.keys())
        return keys
    
    def describe(self) -> dict[str, Any]:
        """
        Return summary statistics about the corpus.
        
        Returns:
            Dictionary with:
                - documents: Number of documents
                - tokens: Total token count
                - vocab_size: Number of unique tokens
                - avg_doc_length: Average tokens per document
                - metadata_keys: List of metadata keys
        
        Example:
            >>> corpus.describe()
            {'documents': 100, 'tokens': 5432, 'vocab_size': 1205, ...}
        """
        n_docs = len(self._documents)
        return {
            'documents': n_docs,
            'tokens': self.token_count,
            'vocab_size': self.vocab_size,
            'avg_doc_length': self.token_count / n_docs if n_docs > 0 else 0,
            'metadata_keys': sorted(self.metadata_keys),
        }
    
    def metadata_values(self, key: str) -> set[Any]:
        """
        Get all unique values for a metadata key.
        
        Args:
            key: The metadata key to query.
        
        Returns:
            Set of unique values (excludes documents without this key).
        
        Example:
            >>> corpus.metadata_values('author')
            {'鲁迅', '茅盾', '沈从文'}
        """
        return {
            doc.metadata[key]
            for doc in self._documents
            if key in doc.metadata
        }
    
    def type_token_ratio(self, variant: str = 'standard') -> float:
        """
        Calculate Type-Token Ratio (TTR) for the corpus.
        
        TTR measures lexical diversity - the ratio of unique words (types)
        to total words (tokens). Higher values indicate more diverse vocabulary.
        
        Note: Standard TTR is sensitive to text length (longer texts tend to
        have lower TTR). Use 'root' or 'log' variants for length-normalized
        measures, or use mattr() for comparing texts of different lengths.
        
        Args:
            variant: Calculation method:
                - 'standard': types / tokens (range: 0.0 to 1.0)
                - 'root': types / sqrt(tokens) - Guiraud's R
                - 'log': log(types) / log(tokens) - Herdan's C
        
        Returns:
            The TTR value. For 'standard', this is between 0.0 and 1.0.
            For 'root' and 'log', values vary based on corpus size.
        
        Raises:
            ValueError: If variant is not recognized or corpus is empty.
        
        Example:
            >>> corpus.type_token_ratio()
            0.342
            >>> corpus.type_token_ratio(variant='root')
            45.67
            >>> corpus.type_token_ratio(variant='log')
            0.891
        """
        if variant not in ('standard', 'root', 'log'):
            raise ValueError(
                f"variant must be 'standard', 'root', or 'log', got '{variant}'"
            )
        
        tokens = self.token_count
        types = self.vocab_size
        
        if tokens == 0:
            raise ValueError("Cannot calculate TTR for empty corpus")
        
        if variant == 'standard':
            return types / tokens
        elif variant == 'root':
            # Guiraud's R: types / sqrt(tokens)
            return types / np.sqrt(tokens)
        else:  # log
            # Herdan's C: log(types) / log(tokens)
            if tokens == 1:
                return 1.0  # Edge case: single token
            return np.log(types) / np.log(tokens)
    
    def mattr(self, window_size: int = 500) -> float:
        """
        Calculate Moving Average Type-Token Ratio (MATTR).
        
        MATTR is more reliable than standard TTR for comparing texts of
        different lengths. It calculates TTR for a sliding window across
        the corpus and returns the mean.
        
        Args:
            window_size: Number of tokens per window. Default is 500,
                which is standard in the literature. Smaller windows
                give higher MATTR values.
        
        Returns:
            Mean TTR across all windows (0.0 to 1.0).
        
        Raises:
            ValueError: If corpus has fewer tokens than window_size.
        
        Example:
            >>> corpus.mattr()
            0.723
            >>> corpus.mattr(window_size=100)
            0.856
        
        Reference:
            Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot:
            The moving-average type-token ratio (MATTR). Journal of Quantitative
            Linguistics, 17(2), 94-100.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        
        # Flatten corpus to single token stream
        all_tokens = []
        for doc in self._documents:
            all_tokens.extend(doc.tokens)
        
        n_tokens = len(all_tokens)
        
        if n_tokens < window_size:
            raise ValueError(
                f"Corpus has {n_tokens} tokens, but window_size is {window_size}. "
                f"Use a smaller window_size or add more documents."
            )
        
        # Calculate TTR for each window position
        n_windows = n_tokens - window_size + 1
        ttr_sum = 0.0
        
        # Use a sliding window with incremental updates for efficiency
        window_counts: Counter = Counter(all_tokens[:window_size])
        ttr_sum += len(window_counts) / window_size
        
        for i in range(1, n_windows):
            # Remove token leaving the window
            leaving = all_tokens[i - 1]
            window_counts[leaving] -= 1
            if window_counts[leaving] == 0:
                del window_counts[leaving]
            
            # Add token entering the window
            entering = all_tokens[i + window_size - 1]
            window_counts[entering] += 1
            
            # Calculate TTR for this window
            ttr_sum += len(window_counts) / window_size
        
        return ttr_sum / n_windows
    
    def hapax_legomena(self) -> set[str]:
        """
        Return words that appear exactly once in the corpus.
        
        Hapax legomena (Greek: "said once") are words occurring only once.
        They are important for:
        - Vocabulary richness analysis
        - Authorship attribution
        - Zipf's law studies
        
        A corpus typically has 40-60% of its vocabulary as hapax legomena.
        
        Returns:
            Set of words with frequency == 1.
        
        Example:
            >>> hapax = corpus.hapax_legomena()
            >>> len(hapax)
            1247
            >>> print(f"Hapax ratio: {len(hapax) / corpus.vocab_size:.1%}")
            Hapax ratio: 48.2%
        """
        return {word for word, count in self.word_counts.items() if count == 1}
    
    def hapax_dislegomena(self) -> set[str]:
        """
        Return words that appear exactly twice in the corpus.
        
        Hapax dislegomena (Greek: "said twice") complement hapax legomena
        in vocabulary richness analysis.
        
        Returns:
            Set of words with frequency == 2.
        
        Example:
            >>> dis = corpus.hapax_dislegomena()
            >>> len(dis)
            523
        """
        return {word for word, count in self.word_counts.items() if count == 2}
    
    def count(self, word: str) -> int:
        """
        Return the number of times a word appears across all documents.
        
        Args:
            word: The word to count.
        
        Returns:
            Total count of the word in the corpus. Returns 0 if the word
            is not found.
        
        Example:
            >>> corpus.count('的')
            1523
            >>> corpus.count('不存在的词')
            0
        """
        return self.word_counts.get(word, 0)
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Make a string safe for use as a filename.
        
        Replaces characters that are invalid in filenames on common filesystems.
        """
        for char in r'/\:*?"<>|':
            name = name.replace(char, '_')
        # Truncate if too long (most filesystems limit to 255 bytes)
        if len(name.encode('utf-8')) > 200:
            # Truncate safely for UTF-8
            encoded = name.encode('utf-8')[:200]
            name = encoded.decode('utf-8', errors='ignore')
        return name
    
    def save(self, path: str | Path, format: str | None = None) -> None:
        """
        Save corpus to a file.
        
        Args:
            path: Output file path.
            format: File format - 'json', 'pickle', or 'txt'. If None, inferred from
                file extension (.json for JSON, .pkl/.pickle for pickle, .txt for text).
                
                - json: Human-readable, includes metadata
                - pickle: Compact binary, includes metadata  
                - txt: Streaming format for Word2Vec/TempRefWord2Vec (tokens only, no metadata)
        
        Example:
            >>> corpus.save('my_corpus.json')           # JSON format
            >>> corpus.save('my_corpus.pkl')            # Pickle format (smaller)
            >>> corpus.save('my_corpus.txt')            # Streaming text format
            >>> corpus.save('corpus', format='pickle')  # Explicit format
        
        Note:
            The 'txt' format is designed for streaming with Word2Vec and TempRefWord2Vec.
            It only saves tokens (one sentence per line), not document IDs or metadata.
        """
        import pickle
        
        path = Path(path)
        
        # Infer format from extension if not specified
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ('.pkl', '.pickle'):
                format = 'pickle'
            elif suffix == '.txt':
                format = 'txt'
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    f"Use format='json', 'pickle', or 'txt', or use .json/.pkl/.txt extension."
                )
        
        format = format.lower()
        if format not in ('json', 'pickle', 'txt'):
            raise ValueError(f"format must be 'json', 'pickle', or 'txt', got '{format}'")
        
        if format == 'txt':
            # Streaming format: one document per line (tokens only, no metadata)
            with open(path, 'w', encoding='utf-8') as f:
                for doc in self._documents:
                    f.write(' '.join(doc.tokens) + '\n')
            
            logger.info(f"Saved corpus with {len(self._documents)} documents to {path} (txt)")
        elif format == 'json':
            data = {
                'version': '1.0',
                'default_metadata': self._default_metadata,
                'documents': [
                    {
                        'doc_id': doc.doc_id,
                        'tokens': doc.tokens,
                        'metadata': doc.metadata,
                    }
                    for doc in self._documents
                ],
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved corpus with {len(self._documents)} documents to {path} (json)")
        else:  # pickle
            data = {
                'version': '1.0',
                'default_metadata': self._default_metadata,
                'documents': [
                    {
                        'doc_id': doc.doc_id,
                        'tokens': doc.tokens,
                        'metadata': doc.metadata,
                    }
                    for doc in self._documents
                ],
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved corpus with {len(self._documents)} documents to {path} (pickle)")
    
    def save_folder(
        self,
        path: str | Path,
        format: str = "json",
        clear: bool = True
    ) -> None:
        """
        Save corpus as a folder with one file per document.
        
        Creates a folder containing individual files for each document, plus a
        ``_meta.json`` file with corpus-level metadata and document order.
        
        Args:
            path: Output folder path. Created if it doesn't exist.
            format: File format for documents - ``'json'`` (default) or ``'txt'``.
                
                - json: Each document saved as ``{doc_id}.json`` with tokens and metadata
                - txt: Each document saved as ``{doc_id}.txt`` with space-separated tokens
            clear: If True (default), remove existing ``.json`` and ``.txt`` files
                from the folder before saving. This prevents stale files from
                being loaded by ``load_folder()``. Set to False to preserve
                existing files (useful for incremental updates).
        
        Example:
            >>> corpus.save_folder('my_corpus/')              # JSON files
            >>> corpus.save_folder('my_corpus/', format='txt') # Text files
        
        Note:
            Document filenames are derived from ``doc_id``. Characters unsafe for
            filenames are replaced with underscores. If two documents have the same
            sanitized filename, a numeric suffix is added (e.g., ``doc_1.json``).
            
            The ``_meta.json`` file stores corpus-level metadata and the original
            document order, which is used by ``load_folder()`` to preserve ordering.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Clear existing corpus files if requested
        if clear:
            for existing_file in path.glob('*.json'):
                if existing_file.name != '_meta.json':
                    existing_file.unlink()
            for existing_file in path.glob('*.txt'):
                existing_file.unlink()
        
        format = format.lower()
        if format not in ('json', 'txt'):
            raise ValueError(f"format must be 'json' or 'txt' for folder mode, got '{format}'")
        
        ext = '.json' if format == 'json' else '.txt'
        
        # Track used filenames to handle collisions
        used_filenames: dict[str, int] = {}
        document_order: list[str] = []
        filename_mapping: dict[str, str] = {}  # doc_id -> actual filename
        
        for doc in self._documents:
            # Sanitize doc_id for filename
            base_name = self._sanitize_filename(doc.doc_id) if doc.doc_id else 'document'
            
            # Handle collisions
            if base_name in used_filenames:
                used_filenames[base_name] += 1
                filename = f"{base_name}_{used_filenames[base_name]}{ext}"
            else:
                used_filenames[base_name] = 0
                filename = f"{base_name}{ext}"
            
            filepath = path / filename
            document_order.append(doc.doc_id)
            filename_mapping[doc.doc_id] = filename
            
            if format == 'json':
                doc_data = {
                    'doc_id': doc.doc_id,
                    'tokens': doc.tokens,
                    'metadata': doc.metadata,
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
            else:  # txt
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(' '.join(doc.tokens))
        
        # Save corpus-level metadata
        meta = {
            'version': '1.0',
            'format': format,
            'default_metadata': self._default_metadata,
            'document_order': document_order,
            'filename_mapping': filename_mapping,
        }
        with open(path / '_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved corpus with {len(self._documents)} documents to {path}/ ({format})")
    
    @classmethod
    def load(cls, path: str | Path, format: str | None = None) -> 'Corpus':
        """
        Load corpus from a file.
        
        Args:
            path: Input file path.
            format: File format - 'json', 'pickle', or 'txt'. If None, inferred from
                file extension (.json for JSON, .pkl/.pickle for pickle, .txt for text).
        
        Returns:
            Loaded Corpus object.
        
        Example:
            >>> corpus = Corpus.load('my_corpus.json')   # JSON format
            >>> corpus = Corpus.load('my_corpus.pkl')    # Pickle format
            >>> corpus = Corpus.load('my_corpus.txt')    # Streaming text format
        
        Note:
            Loading from 'txt' format creates documents without metadata since
            the text format only stores tokens.
        """
        import pickle
        
        path = Path(path)
        
        # Infer format from extension if not specified
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ('.pkl', '.pickle'):
                format = 'pickle'
            elif suffix == '.txt':
                format = 'txt'
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    f"Use format='json', 'pickle', or 'txt', or use .json/.pkl/.txt extension."
                )
        
        format = format.lower()
        if format not in ('json', 'pickle', 'txt'):
            raise ValueError(f"format must be 'json', 'pickle', or 'txt', got '{format}'")
        
        if format == 'txt':
            # Streaming format: one document per line
            corpus = cls()
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line:
                        tokens = line.split(' ')
                        corpus.add(tokens)
            logger.info(f"Loaded corpus with {len(corpus)} documents from {path} (txt)")
            return corpus
        elif format == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:  # pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        corpus = cls(metadata=data.get('default_metadata'))
        
        for doc_data in data['documents']:
            corpus.add(
                doc_data['tokens'],
                doc_id=doc_data.get('doc_id'),
                **doc_data.get('metadata', {})
            )
        
        logger.info(f"Loaded corpus with {len(corpus)} documents from {path} ({format})")
        return corpus
    
    @classmethod
    def load_folder(cls, path: str | Path, pattern: str | None = None) -> 'Corpus':
        """
        Load corpus from a folder of document files.
        
        Loads all supported files (``.json``, ``.txt``) from a folder. If a
        ``_meta.json`` file exists, it is used to restore document order and
        corpus-level metadata; otherwise documents are sorted alphabetically
        by filename.
        
        Args:
            path: Input folder path.
            pattern: Optional glob pattern to filter files (e.g., ``'*.txt'``,
                ``'chapter_*.json'``). If None, loads all ``.json`` and ``.txt`` files.
        
        Returns:
            Loaded Corpus object.
        
        Example:
            >>> corpus = Corpus.load_folder('my_corpus/')
            >>> corpus = Corpus.load_folder('texts/', pattern='*.txt')
        
        Note:
            For ``.txt`` files, the filename (without extension) becomes the ``doc_id``
            and tokens are split by whitespace. ``.txt`` files have no metadata.
            
            For ``.json`` files, the file should contain ``doc_id``, ``tokens``,
            and optionally ``metadata``.
        """
        path = Path(path)
        
        if not path.is_dir():
            raise ValueError(f"Path '{path}' is not a directory. Use load() for single files.")
        
        # Check for _meta.json
        meta_path = path / '_meta.json'
        meta = None
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        
        # Find files to load
        if pattern:
            files = list(path.glob(pattern))
        else:
            files = list(path.glob('*.json')) + list(path.glob('*.txt'))
        
        # Exclude _meta.json
        files = [f for f in files if f.name != '_meta.json' and f.is_file()]
        
        if not files:
            logger.warning(f"No supported files found in {path}")
            return cls(metadata=meta.get('default_metadata') if meta else None)
        
        # Determine load order
        if meta and 'filename_mapping' in meta:
            # Use filename_mapping to restore original order
            filename_to_docid = {v: k for k, v in meta['filename_mapping'].items()}
            doc_order = meta.get('document_order', [])
            
            # Build ordered file list
            ordered_files = []
            remaining_files = set(files)
            
            for doc_id in doc_order:
                filename = meta['filename_mapping'].get(doc_id)
                if filename:
                    filepath = path / filename
                    if filepath in remaining_files:
                        ordered_files.append((filepath, doc_id))
                        remaining_files.remove(filepath)
            
            # Add any remaining files not in meta (sorted alphabetically)
            for f in sorted(remaining_files, key=lambda x: x.name):
                doc_id = filename_to_docid.get(f.name, f.stem)
                ordered_files.append((f, doc_id))
            
            files_with_ids = ordered_files
        else:
            # No meta or no mapping - sort alphabetically, use filename as doc_id
            files = sorted(files, key=lambda x: x.name)
            files_with_ids = [(f, f.stem) for f in files]
        
        # Create corpus
        corpus = cls(metadata=meta.get('default_metadata') if meta else None)
        
        for filepath, doc_id in files_with_ids:
            suffix = filepath.suffix.lower()
            
            if suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                corpus.add(
                    doc_data.get('tokens', []),
                    doc_id=doc_data.get('doc_id', doc_id),
                    **doc_data.get('metadata', {})
                )
            elif suffix == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                tokens = content.split() if content else []
                corpus.add(tokens, doc_id=doc_id)
            else:
                logger.warning(f"Skipping unsupported file: {filepath}")
        
        logger.info(f"Loaded corpus with {len(corpus)} documents from {path}/")
        return corpus
    
    @classmethod
    def load_cached(cls, corpus: str) -> 'Corpus':
        """
        Load a corpus from local cache without network access.
        
        This loads a previously downloaded corpus from the cache directory
        (``~/.cache/qhchina/corpora/{corpus}/``). The corpus must have been
        downloaded earlier using ``Corpus.download()``.
        
        Args:
            corpus: Name of the cached corpus (e.g., ``'songshi'``).
        
        Returns:
            Corpus with raw text in metadata (same as ``download()`` returns).
            Tokens are empty; use ``.tokenize()`` to segment.
        
        Raises:
            FileNotFoundError: If the corpus is not in cache.
        
        Example:
            >>> # First time: download and cache
            >>> corpus = Corpus.download('songshi')
            >>> corpus.tokenize().save('songshi_tokenized.json')
            
            >>> # Later: load from cache (no network needed)
            >>> corpus = Corpus.load_cached('songshi')
            >>> corpus.tokenize()
        
        Note:
            Use ``Corpus.list_cached()`` to see available cached corpora.
        """
        cache_path = _CORPUS_CACHE_DIR / corpus
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Corpus '{corpus}' not found in cache at {cache_path}. "
                f"Download it first with: Corpus.download('{corpus}')"
            )
        
        # Load all .txt files from cache (same structure as download creates)
        txt_files = sorted(cache_path.glob('*.txt'))
        
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in cache for corpus '{corpus}'. "
                f"The cache may be corrupted. Try: Corpus.clear_cache('{corpus}') "
                f"and then Corpus.download('{corpus}')"
            )
        
        result = cls()
        
        for filepath in txt_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            doc_id = filepath.stem
            result.add(raw_text, doc_id=doc_id, corpus=corpus)
        
        logger.info(f"Loaded corpus '{corpus}' with {len(result)} documents from cache")
        return result
    
    @staticmethod
    def download(
        corpus: str,
        show_progress: bool = True,
        force_download: bool = False
    ) -> 'Corpus':
        """
        Download a corpus from the qhchina-data GitHub repository.
        
        Downloads .txt files from ``qhchina-data/corpora/{corpus}/`` and creates
        a Corpus where each file becomes a Document. The raw text is stored in
        the ``raw_text`` metadata field; tokens are initially empty (use a
        segmenter to tokenize).
        
        Args:
            corpus: Either:
                - A corpus name (e.g., ``'songshi'``) to download all .txt files
                  from that corpus folder
                - A full path (e.g., ``'songshi/宋史.txt'``) to download a single file
            show_progress: Show download progress (default True).
            force_download: Re-download even if files are cached (default False).
        
        Returns:
            Corpus with one Document per downloaded file. Each document has:
                - ``doc_id``: filename without .txt extension (e.g., '莫言_红高粱家族')
                - ``tokens``: empty list (raw text, needs segmentation)
                - ``metadata``: ``{'corpus': corpus_name, 'raw_text': content}``
        
        Raises:
            ValueError: If corpus or file not found in repository.
            requests.RequestException: If download fails.
        
        Example:
            >>> # Download all files from a corpus
            >>> corpus = Corpus.download('songshi')
            >>> print(f"Downloaded {len(corpus)} documents")
            
            >>> # Download a single file
            >>> corpus = Corpus.download('宋史/宋史.txt')
            
            >>> # Download and tokenize in one chain (uses default jieba backend)
            >>> corpus = Corpus.download('songshi').tokenize()
            
            >>> # Or use a custom segmenter
            >>> from qhchina.preprocessing import create_segmenter
            >>> segmenter = create_segmenter('spacy', filters={'min_word_length': 2})
            >>> corpus = Corpus.download('songshi').tokenize(segmenter=segmenter)
        
        Note:
            Files are cached in ``~/.cache/qhchina/corpora/``. Use
            ``Corpus.list_cached()`` to see cached corpora and
            ``Corpus.clear_cache()`` to remove them.
        """
        # Extract corpus name from path
        corpus_path = corpus.strip('/')
        if '/' in corpus_path:
            # Path like 'songshi/file.txt' or 'corpora/songshi/file.txt'
            parts = corpus_path.replace('corpora/', '').split('/')
            corpus_name = parts[0]
        else:
            corpus_name = corpus_path
        
        # Get file info from GitHub
        files = _get_corpus_files(corpus_path)
        
        # Download and read each file
        result = Corpus()
        total_size = 0
        
        for file_info in files:
            cached_path, _ = _ensure_corpus_file_cached(
                file_info,
                corpus_name,
                force_download=force_download
            )
            total_size += cached_path.stat().st_size
            
            # Read file content
            with open(cached_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Create document with raw text (user should segment with .tokenize())
            # doc_id is filename without .txt extension
            doc_id = Path(file_info['name']).stem
            result.add(raw_text, doc_id=doc_id, corpus=corpus_name)
        
        # Print completion message
        if show_progress:
            size_kb = total_size / 1024
            if size_kb > 1024:
                size_str = f"{size_kb / 1024:.1f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"
            print(f"Loaded {len(result)} file(s) ({size_str})")
        
        return result
    
    @staticmethod
    def list_remote() -> list[dict]:
        """
        List available corpora in the qhchina-data repository.
        
        Returns:
            List of dicts with corpus information:
            ``[{'name': 'songshi', 'type': 'dir'}, ...]``
        
        Example:
            >>> Corpus.list_remote()
            [{'name': 'songshi', 'type': 'dir'}, {'name': 'mingshi', 'type': 'dir'}]
        """
        contents = _query_github_api('corpora')
        
        corpora = []
        for item in contents:
            if item['type'] == 'dir':
                corpora.append({
                    'name': item['name'],
                    'type': 'dir',
                })
        
        return corpora
    
    @staticmethod
    def list_cached() -> list[dict]:
        """
        List locally cached corpora.
        
        Returns:
            List of dicts with cached corpus information:
            ``[{'name': 'songshi', 'files': 3, 'size_mb': 1.5, 'path': '/path/to/cache'}, ...]``
        
        Example:
            >>> Corpus.list_cached()
            [{'name': 'songshi', 'files': 3, 'size_mb': 1.5, 'path': '~/.cache/qhchina/corpora/songshi'}]
        """
        if not _CORPUS_CACHE_DIR.exists():
            return []
        
        result = []
        for corpus_dir in _CORPUS_CACHE_DIR.iterdir():
            if corpus_dir.is_dir() and not corpus_dir.name.startswith('.'):
                files = list(corpus_dir.glob('*.txt'))
                total_size = sum(f.stat().st_size for f in files)
                result.append({
                    'name': corpus_dir.name,
                    'files': len(files),
                    'size_mb': round(total_size / 1024 / 1024, 2),
                    'path': str(corpus_dir),
                })
        
        return result
    
    @staticmethod
    def clear_cache(corpus: str | None = None) -> None:
        """
        Clear cached corpus files.
        
        Args:
            corpus: Name of corpus to clear, or None to clear all cached corpora.
        
        Example:
            >>> Corpus.clear_cache('songshi')  # Clear specific corpus
            >>> Corpus.clear_cache()           # Clear all corpora
        """
        if not _CORPUS_CACHE_DIR.exists():
            return
        
        if corpus is None:
            # Clear all
            for corpus_dir in _CORPUS_CACHE_DIR.iterdir():
                if corpus_dir.is_dir():
                    for f in corpus_dir.glob('*'):
                        if f.is_file():
                            f.unlink()
                    corpus_dir.rmdir()
        else:
            # Clear specific corpus
            corpus_dir = _CORPUS_CACHE_DIR / corpus
            if corpus_dir.exists():
                for f in corpus_dir.glob('*'):
                    if f.is_file():
                        f.unlink()
                corpus_dir.rmdir()
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert corpus to a pandas DataFrame.
        
        Returns:
            DataFrame with columns: doc_id, tokens, token_count, and all metadata keys.
        
        Example:
            >>> df = corpus.to_dataframe()
            >>> df.groupby('author')['token_count'].mean()
        """
        import pandas as pd
        
        rows = []
        for doc in self._documents:
            row = {
                'doc_id': doc.doc_id,
                'tokens': doc.tokens,
                'token_count': len(doc.tokens),
                **doc.metadata
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # =========================================================================
    # Tokenization
    # =========================================================================
    
    def tokenize(
        self,
        segmenter: Any | None = None,
        backend: str = "jieba",
        strategy: str = "document",
        raw_text_key: str = "raw_text",
        skip_tokenized: bool = True,
        **kwargs
    ) -> 'Corpus':
        """
        Tokenize all documents in-place using a segmenter.
        
        Applies a segmenter to each document's raw text (from metadata) and
        populates the document's tokens list. This is typically used after
        ``Corpus.download()`` which loads raw text but leaves tokens empty.
        
        Args:
            segmenter: Pre-configured segmenter instance. If provided, ``backend``,
                ``strategy``, and ``kwargs`` are ignored.
            backend: Segmentation backend if no segmenter provided. Options:
                ``'spacy'``, ``'pkuseg'``, ``'jieba'`` (default), ``'bert'``, ``'llm'``.
            strategy: Text splitting strategy. Options:
                - ``'document'`` (default): Treat entire raw_text as one unit.
                - ``'sentence'``: Split by sentence boundaries into separate documents.
                - ``'line'``: Split by newlines into separate documents.
                - ``'chunk'``: Split into fixed-size chunks as separate documents.
                
                For strategies other than ``'document'``, the corpus is modified
                in-place: each original document is replaced by multiple new documents
                (one per sentence/line/chunk). New documents inherit metadata
                (excluding ``raw_text_key``) and get ``doc_id``s like
                ``"{original_id}_0"``, ``"{original_id}_1"``, etc.
            raw_text_key: Metadata key containing the raw text (default: ``'raw_text'``).
            skip_tokenized: If True, skip documents that already have tokens
                (default: True).
            **kwargs: Additional arguments passed to ``create_segmenter()``
                (e.g., ``filters``, ``user_dict``, ``chunk_size``).
        
        Returns:
            Self for method chaining (always modifies in-place).
        
        Raises:
            ImportError: If the specified backend is not installed.
        
        Example:
            >>> # Download and tokenize in one chain
            >>> corpus = Corpus.download('songshi').tokenize()
            
            >>> # Split into sentences (modifies corpus in-place)
            >>> corpus.tokenize(strategy='sentence')
            >>> # Original doc "宋史" is replaced by "宋史_0", "宋史_1", ...
            
            >>> # With filters
            >>> corpus.tokenize(
            ...     backend='spacy',
            ...     filters={'stopwords': stopwords, 'min_word_length': 2}
            ... )
        """
        from qhchina.preprocessing import create_segmenter
        
        # Create segmenter if not provided
        if segmenter is None:
            segmenter = create_segmenter(backend=backend, strategy=strategy, **kwargs)
        
        # Get strategy from the segmenter (which inherits strategy param if we created it)
        actual_strategy = segmenter.strategy
        
        # For 'document' strategy: simple in-place tokenization
        if actual_strategy == "document":
            for doc in self._documents:
                if skip_tokenized and doc.tokens:
                    continue
                
                raw_text = doc.metadata.get(raw_text_key)
                if raw_text is None:
                    continue
                
                doc.tokens = segmenter.segment(raw_text)
            
            self._invalidate_cache()
            return self
        
        # For other strategies: replace documents with split versions
        new_documents: list[Document] = []
        
        for doc in self._documents:
            if skip_tokenized and doc.tokens:
                new_documents.append(doc)
                continue
            
            raw_text = doc.metadata.get(raw_text_key)
            if raw_text is None:
                new_documents.append(doc)
                continue
            
            # Segment the text (returns list of token lists)
            token_lists = segmenter.segment(raw_text)
            
            # Create metadata without raw_text to avoid bloat
            new_metadata = {k: v for k, v in doc.metadata.items() if k != raw_text_key}
            
            # Create a new document for each token list
            for i, tokens in enumerate(token_lists):
                new_doc_id = f"{doc.doc_id}_{i}"
                new_doc = Document(tokens=tokens, metadata=new_metadata.copy(), doc_id=new_doc_id)
                new_documents.append(new_doc)
        
        # Replace documents in-place
        self._documents = new_documents
        self._rebuild_index()
        self._invalidate_cache()
        
        return self
    
    # =========================================================================
    # Private Helpers
    # =========================================================================
    
    def _rebuild_index(self) -> None:
        """Rebuild the doc_id to index mapping."""
        self._id_to_index = {
            doc.doc_id: i
            for i, doc in enumerate(self._documents)
        }
        # Update counter to avoid ID collisions
        max_numeric_id = -1
        for doc_id in self._id_to_index:
            if doc_id.startswith('doc_'):
                try:
                    num = int(doc_id[4:])
                    max_numeric_id = max(max_numeric_id, num)
                except ValueError:
                    pass
        self._doc_id_counter = max_numeric_id + 1
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached statistics."""
        self._token_count_cache = None
        self._word_counts_cache = None


