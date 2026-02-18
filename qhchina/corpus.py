"""
Corpus management for qhchina analytics.

The Corpus class provides a unified data structure for managing tokenized documents
with metadata, while maintaining full compatibility with existing analytics modules.

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
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("qhchina.corpus")


__all__ = ['Corpus', 'Document']


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
        '_id_to_index', '_token_count_cache', '_vocab_cache'
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
        self._vocab_cache: Counter | None = None
        
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
            result._vocab_cache = None
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
        tokens: list[str],
        doc_id: str | None = None,
        **metadata: Any
    ) -> str:
        """
        Add a document to the corpus.
        
        Args:
            tokens: List of string tokens (the segmented text).
            doc_id: Optional document ID. If not provided, one is auto-generated.
            **metadata: Metadata key-value pairs (author, date, source, etc.).
        
        Returns:
            The document ID (generated or provided).
        
        Raises:
            TypeError: If tokens is not a list.
            ValueError: If doc_id already exists in corpus.
        
        Example:
            >>> corpus.add(['没有', '吃', '过', '人', '的', '孩子'], author='鲁迅')
            'doc_0'
            >>> corpus.add(['小溪', '流', '下去'], doc_id='边城', author='沈从文')
            '边城'
            >>> corpus[0].tokens      # Access by index
            ['没有', '吃', '过', '人', '的', '孩子']
            >>> corpus['边城'].tokens  # Access by doc_id
            ['小溪', '流', '下去']
        """
        if not isinstance(tokens, list):
            raise TypeError(f"tokens must be a list, got {type(tokens).__name__}")
        
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
        documents: list[list[str]],
        metadata_list: list[dict[str, Any]] | None = None,
        **shared_metadata: Any
    ) -> list[str]:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of token lists.
            metadata_list: Optional per-document metadata. Must match length of documents.
            **shared_metadata: Metadata applied to all documents.
        
        Returns:
            List of document IDs.
        
        Raises:
            ValueError: If metadata_list length doesn't match documents length.
        
        Example:
            >>> docs = [['word1'], ['word2', 'word3']]
            >>> corpus.add_many(docs, period='民国')
            ['doc_0', 'doc_1']
        """
        if metadata_list is not None:
            if len(metadata_list) != len(documents):
                raise ValueError(
                    f"metadata_list length ({len(metadata_list)}) must match "
                    f"documents length ({len(documents)})"
                )
        
        doc_ids = []
        for i, tokens in enumerate(documents):
            per_doc_meta = metadata_list[i] if metadata_list else {}
            merged = {**shared_metadata, **per_doc_meta}
            doc_id = self.add(tokens, **merged)
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
        result._vocab_cache = None
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
        train_corpus._vocab_cache = None
        train_corpus._id_to_index = {}
        
        test_corpus = Corpus.__new__(Corpus)
        test_corpus._documents = []
        test_corpus._default_metadata = self._default_metadata.copy()
        test_corpus._doc_id_counter = 0
        test_corpus._token_count_cache = None
        test_corpus._vocab_cache = None
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
    def vocab(self) -> Counter:
        """
        Vocabulary with word frequencies.
        
        Returns a Counter mapping each unique token to its total count
        across all documents. This value is cached.
        
        Example:
            >>> corpus.vocab.most_common(10)
            [('的', 1523), ('是', 892), ...]
        """
        if self._vocab_cache is None:
            self._vocab_cache = Counter()
            for doc in self._documents:
                self._vocab_cache.update(doc.tokens)
        return self._vocab_cache
    
    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the corpus."""
        return len(self.vocab)
    
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
        return {word for word, count in self.vocab.items() if count == 1}
    
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
        return {word for word, count in self.vocab.items() if count == 2}
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def save(self, path: str | Path, format: str | None = None) -> None:
        """
        Save corpus to a file.
        
        Args:
            path: Output file path.
            format: File format - 'json' or 'pickle'. If None, inferred from
                file extension (.json for JSON, .pkl/.pickle for pickle).
                Pickle is recommended for large corpora as it's more compact
                and faster to load.
        
        Example:
            >>> corpus.save('my_corpus.json')           # JSON format
            >>> corpus.save('my_corpus.pkl')            # Pickle format (smaller)
            >>> corpus.save('corpus', format='pickle')  # Explicit format
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
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    f"Use format='json' or format='pickle', or use .json/.pkl extension."
                )
        
        format = format.lower()
        if format not in ('json', 'pickle'):
            raise ValueError(f"format must be 'json' or 'pickle', got '{format}'")
        
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
        
        if format == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:  # pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved corpus with {len(self._documents)} documents to {path} ({format})")
    
    @classmethod
    def load(cls, path: str | Path, format: str | None = None) -> Corpus:
        """
        Load corpus from a file.
        
        Args:
            path: Input file path.
            format: File format - 'json' or 'pickle'. If None, inferred from
                file extension (.json for JSON, .pkl/.pickle for pickle).
        
        Returns:
            Loaded Corpus object.
        
        Example:
            >>> corpus = Corpus.load('my_corpus.json')   # JSON format
            >>> corpus = Corpus.load('my_corpus.pkl')    # Pickle format
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
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    f"Use format='json' or format='pickle', or use .json/.pkl extension."
                )
        
        format = format.lower()
        if format not in ('json', 'pickle'):
            raise ValueError(f"format must be 'json' or 'pickle', got '{format}'")
        
        if format == 'json':
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
        self._vocab_cache = None
