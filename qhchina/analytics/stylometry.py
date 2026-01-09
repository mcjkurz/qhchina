"""
Stylometry module for authorship attribution and document clustering.

Inspired by the R package 'stylo' (https://github.com/computationalstylistics/stylo),
a much more comprehensive implementation for computational stylistics.

Workflow:
1. Create a Stylometry instance with desired parameters
2. Call fit_transform() with your corpus (dict or list of tokenized documents)
3. Analyze with: plot(), dendrogram(), most_similar(), similarity(), distance(), predict()

Two modes for supervised learning:
- 'centroid': Aggregate all author texts into one profile, compare disputed text to centroids
- 'instance': Keep individual texts separate, find nearest neighbor among all texts
"""

import warnings
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .vectors import cosine_similarity as _cosine_similarity
from ..config import resolve_seed


# =============================================================================
# Standalone Functions
# =============================================================================

def extract_mfw(ngram_counts: Counter, n: int = 100) -> List[str]:
    """
    Extract the n Most Frequent n-grams from a Counter.
    
    Returns a list of the n most common n-grams.
    """
    return [ngram for ngram, _ in ngram_counts.most_common(n)]


def burrows_delta(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Burrows' Delta distance: the mean absolute difference between z-score vectors.
    
    A classic stylometric measure; lower values indicate more similar writing styles.
    """
    return np.mean(np.abs(vec_a - vec_b))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity)."""
    return 1.0 - _cosine_similarity(vec_a, vec_b)


def manhattan_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Manhattan (L1) distance: sum of absolute differences."""
    return np.sum(np.abs(vec_a - vec_b))


def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Euclidean (L2) distance: square root of sum of squared differences."""
    return np.sqrt(np.sum((vec_a - vec_b) ** 2))


def eder_delta(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Eder's Delta distance: a variation of Burrows' Delta with different weighting.
    
    Eder's Delta squares the differences and takes the square root of the mean,
    giving more weight to larger differences. It also normalizes by vector length.
    
    Formula: sqrt(sum((a_i - b_i)^2 / n))
    
    Reference: Eder, M. (2013). "Mind your corpus: systematic errors in authorship attribution"
    """
    n = len(vec_a)
    if n == 0:
        return 0.0
    return np.sqrt(np.sum((vec_a - vec_b) ** 2) / n)


def get_relative_frequencies(items: List[str]) -> Dict[str, float]:
    """
    Compute relative frequencies for a list of items (tokens or n-grams).
    
    Returns:
        Dict mapping each unique item to its relative frequency (count / total)
    """
    if not items:
        return {}
    counts = Counter(items)
    total = len(items)
    return {item: count / total for item, count in counts.items()}


def compute_yule_k(tokens: List[str]) -> float:
    """
    Compute Yule's K characteristic for vocabulary richness.
    
    Yule's K is a measure of lexical diversity that is relatively independent
    of text length. Higher values indicate less diverse vocabulary.
    
    Formula: K = 10^4 * (M2 - M1) / (M1^2)
    where M1 = total tokens, M2 = sum of (frequency^2 * count_of_words_with_that_frequency)
    
    Args:
        tokens: List of tokens
        
    Returns:
        Yule's K value (typically between 50-200 for normal texts)
    """
    if not tokens:
        return 0.0
    
    freq_counts = Counter(tokens)
    m1 = len(tokens)  # Total number of tokens
    
    # Count how many words have each frequency
    freq_of_freqs = Counter(freq_counts.values())
    
    # M2 = sum of (r^2 * V_r) where V_r is number of words appearing r times
    m2 = sum(r * r * v_r for r, v_r in freq_of_freqs.items())
    
    if m1 <= 1:
        return 0.0
    
    # Yule's K formula
    k = 10000 * (m2 - m1) / (m1 * m1)
    return k


# =============================================================================
# Main Stylometry Class
# =============================================================================

class Stylometry:
    """
    Stylometry for authorship attribution and document clustering.
    
    Parameters:
        n_features: Number of most frequent n-grams to use as features.
        ngram_range: Tuple (min_n, max_n) for n-grams. Default (1, 1) = unigrams.
        ngram_type: Type of n-grams - 'word' (default) or 'char'.
            - 'word': Word n-grams (e.g., "the cat" for bigram)
            - 'char': Character n-grams (e.g., "th", "he" for bigrams of "the")
            Character n-grams are useful for short texts, cross-language analysis,
            or capturing morphological patterns.
        transform: Feature transformation - 'zscore' or 'tfidf'.
        distance: Distance metric - 'cosine' (default), 'burrows_delta', 'manhattan',
            'euclidean', or 'eder_delta'.
        classifier: Classification method - 'delta' or 'svm'.
        cull: Minimum document frequency ratio (0.0-1.0). N-grams appearing in fewer
              than cull*100% of documents are removed before feature selection.
        chunk_size: If set, split documents into chunks of this many tokens.
        mode: Attribution mode for delta classifier - 'centroid' or 'instance'.
    
    Example usage:
        >>> stylo = Stylometry(n_features=100, ngram_range=(1, 2), cull=0.2)
        >>> stylo.fit_transform({'AuthorA': [doc1, doc2], 'AuthorB': [doc3, doc4]})
        >>> stylo.predict(disputed_text)
        >>> stylo.plot()
        >>> stylo.dendrogram()
        
        # Using character n-grams:
        >>> stylo = Stylometry(n_features=200, ngram_range=(2, 4), ngram_type='char')
    """
    
    # Registries for extensibility
    DISTANCE_FUNCTIONS = {
        'burrows_delta': burrows_delta,
        'cosine': cosine_distance,
        'manhattan': manhattan_distance,
        'euclidean': euclidean_distance,
        'eder_delta': eder_delta,
    }
    
    VALID_TRANSFORMS = ('zscore', 'tfidf')
    VALID_CLASSIFIERS = ('delta', 'svm')
    VALID_MODES = ('centroid', 'instance')
    VALID_NGRAM_TYPES = ('word', 'char')
    VALID_CLUSTERING_METHODS = ('single', 'complete', 'average', 'weighted', 'ward')
    
    def __init__(
        self,
        n_features: int = 100,
        ngram_range: Tuple[int, int] = (1, 1),
        ngram_type: str = 'word',
        transform: str = 'zscore',
        distance: str = 'cosine',
        classifier: str = 'delta',
        cull: Optional[float] = None,
        chunk_size: Optional[int] = None,
        mode: str = 'centroid',
    ):
        # Validate n_features
        if not isinstance(n_features, int):
            raise TypeError(f"n_features must be an integer, got {type(n_features).__name__}")
        if n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {n_features}")
        
        # Validate ngram_range
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
            raise TypeError(f"ngram_range must be a tuple of (min_n, max_n), got {ngram_range}")
        min_n, max_n = ngram_range
        if not (isinstance(min_n, int) and isinstance(max_n, int)):
            raise TypeError(f"ngram_range values must be integers")
        if min_n < 1 or max_n < min_n:
            raise ValueError(f"ngram_range must satisfy 1 <= min_n <= max_n, got {ngram_range}")
        
        # Validate ngram_type
        if ngram_type not in self.VALID_NGRAM_TYPES:
            raise ValueError(f"ngram_type must be one of {self.VALID_NGRAM_TYPES}, got '{ngram_type}'")
        
        # Validate transform
        if transform not in self.VALID_TRANSFORMS:
            raise ValueError(f"transform must be one of {self.VALID_TRANSFORMS}, got '{transform}'")
        
        # Validate distance metric
        if distance not in self.DISTANCE_FUNCTIONS:
            raise ValueError(f"distance must be one of {list(self.DISTANCE_FUNCTIONS.keys())}, got '{distance}'")
        
        # Validate classifier
        if classifier not in self.VALID_CLASSIFIERS:
            raise ValueError(f"classifier must be one of {self.VALID_CLASSIFIERS}, got '{classifier}'")
        
        # Validate cull
        if cull is not None:
            if not isinstance(cull, (int, float)) or not (0.0 < cull < 1.0):
                raise ValueError(f"cull must be a float between 0 and 1, got {cull}")
        
        # Validate chunk_size
        if chunk_size is not None:
            if not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
        
        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        
        # Store parameters
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.ngram_type = ngram_type
        self.transform_type = transform
        self.distance_metric = distance
        self.classifier = classifier
        self.cull = cull
        self.chunk_size = chunk_size
        self.mode = mode
        
        # Feature vocabulary learned from corpus
        self.features: List[str] = []  # Selected n-gram features
        self.feature_means: Optional[np.ndarray] = None  # For z-score
        self.feature_stds: Optional[np.ndarray] = None   # For z-score
        self.idf_weights: Optional[np.ndarray] = None    # For TF-IDF
        
        # Author information
        self.authors: List[str] = []
        
        # Document storage - extensible structure
        # Each doc has: tokens, ngrams, ngram_freqs, yule_k, etc.
        self._doc_features: Dict[str, Dict] = {}
        
        # Transformed vectors for classification
        self.document_vectors: List[np.ndarray] = []
        self.document_labels: List[str] = []  # Author name for each document
        self.document_ids: List[str] = []     # Unique ID for each document
        
        # Centroid mode: one vector per author
        self.author_centroids: Dict[str, np.ndarray] = {}
        
        # Mapping from doc_id to index for fast lookup
        self._doc_id_to_index: Dict[str, int] = {}
        
        # SVM model (fitted on demand)
        self._svm_model = None
        
        self._is_fitted: bool = False
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _get_distance_fn(self, distance: Optional[str] = None) -> Tuple[Callable, str]:
        """Get the distance function, using provided metric or falling back to default."""
        metric = distance if distance is not None else self.distance_metric
        if metric not in self.DISTANCE_FUNCTIONS:
            raise ValueError(f"distance must be one of {list(self.DISTANCE_FUNCTIONS.keys())}, got '{metric}'")
        return self.DISTANCE_FUNCTIONS[metric], metric
    
    def _validate_tokens(self, tokens: List[str], name: str = "tokens") -> None:
        """Validate that tokens is a non-empty list of strings."""
        if not isinstance(tokens, list):
            raise TypeError(f"{name} must be a list, got {type(tokens).__name__}")
        if not tokens:
            raise ValueError(f"{name} cannot be empty")
        # Only check first few for performance
        for i, token in enumerate(tokens[:10]):
            if not isinstance(token, str):
                raise TypeError(f"Token {i} must be a string, got {type(token).__name__}")
    
    def _validate_level(self, level: str) -> None:
        """Validate the level parameter."""
        if level not in ('document', 'author'):
            raise ValueError(f"level must be 'document' or 'author', got '{level}'")
    
    # =========================================================================
    # N-gram Extraction
    # =========================================================================
    
    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Extract n-grams from a list of tokens based on ngram_range and ngram_type.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of n-gram strings.
            - For word n-grams: joined with space for n > 1 (e.g., "the cat")
            - For char n-grams: concatenated characters (e.g., "th", "he")
        """
        min_n, max_n = self.ngram_range
        ngrams = []
        
        if self.ngram_type == 'word':
            # Word n-grams
            for n in range(min_n, max_n + 1):
                if n == 1:
                    ngrams.extend(tokens)
                else:
                    for i in range(len(tokens) - n + 1):
                        ngram = ' '.join(tokens[i:i + n])
                        ngrams.append(ngram)
        else:
            # Character n-grams
            # Concatenate all tokens into a single string (preserving word boundaries)
            text = ''.join(tokens)
            for n in range(min_n, max_n + 1):
                for i in range(len(text) - n + 1):
                    ngram = text[i:i + n]
                    ngrams.append(ngram)
        
        return ngrams
    
    # =========================================================================
    # Culling (Sparsity Filter)
    # =========================================================================
    
    def _apply_culling(
        self, 
        corpus_ngram_counts: Counter, 
        doc_ngram_sets: List[set],
    ) -> Counter:
        """
        Remove n-grams that appear in fewer than cull% of documents.
        
        Args:
            corpus_ngram_counts: Total n-gram counts across corpus
            doc_ngram_sets: List of sets, each containing unique n-grams in a document
            
        Returns:
            Filtered Counter with sparse n-grams removed
        """
        if self.cull is None:
            return corpus_ngram_counts
        
        n_docs = len(doc_ngram_sets)
        min_docs = int(np.ceil(self.cull * n_docs))
        
        # Count document frequency for each n-gram
        doc_freq = Counter()
        for ngram_set in doc_ngram_sets:
            doc_freq.update(ngram_set)
        
        # Filter: keep only n-grams appearing in >= min_docs documents
        filtered = Counter()
        for ngram, count in corpus_ngram_counts.items():
            if doc_freq[ngram] >= min_docs:
                filtered[ngram] = count
        
        if len(filtered) == 0:
            warnings.warn(
                f"Culling with cull={self.cull} removed all n-grams. "
                f"Consider lowering the cull threshold.",
                UserWarning
            )
        
        return filtered
    
    # =========================================================================
    # Document Chunking
    # =========================================================================
    
    def _chunk_documents(
        self, 
        corpus: Dict[str, List[List[str]]],
    ) -> Dict[str, List[List[str]]]:
        """
        Split documents into chunks of chunk_size tokens.
        
        Args:
            corpus: Dict mapping author -> list of tokenized documents
            
        Returns:
            New corpus dict with chunked documents
        """
        if self.chunk_size is None:
            return corpus
        
        chunked_corpus: Dict[str, List[List[str]]] = {}
        
        for author, documents in corpus.items():
            chunks = []
            for doc in documents:
                # Split document into chunks
                for i in range(0, len(doc), self.chunk_size):
                    chunk = doc[i:i + self.chunk_size]
                    if chunk:  # Include even small remainder chunks
                        chunks.append(chunk)
            chunked_corpus[author] = chunks
        
        return chunked_corpus
    
    # =========================================================================
    # Transformation Methods
    # =========================================================================
    
    def _compute_zscore(self, freq_vector: np.ndarray) -> np.ndarray:
        """Convert a frequency vector to z-scores using corpus statistics."""
        return (freq_vector - self.feature_means) / self.feature_stds
    
    def _compute_tfidf(self, freq_vector: np.ndarray) -> np.ndarray:
        """Apply TF-IDF weighting to a frequency vector."""
        return freq_vector * self.idf_weights
    
    def _transform_vector(self, freq_vector: np.ndarray) -> np.ndarray:
        """Transform frequency vector according to configured transform type."""
        if self.transform_type == 'zscore':
            return self._compute_zscore(freq_vector)
        elif self.transform_type == 'tfidf':
            return self._compute_tfidf(freq_vector)
        else:
            return freq_vector
    
    # =========================================================================
    # Vector Resolution
    # =========================================================================
    
    def _tokens_to_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Transform raw tokens to a feature vector using fitted features and transformation.
        """
        ngrams = self._extract_ngrams(tokens)
        freq_dict = get_relative_frequencies(ngrams)
        freq_vector = np.array([freq_dict.get(f, 0.0) for f in self.features])
        return self._transform_vector(freq_vector)
    
    def _get_author_vector(self, author: str) -> np.ndarray:
        """
        Get the vector for an author.
        
        In centroid mode, returns the precomputed centroid.
        In instance mode, returns the mean of all document vectors for that author.
        """
        if self.mode == 'centroid':
            return self.author_centroids[author]
        else:
            author_indices = [i for i, lbl in enumerate(self.document_labels) if lbl == author]
            author_vecs = [self.document_vectors[i] for i in author_indices]
            return np.mean(author_vecs, axis=0)
    
    def _resolve_to_vector(self, query: Union[str, List[str]]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Resolve a query (doc_id or tokens) to a feature vector.
        
        Returns:
            Tuple of (vector, doc_id if query was a doc_id else None)
        """
        if isinstance(query, str):
            # It's a doc_id
            if query not in self._doc_id_to_index:
                partial_matches = [doc_id for doc_id in self.document_ids if query in doc_id]
                if partial_matches:
                    hint = f"Did you mean one of: {partial_matches[:10]}"
                    if len(partial_matches) > 10:
                        hint += f" ... ({len(partial_matches)} matches)"
                else:
                    hint = f"Available: {self.document_ids[:10]}"
                    if len(self.document_ids) > 10:
                        hint += f" ... ({len(self.document_ids)} total)"
                raise ValueError(f"Unknown document ID '{query}'. {hint}")
            idx = self._doc_id_to_index[query]
            return self.document_vectors[idx], query
        elif isinstance(query, list):
            self._validate_tokens(query, "query")
            return self._tokens_to_vector(query), None
        else:
            raise TypeError(f"query must be a string (doc_id) or list of tokens, got {type(query).__name__}")
    
    def _get_vectors_and_labels(self, level: str) -> Tuple[List[np.ndarray], List[str]]:
        """Get vectors and labels for the specified level."""
        self._validate_level(level)
        
        if level == 'document':
            return self.document_vectors, self.document_ids.copy()
        else:  # level == 'author'
            vectors = [self._get_author_vector(author) for author in self.authors]
            return vectors, self.authors.copy()
    
    # =========================================================================
    # Fit / Transform
    # =========================================================================
    
    def fit_transform(
        self, 
        corpus: Union[Dict[str, List[List[str]]], List[List[str]]],
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Fit the model on a corpus and transform documents to feature vectors.
        
        Args:
            corpus: Either:
                - Dict mapping author names to their documents (supervised):
                  {'AuthorA': [[tok1, tok2, ...], [tok1, ...]], 'AuthorB': [...]}
                - List of tokenized documents (unsupervised):
                  [[tok1, tok2, ...], [tok1, ...], ...]
            labels: Optional list of labels for list input. Documents sharing
                    the same label are grouped together.
        
        Pipeline:
        1. Apply chunking (if chunk_size is set)
        2. Extract n-grams from all documents
        3. Apply culling (remove sparse n-grams)
        4. Select top n_features by frequency
        5. Compute feature vectors for each document
        6. Apply transformation (z-score or TF-IDF)
        7. Compute author centroids (if mode='centroid')
        """
        # Convert list input to dict format
        if isinstance(corpus, list):
            corpus = self._list_to_dict(corpus, labels)
        elif not isinstance(corpus, dict):
            raise TypeError(f"corpus must be a dict or list, got {type(corpus).__name__}")
        
        # Validate corpus
        if not corpus:
            raise ValueError("corpus cannot be empty")
        
        for author, documents in corpus.items():
            if not isinstance(author, str):
                raise TypeError(f"Author keys must be strings")
            if not isinstance(documents, list) or len(documents) == 0:
                raise ValueError(f"Author '{author}' must have at least one document")
            for i, doc in enumerate(documents):
                if not isinstance(doc, list) or len(doc) == 0:
                    raise ValueError(f"Document {i} for '{author}' must be non-empty list of tokens")
        
        # Step 1: Apply chunking
        corpus = self._chunk_documents(corpus)
        
        # Validate we have enough documents
        total_docs = sum(len(docs) for docs in corpus.values())
        if total_docs < 2:
            raise ValueError("corpus must contain at least 2 documents (after chunking)")
        
        self.authors = list(corpus.keys())
        
        # Check for imbalanced corpus sizes
        if len(self.authors) >= 2:
            author_token_counts = {}
            for author, documents in corpus.items():
                total_tokens = sum(len(doc) for doc in documents)
                author_token_counts[author] = total_tokens
            
            min_tokens = min(author_token_counts.values())
            max_tokens = max(author_token_counts.values())
            
            if min_tokens > 0 and max_tokens >= 3 * min_tokens:
                min_author = min(author_token_counts, key=author_token_counts.get)
                max_author = max(author_token_counts, key=author_token_counts.get)
                ratio = max_tokens / min_tokens
                warnings.warn(
                    f"Imbalanced corpus: '{max_author}' has {max_tokens:,} tokens while "
                    f"'{min_author}' has only {min_tokens:,} tokens ({ratio:.1f}x difference). "
                    f"This may skew feature selection toward the larger corpus.",
                    UserWarning
                )
        
        # Step 2: Extract n-grams and build corpus-wide counts
        corpus_ngram_counts = Counter()
        doc_ngram_sets: List[set] = []
        
        # Temporary storage for per-document data
        temp_doc_data: List[Dict] = []
        
        for author, documents in corpus.items():
            for i, tokens in enumerate(documents):
                ngrams = self._extract_ngrams(tokens)
                doc_ngram_set = set(ngrams)
                doc_ngram_sets.append(doc_ngram_set)
                corpus_ngram_counts.update(ngrams)
                
                # Generate doc_id
                if len(documents) == 1 and self.chunk_size is None:
                    doc_id = author
                else:
                    doc_id = f"{author}_{i + 1}"
                
                temp_doc_data.append({
                    'doc_id': doc_id,
                    'author': author,
                    'tokens': tokens,
                    'ngrams': ngrams,
                    'yule_k': compute_yule_k(tokens),
                })
        
        # Step 3: Apply culling
        filtered_counts = self._apply_culling(corpus_ngram_counts, doc_ngram_sets)
        
        if len(filtered_counts) == 0:
            raise ValueError("No features remaining after culling. Lower the cull threshold.")
        
        # Step 4: Select top n_features
        self.features = extract_mfw(filtered_counts, min(self.n_features, len(filtered_counts)))
        
        if len(self.features) < self.n_features:
            warnings.warn(
                f"Only {len(self.features)} features available (requested {self.n_features}). "
                f"Consider lowering n_features or cull threshold.",
                UserWarning
            )
        
        # Step 5: Compute frequency vectors for each document
        all_freq_vectors = []
        doc_freq_counts = np.zeros(len(self.features))  # For IDF calculation
        
        for doc_data in temp_doc_data:
            freq_dict = get_relative_frequencies(doc_data['ngrams'])
            freq_vector = np.array([freq_dict.get(f, 0.0) for f in self.features])
            doc_data['freq_vector'] = freq_vector
            all_freq_vectors.append(freq_vector)
            
            # Track document frequency for IDF
            doc_freq_counts += (freq_vector > 0).astype(float)
        
        freq_matrix = np.array(all_freq_vectors)
        n_docs = len(temp_doc_data)
        
        # Step 6: Compute transformation statistics
        if self.transform_type == 'zscore':
            self.feature_means = np.mean(freq_matrix, axis=0)
            self.feature_stds = np.std(freq_matrix, axis=0)
            self.feature_stds[self.feature_stds < 1e-10] = 1.0  # Avoid division by zero
        elif self.transform_type == 'tfidf':
            # IDF = log(N / df) where df is document frequency
            doc_freq_counts[doc_freq_counts == 0] = 1  # Avoid division by zero
            self.idf_weights = np.log(n_docs / doc_freq_counts)
        
        # Step 7: Build final document storage and transformed vectors
        self._doc_features = {}
        self.document_vectors = []
        self.document_labels = []
        self.document_ids = []
        self._doc_id_to_index = {}
        
        for idx, doc_data in enumerate(temp_doc_data):
            doc_id = doc_data['doc_id']
            author = doc_data['author']
            
            # Store document features (extensible dict)
            self._doc_features[doc_id] = {
                'tokens': doc_data['tokens'],
                'ngrams': doc_data['ngrams'],
                'freq_vector': doc_data['freq_vector'],
                'yule_k': doc_data['yule_k'],
                # Future: add more features here
            }
            
            # Transform and store vector
            transformed = self._transform_vector(doc_data['freq_vector'])
            self.document_vectors.append(transformed)
            self.document_labels.append(author)
            self.document_ids.append(doc_id)
            self._doc_id_to_index[doc_id] = idx
        
        # Step 8: Compute author centroids (for centroid mode)
        if self.mode == 'centroid':
            for author in self.authors:
                author_indices = [i for i, lbl in enumerate(self.document_labels) if lbl == author]
                # Average the raw frequency vectors, then transform
                author_freq_vecs = [temp_doc_data[i]['freq_vector'] for i in author_indices]
                avg_freq = np.mean(author_freq_vecs, axis=0)
                self.author_centroids[author] = self._transform_vector(avg_freq)
        
        # Reset SVM model (needs retraining)
        self._svm_model = None
        
        self._is_fitted = True
    
    def transform(self, tokens: List[str], warn_oov: bool = True) -> np.ndarray:
        """
        Transform a tokenized text to a feature vector using fitted features.
        
        Args:
            tokens: List of tokens (a tokenized document)
            warn_oov: If True (default), warn when the text has low overlap with
                      the trained features (less than 50% of n-grams recognized).
        
        Returns:
            Feature vector (numpy array)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        self._validate_tokens(tokens)
        
        # Check OOV ratio and warn if needed
        if warn_oov:
            ngrams = self._extract_ngrams(tokens)
            if ngrams:
                feature_set = set(self.features)
                known_count = sum(1 for ng in ngrams if ng in feature_set)
                overlap_ratio = known_count / len(ngrams)
                
                if overlap_ratio < 0.5:
                    warnings.warn(
                        f"Low feature overlap: only {known_count}/{len(ngrams)} "
                        f"({overlap_ratio:.1%}) of the text's n-grams match trained features. "
                        f"Results may be unreliable. This can happen when the text is from "
                        f"a very different domain or style than the training corpus.",
                        UserWarning
                    )
        
        return self._tokens_to_vector(tokens)
    
    def _list_to_dict(
        self, 
        documents: List[List[str]], 
        labels: Optional[List[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        """Convert a list of documents to dict format, grouped by label."""
        if not documents:
            raise ValueError("documents cannot be empty")
        
        if labels is None:
            return {'unk': documents}
        
        if len(labels) != len(documents):
            raise ValueError(f"labels length ({len(labels)}) must match documents length ({len(documents)})")
        
        result: Dict[str, List[List[str]]] = {}
        for label, doc in zip(labels, documents):
            if not isinstance(label, str):
                raise TypeError(f"labels must be strings, got {type(label).__name__}")
            if label not in result:
                result[label] = []
            result[label].append(doc)
        
        return result
    
    # =========================================================================
    # Prediction Methods
    # =========================================================================
    
    def predict(
        self, 
        text: List[str],
        k: int = 1,
        distance: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely author for a tokenized text.
        
        Args:
            text: List of tokens (the disputed text)
            k: Number of top results to return.
            distance: Distance metric override (for delta classifier).
            classifier: Classifier override ('delta' or 'svm').
        
        Returns:
            List of (author, score) tuples.
            - For 'delta': score is distance (lower = more similar)
            - For 'svm': score is probability (higher = more likely)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        self._validate_tokens(text, "text")
        
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        clf = classifier if classifier is not None else self.classifier
        
        if clf == 'delta':
            return self._predict_delta(text, k, distance)
        elif clf == 'svm':
            return self._predict_svm(text, k)
        else:
            raise ValueError(f"Unknown classifier: {clf}")
    
    def _predict_delta(
        self, 
        text: List[str], 
        k: int,
        distance: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Delta (distance-based) prediction."""
        text_vector = self._tokens_to_vector(text)
        distance_fn, _ = self._get_distance_fn(distance)
        
        if self.mode == 'centroid':
            results = []
            for author in self.authors:
                dist = distance_fn(text_vector, self.author_centroids[author])
                results.append((author, float(dist)))
            results.sort(key=lambda x: x[1])
            return results[:k]
        else:  # mode == 'instance'
            distances = []
            for i, doc_vector in enumerate(self.document_vectors):
                dist = distance_fn(text_vector, doc_vector)
                distances.append((self.document_labels[i], self.document_ids[i], float(dist)))
            distances.sort(key=lambda x: x[2])
            results = [(author, dist) for author, doc_id, dist in distances[:k]]
            return results
    
    def _predict_svm(self, text: List[str], k: int) -> List[Tuple[str, float]]:
        """SVM prediction with probability output."""
        if self._svm_model is None:
            self._fit_svm()
        
        text_vector = self._tokens_to_vector(text).reshape(1, -1)
        
        # Get probability predictions
        probas = self._svm_model.predict_proba(text_vector)[0]
        classes = self._svm_model.classes_
        
        # Sort by probability (descending)
        results = [(str(cls), float(prob)) for cls, prob in zip(classes, probas)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def _fit_svm(self) -> None:
        """Fit SVM classifier on current document vectors."""
        from sklearn.svm import SVC
        
        X = np.array(self.document_vectors)
        y = np.array(self.document_labels)
        
        # SVM with probability=True for predict_proba
        # Use global seed if set, otherwise use None for sklearn's default behavior
        seed = resolve_seed(None)
        self._svm_model = SVC(kernel='linear', probability=True, random_state=seed)
        self._svm_model.fit(X, y)
    
    def predict_author(
        self, 
        text: List[str], 
        k: int = 1, 
        distance: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> str:
        """
        Convenience method to get just the predicted author name.
        
        For 'instance' mode with k > 1 and delta classifier, returns majority vote.
        """
        clf = classifier if classifier is not None else self.classifier
        results = self.predict(text, k=k, distance=distance, classifier=classifier)
        
        if clf == 'delta' and self.mode == 'instance' and k > 1:
            # Majority vote for instance mode
            author_counts = Counter(author for author, _ in results)
            return author_counts.most_common(1)[0][0]
        else:
            return results[0][0]
    
    def predict_confidence(
        self, 
        text: List[str], 
        k: int = 1,
        classifier: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Predict with unified confidence scores (higher = more likely).
        
        Abstracts away the difference between delta (distance) and SVM (probability).
        
        Returns:
            List of (author, confidence) tuples where confidence is 0-1, higher = more likely.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        clf = classifier if classifier is not None else self.classifier
        
        if clf == 'svm':
            # SVM already returns probabilities
            return self.predict(text, k=k, classifier='svm')
        else:
            # Delta: convert distances to confidences
            results = self.predict(text, k=len(self.authors), distance=None, classifier='delta')
            
            # Convert distances to similarities using softmax-like normalization
            distances = np.array([dist for _, dist in results])
            # Avoid overflow by subtracting max
            exp_neg_dist = np.exp(-distances + distances.min())
            confidences = exp_neg_dist / exp_neg_dist.sum()
            
            conf_results = [(author, float(conf)) for (author, _), conf in zip(results, confidences)]
            return conf_results[:k]
    
    # =========================================================================
    # Bootstrap Analysis
    # =========================================================================
    
    def bootstrap_predict(
        self,
        text: List[str],
        n_iter: int = 100,
        sample_ratio: float = 0.8,
        distance: Optional[str] = None,
    ) -> Dict:
        """
        Bootstrap analysis for prediction robustness.
        
        Resamples features n_iter times and computes prediction statistics
        to assess how robust the attribution is.
        
        Args:
            text: List of tokens (the disputed text)
            n_iter: Number of bootstrap iterations
            sample_ratio: Fraction of features to use per iteration (0.0-1.0)
            distance: Distance metric override
        
        Returns:
            Dict with:
                - 'prediction': Most frequent prediction across iterations
                - 'confidence': Proportion of iterations agreeing with top prediction
                - 'distribution': Dict of author -> proportion of iterations
                - 'distances': Dict of author -> (mean_distance, std_distance)
                - 'n_iterations': Number of iterations performed
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        self._validate_tokens(text, "text")
        
        if not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError(f"n_iter must be a positive integer, got {n_iter}")
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"sample_ratio must be between 0 and 1, got {sample_ratio}")
        
        distance_fn, _ = self._get_distance_fn(distance)
        n_features = len(self.features)
        sample_size = max(1, int(n_features * sample_ratio))
        
        # Extract n-grams from text once
        text_ngrams = self._extract_ngrams(text)
        text_freq_dict = get_relative_frequencies(text_ngrams)
        
        # Track predictions and distances per iteration
        predictions = []
        author_distances: Dict[str, List[float]] = {author: [] for author in self.authors}
        
        rng = np.random.default_rng(42)
        
        for _ in range(n_iter):
            # Sample features
            feature_indices = rng.choice(n_features, size=sample_size, replace=False)
            sampled_features = [self.features[i] for i in feature_indices]
            
            # Build text vector with sampled features
            text_freq_vec = np.array([text_freq_dict.get(f, 0.0) for f in sampled_features])
            
            # Transform using sampled feature statistics
            if self.transform_type == 'zscore':
                sampled_means = self.feature_means[feature_indices]
                sampled_stds = self.feature_stds[feature_indices]
                text_vec = (text_freq_vec - sampled_means) / sampled_stds
            elif self.transform_type == 'tfidf':
                sampled_idf = self.idf_weights[feature_indices]
                text_vec = text_freq_vec * sampled_idf
            else:
                text_vec = text_freq_vec
            
            # Compare to each author centroid (using sampled features)
            best_author = None
            best_dist = float('inf')
            
            for author in self.authors:
                if self.mode == 'centroid':
                    author_full_vec = self.author_centroids[author]
                else:
                    author_full_vec = self._get_author_vector(author)
                
                author_vec = author_full_vec[feature_indices]
                dist = distance_fn(text_vec, author_vec)
                author_distances[author].append(float(dist))
                
                if dist < best_dist:
                    best_dist = dist
                    best_author = author
            
            predictions.append(best_author)
        
        # Compute statistics
        prediction_counts = Counter(predictions)
        top_prediction = prediction_counts.most_common(1)[0][0]
        confidence = prediction_counts[top_prediction] / n_iter
        
        distribution = {author: count / n_iter for author, count in prediction_counts.items()}
        # Ensure all authors are in distribution
        for author in self.authors:
            if author not in distribution:
                distribution[author] = 0.0
        
        distances_stats = {
            author: (float(np.mean(dists)), float(np.std(dists)))
            for author, dists in author_distances.items()
        }
        
        return {
            'prediction': top_prediction,
            'confidence': confidence,
            'distribution': distribution,
            'distances': distances_stats,
            'n_iterations': n_iter,
        }
    
    # =========================================================================
    # Rolling Delta Analysis
    # =========================================================================
    
    def rolling_delta(
        self,
        text: List[str],
        reference: Optional[str] = None,
        window: int = 5000,
        step: int = 1000,
        distance: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Rolling window analysis across a long text.
        
        Computes distance to a reference at each window position,
        useful for detecting authorship changes or style variation within a text.
        
        Args:
            text: List of tokens (the long text to analyze)
            reference: Author name to compare against. If None, compares each
                      window to the average representation of the entire text
                      (self-comparison mode for detecting internal variation).
            window: Window size in tokens
            step: Step size for sliding window
            distance: Distance metric override
            show: If True, display plot
            figsize: Figure size for plot
            title: Plot title
            filename: If provided, save figure to this path
        
        Returns:
            DataFrame with columns:
                - 'position': Starting token position of window
                - 'distance': Distance to reference
                - 'end_position': Ending token position of window
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        self._validate_tokens(text, "text")
        
        if reference is not None and reference not in self.authors:
            raise ValueError(f"Unknown reference author '{reference}'. Known authors: {self.authors}")
        
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")
        if not isinstance(step, int) or step < 1:
            raise ValueError(f"step must be a positive integer, got {step}")
        
        if len(text) < window:
            raise ValueError(f"text length ({len(text)}) is shorter than window size ({window})")
        
        distance_fn, _ = self._get_distance_fn(distance)
        
        # Determine reference vector
        if reference is None:
            # Self-comparison mode: use average of entire text
            ref_vector = self._tokens_to_vector(text)
            ref_label = "text average"
        else:
            # Compare to fitted author
            if self.mode == 'centroid':
                ref_vector = self.author_centroids[reference]
            else:
                ref_vector = self._get_author_vector(reference)
            ref_label = reference
        
        # Compute rolling delta
        results = []
        positions = range(0, len(text) - window + 1, step)
        
        for pos in positions:
            window_tokens = text[pos:pos + window]
            window_vector = self._tokens_to_vector(window_tokens)
            dist = distance_fn(window_vector, ref_vector)
            
            results.append({
                'position': pos,
                'distance': float(dist),
                'end_position': pos + window,
            })
        
        df = pd.DataFrame(results)
        
        # Plot if requested
        if show or filename:
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(df['position'], df['distance'], 'b-', linewidth=1.5, alpha=0.8)
            ax.fill_between(df['position'], df['distance'], alpha=0.3)
            
            ax.set_xlabel('Token Position', fontsize=12)
            ax.set_ylabel(f'Distance to {ref_label}', fontsize=12)
            
            if title:
                ax.set_title(title, fontsize=14)
            else:
                mode_str = "self-comparison" if reference is None else f"vs {reference}"
                ax.set_title(f'Rolling Delta ({mode_str}, window={window}, step={step})', fontsize=14)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add horizontal line for mean
            mean_dist = df['distance'].mean()
            ax.axhline(y=mean_dist, color='r', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_dist:.4f}')
            ax.legend()
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, bbox_inches='tight', dpi=300)
            
            if show:
                plt.show()
            else:
                plt.close()
        
        return df
    
    # =========================================================================
    # Similarity / Distance Methods
    # =========================================================================
    
    def most_similar(
        self, 
        query: Union[str, List[str]], 
        k: Optional[int] = None,
        return_distance: bool = False,
        distance: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar documents to a query.
        
        Args:
            query: Document ID (str) or list of tokens.
            k: Number of results to return. If None, returns all.
            return_distance: If False, returns similarity. If True, returns distance.
            distance: Distance metric override.
        
        Returns:
            List of (doc_id, value) tuples sorted by similarity (most similar first).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        query_vector, query_doc_id = self._resolve_to_vector(query)
        distance_fn, metric = self._get_distance_fn(distance)
        
        results = []
        for i, doc_vector in enumerate(self.document_vectors):
            doc_id = self.document_ids[i]
            if query_doc_id is not None and doc_id == query_doc_id:
                continue
            dist = distance_fn(query_vector, doc_vector)
            
            if return_distance:
                results.append((doc_id, float(dist)))
            else:
                similarity = self._distance_to_similarity(dist, metric)
                results.append((doc_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=(not return_distance))
        
        if k is not None:
            results = results[:k]
        
        return results
    
    def _distance_to_similarity(self, dist: float, metric: Optional[str] = None) -> float:
        """Convert distance to similarity based on the metric."""
        metric = metric if metric is not None else self.distance_metric
        if metric == 'cosine':
            return 1.0 - dist
        else:
            return 1.0 / (1.0 + dist)
    
    def distance(
        self, 
        a: Union[str, List[str]], 
        b: Union[str, List[str]],
        distance: Optional[str] = None,
    ) -> float:
        """
        Compute the distance between two documents. Lower = more similar.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        vector_a, _ = self._resolve_to_vector(a)
        vector_b, _ = self._resolve_to_vector(b)
        
        distance_fn, _ = self._get_distance_fn(distance)
        return float(distance_fn(vector_a, vector_b))
    
    def similarity(
        self, 
        a: Union[str, List[str]], 
        b: Union[str, List[str]],
        distance: Optional[str] = None,
    ) -> float:
        """
        Compute the similarity between two documents. Higher = more similar.
        """
        _, metric = self._get_distance_fn(distance)
        return self._distance_to_similarity(self.distance(a, b, distance=distance), metric)
    
    # =========================================================================
    # Distance Matrix / Clustering
    # =========================================================================
    
    def _compute_distance_matrix(
        self, 
        vectors: List[np.ndarray], 
        distance: Optional[str] = None,
    ) -> np.ndarray:
        """Compute pairwise distance matrix for a list of vectors."""
        from scipy.spatial.distance import cdist
        
        _, metric = self._get_distance_fn(distance)
        vectors_array = np.array(vectors)
        
        # Map our metric names to scipy's cdist metrics
        metric_map = {
            'burrows_delta': 'cityblock',
            'cosine': 'cosine',
            'manhattan': 'cityblock',
            'euclidean': 'euclidean',
            'eder_delta': 'euclidean',  # Eder's delta uses euclidean, then normalizes
        }
        
        scipy_metric = metric_map[metric]
        dist_matrix = cdist(vectors_array, vectors_array, metric=scipy_metric)
        
        # Apply metric-specific post-processing
        n_features = vectors_array.shape[1]
        if metric == 'burrows_delta':
            # Burrows' Delta: mean absolute difference
            dist_matrix = dist_matrix / n_features
        elif metric == 'eder_delta':
            # Eder's Delta: sqrt(sum((a-b)^2) / n) = euclidean / sqrt(n)
            dist_matrix = dist_matrix / np.sqrt(n_features)
        
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix
    
    def distance_matrix(
        self, 
        level: str = 'document', 
        distance: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise distance matrix from fitted data.
        
        Args:
            level: 'document' for individual documents, 'author' for author profiles
            distance: Distance metric override.
        
        Returns:
            (distance_matrix, labels)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        vectors, labels = self._get_vectors_and_labels(level)
        return self._compute_distance_matrix(vectors, distance=distance), labels
    
    def hierarchical_clustering(
        self,
        method: str = 'average',
        level: str = 'document',
        distance: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Perform hierarchical clustering on fitted data.
        
        Args:
            method: Linkage method - 'single', 'complete', 'average', 'weighted', or 'ward'
            level: 'document' or 'author'
            distance: Distance metric override.
        
        Returns:
            (linkage_matrix, labels)
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        if method not in self.VALID_CLUSTERING_METHODS:
            raise ValueError(f"method must be one of {self.VALID_CLUSTERING_METHODS}, got '{method}'")
        
        vectors, doc_labels = self._get_vectors_and_labels(level)
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 items for hierarchical clustering")
        
        dist_matrix = self._compute_distance_matrix(vectors, distance=distance)
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method=method)
        
        return linkage_matrix, doc_labels
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def vocabulary_stats(self) -> pd.DataFrame:
        """
        Get vocabulary richness statistics for all fitted documents.
        
        Returns:
            DataFrame with columns: doc_id, author, yule_k, token_count, type_count
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        rows = []
        for doc_id in self.document_ids:
            doc_data = self._doc_features[doc_id]
            tokens = doc_data['tokens']
            author = self.document_labels[self._doc_id_to_index[doc_id]]
            
            rows.append({
                'doc_id': doc_id,
                'author': author,
                'yule_k': doc_data['yule_k'],
                'token_count': len(tokens),
                'type_count': len(set(tokens)),
                'ttr': len(set(tokens)) / len(tokens) if tokens else 0.0,
            })
        
        return pd.DataFrame(rows)
    
    def get_author_profile(self, author: str) -> pd.DataFrame:
        """
        Get the feature values for a specific author.
        
        Returns a DataFrame with 'feature' and 'value' columns, sorted by value descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        if author not in self.authors:
            raise ValueError(f"Unknown author '{author}'. Known authors: {self.authors}")
        
        vector = self._get_author_vector(author)
        
        return pd.DataFrame({
            'feature': self.features,
            'value': vector,
        }).sort_values('value', ascending=False)
    
    def get_feature_comparison(self) -> pd.DataFrame:
        """
        Get a comparison table of feature values across all fitted authors.
        
        Returns a DataFrame with one column per author plus a 'variance' column.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        data = {'feature': self.features}
        
        for author in self.authors:
            data[author] = self._get_author_vector(author)
        
        df = pd.DataFrame(data)
        author_cols = [col for col in df.columns if col != 'feature']
        df['variance'] = df[author_cols].var(axis=1)
        
        return df.sort_values('variance', ascending=False)
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def plot(
        self,
        method: str = 'pca',
        level: str = 'document',
        figsize: Tuple[int, int] = (10, 8),
        show_labels: bool = True,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        marker_size: int = 100,
        fontsize: int = 12,
        filename: Optional[str] = None,
        random_state: int = 42,
        show: bool = True,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Create a 2D scatter plot of documents or authors.
        
        Args:
            method: Dimensionality reduction - 'pca', 'tsne', or 'mds'
            level: 'document' for individual documents, 'author' for author profiles
            figsize: Figure size as (width, height)
            show_labels: Whether to show text labels on points
            labels: Custom labels for points
            title: Custom title
            colors: Dict mapping author names to colors
            marker_size: Size of scatter points
            fontsize: Base font size
            filename: If provided, save figure to this path
            random_state: Random seed for t-SNE/MDS
            show: If True, display plot. If False, return (fig, ax).
        
        Returns:
            None if show=True, otherwise (fig, ax) tuple.
        """
        import matplotlib
        
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        if method not in ('pca', 'tsne', 'mds'):
            raise ValueError(f"method must be 'pca', 'tsne', or 'mds', got '{method}'")
        
        vectors, doc_labels = self._get_vectors_and_labels(level)
        author_for_point = self.document_labels if level == 'document' else self.authors
        unique_authors = self.authors
        
        if labels is not None:
            if len(labels) != len(doc_labels):
                raise ValueError(f"labels length ({len(labels)}) must match number of points ({len(doc_labels)})")
            doc_labels = list(labels)
        
        vectors = np.array(vectors)
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 items to create a plot")
        
        coords, axis_labels = self._reduce_dimensions(vectors, method, random_state)
        
        if coords.ndim == 1 or coords.shape[1] == 1:
            coords = np.column_stack([coords.flatten(), np.zeros(len(vectors))])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        is_unsupervised = len(unique_authors) == 1 and unique_authors[0] == 'unk'
        
        if is_unsupervised:
            self._plot_unsupervised(ax, coords, doc_labels, show_labels, marker_size, fontsize)
        else:
            cmap = matplotlib.colormaps['tab10']
            self._plot_supervised(
                ax, coords, doc_labels, author_for_point, unique_authors,
                colors, show_labels, marker_size, fontsize, cmap
            )
        
        ax.set_xlabel(axis_labels[0], fontsize=fontsize + 2)
        ax.set_ylabel(axis_labels[1], fontsize=fontsize + 2)
        
        if title:
            ax.set_title(title, fontsize=fontsize + 4)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        if show:
            plt.show()
            return None
        else:
            return fig, ax
    
    def _reduce_dimensions(
        self,
        vectors: np.ndarray,
        method: str,
        random_state: int,
    ) -> Tuple[np.ndarray, Tuple[str, str]]:
        """Reduce high-dimensional vectors to 2D for visualization."""
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=min(2, len(vectors)))
            coords = reducer.fit_transform(vectors)
            var_explained = reducer.explained_variance_ratio_
            if len(var_explained) >= 2:
                axis_labels = (
                    f'PC1 ({var_explained[0]*100:.1f}% variance)',
                    f'PC2 ({var_explained[1]*100:.1f}% variance)'
                )
            else:
                axis_labels = (f'PC1 ({var_explained[0]*100:.1f}% variance)', 'PC2')
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = min(30, len(vectors) - 1)
            reducer = TSNE(n_components=2, perplexity=max(1, perplexity), random_state=random_state)
            coords = reducer.fit_transform(vectors)
            axis_labels = ('t-SNE 1', 't-SNE 2')
        else:  # mds
            from sklearn.manifold import MDS
            reducer = MDS(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(vectors)
            axis_labels = ('MDS 1', 'MDS 2')
        
        return coords, axis_labels
    
    def _plot_unsupervised(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        doc_labels: List[str],
        show_labels: bool,
        marker_size: int,
        fontsize: int,
    ) -> None:
        """Plot points for unsupervised mode."""
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c='steelblue', s=marker_size,
            edgecolors='black', linewidths=0.5, alpha=0.7
        )
        if show_labels:
            for i, label in enumerate(doc_labels):
                ax.annotate(
                    label, (coords[i, 0], coords[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=fontsize - 2, alpha=0.8
                )
    
    def _plot_supervised(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        doc_labels: List[str],
        author_for_point: List[str],
        unique_authors: List[str],
        colors: Optional[Dict[str, str]],
        show_labels: bool,
        marker_size: int,
        fontsize: int,
        cmap,
    ) -> None:
        """Plot points for supervised mode."""
        if colors is None:
            color_map = {author: cmap(i % 10) for i, author in enumerate(unique_authors)}
        else:
            color_map = colors
        
        plotted_authors = set()
        for i, (label, author) in enumerate(zip(doc_labels, author_for_point)):
            color = color_map.get(author, 'gray')
            legend_label = author if author not in plotted_authors else None
            plotted_authors.add(author)
            
            ax.scatter(
                coords[i, 0], coords[i, 1],
                c=[color], s=marker_size,
                label=legend_label, edgecolors='black', linewidths=0.5
            )
            
            if show_labels:
                ax.annotate(
                    label, (coords[i, 0], coords[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=fontsize - 2, alpha=0.8
                )
        
        ax.legend(loc='best', fontsize=fontsize - 2)
    
    def dendrogram(
        self,
        method: str = 'average',
        level: str = 'document',
        orientation: str = 'top',
        figsize: Tuple[int, int] = (12, 8),
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        fontsize: int = 10,
        color_threshold: Optional[float] = None,
        filename: Optional[str] = None,
        show: bool = True,
        distance: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Visualize hierarchical clustering as a dendrogram.
        
        Args:
            method: Linkage method
            level: 'document' or 'author'
            orientation: 'top', 'bottom', 'left', or 'right'
            figsize: Figure size
            labels: Custom labels for leaves
            title: Plot title
            fontsize: Font size for labels
            color_threshold: Distance threshold for coloring
            filename: If provided, save figure to this path
            show: If True, display plot. If False, return result dict.
            distance: Distance metric override.
        
        Returns:
            None if show=True, otherwise dict with 'fig', 'ax', and dendrogram data.
        """
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        valid_orientations = ('top', 'bottom', 'left', 'right')
        if orientation not in valid_orientations:
            raise ValueError(f"orientation must be one of {valid_orientations}, got '{orientation}'")
        
        linkage_matrix, doc_labels = self.hierarchical_clustering(
            method=method,
            level=level,
            distance=distance,
        )
        
        if labels is not None:
            if len(labels) != len(doc_labels):
                raise ValueError(f"labels length ({len(labels)}) must match number of leaves ({len(doc_labels)})")
            doc_labels = list(labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        dendro_result = scipy_dendrogram(
            linkage_matrix,
            labels=doc_labels,
            orientation=orientation,
            leaf_font_size=fontsize,
            color_threshold=color_threshold,
            ax=ax,
        )
        
        if title:
            ax.set_title(title, fontsize=fontsize + 4)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        if show:
            plt.show()
            return None
        else:
            dendro_result['fig'] = fig
            dendro_result['ax'] = ax
            return dendro_result
