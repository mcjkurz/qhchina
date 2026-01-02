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
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .vectors import cosine_similarity as _cosine_similarity


def extract_mfw(texts: List[List[str]], n: int = 100) -> List[str]:
    """
    Extract the n Most Frequent Words (MFW) from a collection of tokenized texts.
    
    Returns a list of the n most common words across all documents.
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts must be a list, got {type(texts).__name__}")
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    
    word_counts = Counter()
    for doc in texts:
        word_counts.update(doc)
    
    return [word for word, _ in word_counts.most_common(n)]


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


def get_relative_frequencies(tokens: List[str]) -> Dict[str, float]:
    """
    Compute relative word frequencies for a tokenized text.
    
    Args:
        tokens: List of tokens
    
    Returns:
        Dict mapping each unique word to its relative frequency (count / total)
    """
    if not tokens:
        return {}
    word_counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in word_counts.items()}


class Stylometry:
    """
    Stylometry for authorship attribution and document clustering.
    
    Parameters:
        n_features: Number of most frequent words to use as features.
        distance: Distance metric - 'burrows_delta', 'cosine', or 'manhattan'.
        mode: Attribution mode - 'centroid' or 'instance'.
            - 'centroid': Aggregate all author texts into one profile per author.
                          When predicting, compare to author centroids.
            - 'instance': Keep individual texts separate.
                          When predicting, find the nearest text (k-NN style).
    
    Example usage:
        >>> stylo = Stylometry(n_features=100, distance='cosine')
        >>> # Supervised: dict with author -> documents
        >>> stylo.fit_transform({'AuthorA': [doc1, doc2], 'AuthorB': [doc3, doc4]})
        >>> # Unsupervised: list of documents
        >>> stylo.fit_transform([doc1, doc2, doc3])
        >>> # Analyze
        >>> stylo.plot()
        >>> stylo.dendrogram()
        >>> stylo.most_similar('AuthorA_1')
        >>> stylo.distance('AuthorA_1', 'AuthorB_1')
    """
    
    DISTANCE_FUNCTIONS = {
        'burrows_delta': burrows_delta,
        'cosine': cosine_distance,
        'manhattan': manhattan_distance,
    }
    
    VALID_MODES = ('centroid', 'instance')
    VALID_CLUSTERING_METHODS = ('single', 'complete', 'average', 'weighted', 'ward')
    
    def __init__(
        self,
        n_features: int = 100,
        distance: str = 'cosine',
        mode: str = 'centroid',
    ):
        # Validate n_features
        if not isinstance(n_features, int):
            raise TypeError(f"n_features must be an integer, got {type(n_features).__name__}")
        if n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {n_features}")
        
        # Validate distance metric
        if not isinstance(distance, str):
            raise TypeError(f"distance must be a string, got {type(distance).__name__}")
        if distance not in self.DISTANCE_FUNCTIONS:
            raise ValueError(f"distance must be one of {list(self.DISTANCE_FUNCTIONS.keys())}, got '{distance}'")
        
        # Validate mode
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a string, got {type(mode).__name__}")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        
        self.n_features = n_features
        self.distance_metric = distance
        self.mode = mode
        
        # Feature vocabulary learned from corpus
        self.features: List[str] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Author information
        self.authors: List[str] = []
        
        # Internal storage: documents organized by author
        # Structure: {author: {doc_id: rel_freq_vector}}
        self._author_doc_rel_freqs: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Centroid mode: one z-score vector per author
        self.author_centroids: Dict[str, np.ndarray] = {}
        
        # Flat lists for easy iteration (derived from _author_doc_rel_freqs)
        self.document_zscores: List[np.ndarray] = []
        self.document_labels: List[str] = []  # Author name for each document
        self.document_ids: List[str] = []     # Unique ID for each document
        
        # Mapping from doc_id to index for fast lookup
        self._doc_id_to_index: Dict[str, int] = {}
        
        self._is_fitted: bool = False
    
    def _validate_tokens(self, tokens: List[str], name: str = "tokens") -> None:
        """
        Validate that tokens is a non-empty list of strings.
        
        Args:
            tokens: The tokens to validate
            name: Name to use in error messages (e.g., "tokens", "text")
        
        Raises:
            TypeError: If tokens is not a list or contains non-strings
            ValueError: If tokens is empty
        """
        if not isinstance(tokens, list):
            raise TypeError(f"{name} must be a list, got {type(tokens).__name__}")
        if not tokens:
            raise ValueError(f"{name} cannot be empty")
        for i, token in enumerate(tokens):
            if not isinstance(token, str):
                raise TypeError(f"Token {i} must be a string, got {type(token).__name__}")
    
    def _compute_zscore(self, rel_freq_vector: np.ndarray) -> np.ndarray:
        """
        Convert a relative frequency vector to z-scores using corpus statistics.
        """
        return (rel_freq_vector - self.feature_means) / self.feature_stds
    
    def _get_author_zscore(self, author: str) -> np.ndarray:
        """
        Get the z-score vector for an author.
        
        In centroid mode, returns the precomputed centroid.
        In instance mode, returns the mean of all document z-scores for that author.
        """
        if self.mode == 'centroid':
            return self.author_centroids[author]
        else:
            author_indices = [i for i, lbl in enumerate(self.document_labels) if lbl == author]
            author_vecs = [self.document_zscores[i] for i in author_indices]
            return np.mean(author_vecs, axis=0)
    
    def _validate_level(self, level: str) -> None:
        """Validate the level parameter."""
        if level not in ('document', 'author'):
            raise ValueError(f"level must be 'document' or 'author', got '{level}'")
    
    def _get_vectors_and_labels(self, level: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get z-score vectors and labels for the specified level.
        
        Args:
            level: 'document' for individual documents, 'author' for author profiles
        
        Returns:
            Tuple of (vectors, labels)
        """
        self._validate_level(level)
        
        if level == 'document':
            return self.document_zscores, self.document_ids.copy()
        else:  # level == 'author'
            vectors = [self._get_author_zscore(author) for author in self.authors]
            return vectors, self.authors.copy()
    
    def _tokens_to_zscore(self, tokens: List[str]) -> np.ndarray:
        """
        Transform raw tokens to a z-score vector using fitted features and statistics.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Z-score vector
        """
        rel_freq_dict = get_relative_frequencies(tokens)
        rel_freq_vec = np.array([rel_freq_dict.get(f, 0.0) for f in self.features])
        return self._compute_zscore(rel_freq_vec)
    
    def _resolve_to_zscore(self, query: Union[str, List[str]]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Resolve a query (doc_id or tokens) to a z-score vector.
        
        Args:
            query: Either a document ID (str) or list of tokens
        
        Returns:
            Tuple of (z-score vector, doc_id if query was a doc_id else None)
        """
        if isinstance(query, str):
            # It's a doc_id
            if query not in self._doc_id_to_index:
                # Check for partial matches to provide helpful suggestions
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
            return self.document_zscores[idx], query
        elif isinstance(query, list):
            # It's tokens
            self._validate_tokens(query, "query")
            return self._tokens_to_zscore(query), None
        else:
            raise TypeError(f"query must be a string (doc_id) or list of tokens, got {type(query).__name__}")
    
    def fit_transform(
        self, 
        corpus: Union[Dict[str, List[List[str]]], List[List[str]]],
        labels: Optional[List[str]] = None,
    ) -> 'Stylometry':
        """
        Fit the model on a corpus and transform documents to z-score vectors.
        
        Args:
            corpus: Either:
                - Dict mapping author names to their documents (supervised):
                  {'AuthorA': [[tok1, tok2, ...], [tok1, ...]], 'AuthorB': [...]}
                - List of tokenized documents (unsupervised):
                  [[tok1, tok2, ...], [tok1, ...], ...]
            labels: Optional list of labels, one per document (must match corpus length).
                    Documents sharing the same label are grouped together as belonging
                    to the same author. If not provided, all documents are assigned the
                    label 'unk'. Ignored for dict input.
                    
                    Examples:
                    - labels=['A', 'A', 'B', 'B'] → groups into {'A': [doc1, doc2], 'B': [doc3, doc4]}
                    - labels=['ch1', 'ch2', 'ch3'] → each doc is its own group (for clustering)
                    - labels=None → all docs grouped as {'unk': [doc1, doc2, ...]}
        
        Document IDs are generated based on grouping: when a label has only one document,
        the label is used directly as the ID (e.g., 'chapter1'). When multiple documents
        share a label, IDs are suffixed with numbers (e.g., 'AuthorA_1', 'AuthorA_2').
        
        The process:
        1. Extract most frequent words (MFW) from all documents
        2. Compute relative frequencies for each document
        3. Calculate corpus-wide mean and std for z-score normalization
        4. Compute z-scores for all documents
        5. In centroid mode: compute author centroids
        
        Returns:
            self (for method chaining)
        """
        # Convert list input to dict format
        if isinstance(corpus, list):
            corpus = self._list_to_dict(corpus, labels)
        elif not isinstance(corpus, dict):
            raise TypeError(f"corpus must be a dict or list, got {type(corpus).__name__}")
        
        # Validate corpus
        if not corpus:
            raise ValueError("corpus cannot be empty")
        if len(corpus) < 2 and len(list(corpus.values())[0]) < 2:
            raise ValueError("corpus must contain at least 2 documents")
        
        for author, documents in corpus.items():
            if not isinstance(author, str):
                raise TypeError(f"Author keys must be strings")
            if not isinstance(documents, list) or len(documents) == 0:
                raise ValueError(f"Author '{author}' must have at least one document")
            for i, doc in enumerate(documents):
                if not isinstance(doc, list) or len(doc) == 0:
                    raise ValueError(f"Document {i} for '{author}' must be non-empty list of tokens")
        
        self.authors = list(corpus.keys())
        
        # Check for imbalanced corpus sizes and warn if necessary
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
                    f"This may skew MFW calculation toward the larger corpus. "
                    f"Consider balancing text sizes across authors.",
                    UserWarning
                )
        
        # Step 1: Collect all documents and extract MFW
        all_documents = []
        for texts in corpus.values():
            all_documents.extend(texts)
        
        self.features = extract_mfw(all_documents, self.n_features)
        if len(self.features) == 0:
            raise ValueError("No features extracted from corpus")
        
        # Step 2: Compute relative frequencies for each document, organized by author
        self._author_doc_rel_freqs = {}
        all_rel_freq_vectors = []  # Flat list for computing corpus statistics
        
        for author, texts in corpus.items():
            self._author_doc_rel_freqs[author] = {}
            for i, doc in enumerate(texts):
                # Only append index suffix if author has multiple documents
                if len(texts) == 1:
                    doc_id = author
                else:
                    doc_id = f"{author}_{i+1}"
                rel_freq_dict = get_relative_frequencies(doc)
                rel_freq_vec = np.array([rel_freq_dict.get(f, 0.0) for f in self.features])
                self._author_doc_rel_freqs[author][doc_id] = rel_freq_vec
                all_rel_freq_vectors.append(rel_freq_vec)
        
        # Step 3: Calculate corpus-wide mean and std
        rel_freq_matrix = np.array(all_rel_freq_vectors)
        self.feature_means = np.mean(rel_freq_matrix, axis=0)
        self.feature_stds = np.std(rel_freq_matrix, axis=0)
        self.feature_stds[self.feature_stds < 1e-10] = 1.0  # Avoid division by zero
        
        # Step 4: Build flat lists and compute z-scores
        self.document_labels = []
        self.document_ids = []
        self.document_zscores = []
        self._doc_id_to_index = {}
        
        for author in self.authors:
            for doc_id, rel_freq_vec in self._author_doc_rel_freqs[author].items():
                idx = len(self.document_ids)
                self.document_labels.append(author)
                self.document_ids.append(doc_id)
                self.document_zscores.append(self._compute_zscore(rel_freq_vec))
                self._doc_id_to_index[doc_id] = idx
        
        # Step 5: In centroid mode, compute author centroids
        if self.mode == 'centroid':
            for author in self.authors:
                # Average the relative frequency vectors for this author
                rel_freq_vectors = list(self._author_doc_rel_freqs[author].values())
                avg_rel_freq = np.mean(rel_freq_vectors, axis=0)
                # Convert averaged frequencies to z-scores
                self.author_centroids[author] = self._compute_zscore(avg_rel_freq)
        
        self._is_fitted = True
        return self
    
    def transform(self, tokens: List[str]) -> np.ndarray:
        """
        Transform a tokenized text to a z-score vector using fitted features.
        
        This method allows you to transform new documents after fitting,
        without modifying the model's internal state.
        
        Args:
            tokens: List of tokens (a tokenized document)
        
        Returns:
            Z-score vector (numpy array) of shape (n_features,)
        
        Example:
            >>> stylo = Stylometry(n_features=100)
            >>> stylo.fit_transform({'AuthorA': [doc1, doc2], 'AuthorB': [doc3]})
            >>> new_doc_vector = stylo.transform(['new', 'document', 'tokens'])
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        self._validate_tokens(tokens)
        
        return self._tokens_to_zscore(tokens)
    
    def _list_to_dict(
        self, 
        documents: List[List[str]], 
        labels: Optional[List[str]] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Convert a list of documents to dict format, grouped by label.
        
        Documents with the same label value are grouped together as belonging
        to the same author. If labels is None, all documents get the 'unk' label.
        
        Args:
            documents: List of tokenized documents
            labels: Optional list of labels, one per document. Documents sharing
                    the same label are grouped together.
        
        Returns:
            Dict mapping label to list of documents with that label
        
        Example:
            documents = [doc1, doc2, doc3, doc4]
            labels = ['A', 'A', 'B', 'B']
            → {'A': [doc1, doc2], 'B': [doc3, doc4]}
        """
        if not documents:
            raise ValueError("documents cannot be empty")
        
        if labels is None:
            # All documents get 'unk' label
            return {'unk': documents}
        
        if len(labels) != len(documents):
            raise ValueError(f"labels length ({len(labels)}) must match documents length ({len(documents)})")
        
        # Group documents by label
        result: Dict[str, List[List[str]]] = {}
        for label, doc in zip(labels, documents):
            if not isinstance(label, str):
                raise TypeError(f"labels must be strings, got {type(label).__name__}")
            if label not in result:
                result[label] = []
            result[label].append(doc)
        
        return result
    
    def predict(
        self, 
        text: List[str],
        k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely author for a tokenized text.
        
        Args:
            text: List of tokens (the disputed text)
            k: Number of top results to return. If k exceeds the number of 
               available items, all items are returned.
        
        Returns:
            List of (author, distance) tuples sorted by distance ascending (most similar first).
            - In 'centroid' mode: returns top k author centroids by distance
            - In 'instance' mode: returns the k nearest documents, with their author labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        # Validate input
        self._validate_tokens(text, "text")
        
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        # Compute z-scores for the disputed text
        text_zscore = self._tokens_to_zscore(text)
        
        distance_fn = self.DISTANCE_FUNCTIONS[self.distance_metric]
        
        if self.mode == 'centroid':
            # Compare to each author centroid
            results = []
            for author in self.authors:
                dist = distance_fn(text_zscore, self.author_centroids[author])
                results.append((author, float(dist)))
            results.sort(key=lambda x: x[1])
            return results[:k]
        
        else:  # mode == 'instance'
            # Compare to each individual document, find k nearest neighbors
            distances = []
            for i, doc_zscore in enumerate(self.document_zscores):
                dist = distance_fn(text_zscore, doc_zscore)
                distances.append((self.document_labels[i], self.document_ids[i], float(dist)))
            
            # Sort by distance
            distances.sort(key=lambda x: x[2])
            
            # Return k nearest with their author labels
            results = [(author, dist) for author, doc_id, dist in distances[:k]]
            return results
    
    def predict_author(self, text: List[str], k: int = 1) -> str:
        """
        Convenience method to get just the predicted author name.
        
        In 'instance' mode with k > 1, returns the majority vote among k nearest neighbors.
        """
        results = self.predict(text, k=k)
        
        if self.mode == 'centroid' or k == 1:
            return results[0][0]
        else:
            # Majority vote for instance mode with k > 1
            author_counts = Counter(author for author, _ in results)
            return author_counts.most_common(1)[0][0]
    
    def most_similar(
        self, 
        query: Union[str, List[str]], 
        k: Optional[int] = None,
        return_distance: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar documents to a query.
        
        Args:
            query: Document ID (str) or list of tokens.
            k: Number of results to return. If None, returns all.
            return_distance: If False (default), returns similarity (higher = more similar).
                           If True, returns distance (lower = more similar).
        
        Returns:
            List of (doc_id, value) tuples sorted by similarity (most similar first).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        query_zscore, query_doc_id = self._resolve_to_zscore(query)
        distance_fn = self.DISTANCE_FUNCTIONS[self.distance_metric]
        
        results = []
        for i, doc_zscore in enumerate(self.document_zscores):
            doc_id = self.document_ids[i]
            if query_doc_id is not None and doc_id == query_doc_id:
                continue
            dist = distance_fn(query_zscore, doc_zscore)
            
            if return_distance:
                results.append((doc_id, float(dist)))
            else:
                similarity = self._distance_to_similarity(dist)
                results.append((doc_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=(not return_distance))
        
        if k is not None:
            results = results[:k]
        
        return results
    
    def _distance_to_similarity(self, dist: float) -> float:
        """Convert distance to similarity based on the current metric."""
        if self.distance_metric == 'cosine':
            return 1.0 - dist  # similarity = 1 - distance, range: -1 to 1
        else:
            return 1.0 / (1.0 + dist)  # range: 0 to 1
    
    def distance(
        self, 
        a: Union[str, List[str]], 
        b: Union[str, List[str]],
    ) -> float:
        """
        Compute the distance between two documents. Lower = more similar.
        
        Args:
            a, b: Document ID (str) or list of tokens.
        
        Returns:
            Distance (float).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        zscore_a, _ = self._resolve_to_zscore(a)
        zscore_b, _ = self._resolve_to_zscore(b)
        
        distance_fn = self.DISTANCE_FUNCTIONS[self.distance_metric]
        return float(distance_fn(zscore_a, zscore_b))
    
    def similarity(
        self, 
        a: Union[str, List[str]], 
        b: Union[str, List[str]],
    ) -> float:
        """
        Compute the similarity between two documents. Higher = more similar.
        
        Args:
            a, b: Document ID (str) or list of tokens.
        
        Returns:
            Similarity (float). For cosine: -1 to 1. For others: 0 to 1.
        """
        return self._distance_to_similarity(self.distance(a, b))
    
    def _compute_distance_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise distance matrix for a list of z-score vectors."""
        from scipy.spatial.distance import cdist
        
        vectors_array = np.array(vectors)
        
        # Map our distance names to scipy metric names
        metric_map = {
            'burrows_delta': 'cityblock',  # Manhattan, then divide by n_features
            'cosine': 'cosine',
            'manhattan': 'cityblock',
        }
        
        metric = metric_map[self.distance_metric]
        dist_matrix = cdist(vectors_array, vectors_array, metric=metric)
        
        # Burrows' Delta is mean absolute difference, not sum
        if self.distance_metric == 'burrows_delta':
            dist_matrix = dist_matrix / vectors_array.shape[1]
        
        # Ensure diagonal is exactly zero (floating-point precision can cause tiny values)
        np.fill_diagonal(dist_matrix, 0.0)
        
        return dist_matrix
    
    def get_author_profile(self, author: str) -> pd.DataFrame:
        """
        Get the z-score normalized feature values for a specific author.
        
        Returns a DataFrame with 'feature' and 'zscore' columns, sorted by z-score descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        if not isinstance(author, str):
            raise TypeError(f"author must be a string, got {type(author).__name__}")
        if author not in self.authors:
            raise ValueError(f"Unknown author '{author}'. Known authors: {self.authors}")
        
        zscores = self._get_author_zscore(author)
        
        return pd.DataFrame({
            'feature': self.features,
            'zscore': zscores,
        }).sort_values('zscore', ascending=False)
    
    def get_feature_comparison(self) -> pd.DataFrame:
        """
        Get a comparison table of feature z-scores across all fitted authors.
        
        Returns a DataFrame with one column per author plus a 'variance' column.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        data = {'feature': self.features}
        
        for author in self.authors:
            data[author] = self._get_author_zscore(author)
        
        df = pd.DataFrame(data)
        author_cols = [col for col in df.columns if col != 'feature']
        df['variance'] = df[author_cols].var(axis=1)
        
        return df.sort_values('variance', ascending=False)
    
    def distance_matrix(self, level: str = 'document') -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise distance matrix from fitted data.
        
        Args:
            level: 'document' for individual documents, 'author' for author profiles
        
        Returns:
            (distance_matrix, labels)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit_transform() first.")
        
        vectors, labels = self._get_vectors_and_labels(level)
        return self._compute_distance_matrix(vectors), labels
    
    def hierarchical_clustering(
        self,
        method: str = 'average',
        level: str = 'document',
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Perform hierarchical clustering on fitted data.
        
        Args:
            method: Linkage method - 'single', 'complete', 'average', 'weighted', or 'ward'
            level: 'document' for individual documents, 'author' for author profiles
        
        Returns:
            (linkage_matrix, labels) for scipy dendrogram
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
        
        dist_matrix = self._compute_distance_matrix(vectors)
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method=method)
        
        return linkage_matrix, doc_labels
    
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
        
        Must call fit_transform() first.
        
        Args:
            method: Dimensionality reduction - 'pca', 'tsne', or 'mds'
            level: 'document' for individual documents, 'author' for author profiles
            figsize: Figure size as (width, height)
            show_labels: Whether to show text labels on points
            labels: Custom labels for points (overrides default doc_ids/author names)
            title: Custom title (auto-generated if None)
            colors: Dict mapping author names to colors
            marker_size: Size of scatter points
            fontsize: Base font size
            filename: If provided, save figure to this path
            random_state: Random seed for t-SNE/MDS
            show: If True, display plot. If False, return (fig, ax) for further editing.
        
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
        
        # Override labels if provided
        if labels is not None:
            if len(labels) != len(doc_labels):
                raise ValueError(f"labels length ({len(labels)}) must match number of points ({len(doc_labels)})")
            doc_labels = list(labels)
        
        vectors = np.array(vectors)
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 items to create a plot")
        
        # Reduce to 2D
        coords, axis_labels = self._reduce_dimensions(vectors, method, random_state)
        
        # Handle 1D output (when only 2 items)
        if coords.ndim == 1 or coords.shape[1] == 1:
            coords = np.column_stack([coords.flatten(), np.zeros(len(vectors))])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check if unsupervised (single 'unk' author)
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
        """Plot points for unsupervised mode (uniform color, no legend)."""
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
        """Plot points for supervised mode (colored by author with legend)."""
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
    ) -> Optional[Dict]:
        """
        Visualize hierarchical clustering as a dendrogram.
        
        Must call fit_transform() first.
        
        Args:
            method: Linkage method - 'single', 'complete', 'average', 'weighted', or 'ward'
            level: 'document' for individual documents, 'author' for author profiles
            orientation: Dendrogram orientation - 'top', 'bottom', 'left', or 'right'
            figsize: Figure size as (width, height)
            labels: Custom labels for leaves (overrides default doc_ids/author names)
            title: Plot title (no title if None)
            fontsize: Font size for labels
            color_threshold: Distance threshold for coloring. Links below this get cluster colors,
                           links above get uniform color. Default (None) uses 0.7 * max distance.
                           Set to 0 for uniform color, or high value to color more clusters.
            filename: If provided, save figure to this path
            show: If True, display plot. If False, return dendrogram result dict.
        
        Returns:
            None if show=True, otherwise dict with 'fig', 'ax', and scipy dendrogram data
            ('ivl', 'leaves', 'color_list', 'icoord', 'dcoord').
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
        )
        
        # Override labels if provided
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
