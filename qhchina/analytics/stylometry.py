"""
Stylometry module for authorship attribution and document clustering.

Supports supervised (with fit) and unsupervised (without fit) analysis.
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
    similarity = _cosine_similarity(vec_a, vec_b)
    # Handle case where _cosine_similarity returns a matrix (for 2D inputs)
    if hasattr(similarity, '__len__') and not isinstance(similarity, (int, float)):
        similarity = float(similarity)
    return 1.0 - similarity


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
        self.distance = distance
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
        
        self._is_fitted: bool = False
    
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
    
    def fit(self, corpus: Dict[str, List[List[str]]]) -> 'Stylometry':
        """
        Fit the model on a labeled corpus for supervised authorship analysis.
        
        Args:
            corpus: Dict mapping author names to their documents.
                    Each document is a list of tokens.
                    Example: {'author_a': [['word1', 'word2', ...], ...], ...}
        
        The fitting process:
        1. Extract most frequent words (MFW) from all documents
        2. Compute relative frequencies for each document
        3. Calculate corpus-wide mean and std for z-score normalization
        4. Depending on mode:
           - 'centroid': Aggregate each author's texts and compute one z-score profile per author
           - 'instance': Compute z-scores for each individual document
        
        Returns:
            self (for method chaining)
        """
        # Validate input
        if not isinstance(corpus, dict):
            raise TypeError(f"corpus must be a dict, got {type(corpus).__name__}")
        if not corpus:
            raise ValueError("corpus cannot be empty")
        if len(corpus) < 2:
            raise ValueError("corpus must contain at least 2 authors")
        
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
        
        for author in self.authors:
            for doc_id, rel_freq_vec in self._author_doc_rel_freqs[author].items():
                self.document_labels.append(author)
                self.document_ids.append(doc_id)
                self.document_zscores.append(self._compute_zscore(rel_freq_vec))
        
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
    
    def predict(
        self, 
        text: List[str],
        k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely author for a tokenized text.
        
        Args:
            text: List of tokens (the disputed text)
            k: Number of nearest neighbors to consider (only used in 'instance' mode)
        
        Returns:
            List of (author, distance) tuples sorted by distance ascending.
            - In 'centroid' mode: returns distances to each author centroid
            - In 'instance' mode: returns the k nearest documents, with their author labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Validate input
        if not isinstance(text, list):
            raise TypeError(f"text must be a list of tokens, got {type(text).__name__}")
        if not text:
            raise ValueError("text cannot be empty")
        for i, token in enumerate(text):
            if not isinstance(token, str):
                raise TypeError(f"Token {i} must be a string, got {type(token).__name__}")
        
        # Compute z-scores for the disputed text
        rel_freq_dict = get_relative_frequencies(text)
        text_rel_freq = np.array([rel_freq_dict.get(f, 0.0) for f in self.features])
        text_zscore = self._compute_zscore(text_rel_freq)
        
        distance_fn = self.DISTANCE_FUNCTIONS[self.distance]
        
        if self.mode == 'centroid':
            # Compare to each author centroid
            results = []
            for author in self.authors:
                dist = distance_fn(text_zscore, self.author_centroids[author])
                results.append((author, float(dist)))
            results.sort(key=lambda x: x[1])
            return results
        
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
    
    def transform(self, documents: List[List[str]], labels: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Transform documents to z-score vectors (unsupervised mode).
        
        This is for clustering/visualization without prior fitting.
        Computes MFW, means, and stds from the provided documents themselves.
        
        Args:
            documents: List of tokenized documents
            labels: Optional labels for each document (defaults to Doc_1, Doc_2, ...)
        
        Returns:
            (z_score_vectors, labels)
        """
        if not isinstance(documents, list):
            raise TypeError("documents must be a list")
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents")
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, list) or len(doc) == 0:
                raise ValueError(f"Document {i} must be a non-empty list of tokens")
        
        if labels is None:
            labels = [f"Doc_{i+1}" for i in range(len(documents))]
        elif len(labels) != len(documents):
            raise ValueError("labels length must match documents length")
        
        # Extract features from these documents
        features = extract_mfw(documents, self.n_features)
        if len(features) == 0:
            raise ValueError("No features extracted")
        
        # Compute relative frequencies
        rel_freq_vectors = []
        for doc in documents:
            rel_freq_dict = get_relative_frequencies(doc)
            rel_freq_vec = np.array([rel_freq_dict.get(f, 0.0) for f in features])
            rel_freq_vectors.append(rel_freq_vec)
        
        # Compute means and stds
        rel_freq_matrix = np.array(rel_freq_vectors)
        means = np.mean(rel_freq_matrix, axis=0)
        stds = np.std(rel_freq_matrix, axis=0)
        stds[stds < 1e-10] = 1.0
        
        # Compute z-scores
        z_scores = [(rfv - means) / stds for rfv in rel_freq_vectors]
        
        return z_scores, labels
    
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
        
        metric = metric_map[self.distance]
        dist_matrix = cdist(vectors_array, vectors_array, metric=metric)
        
        # Burrows' Delta is mean absolute difference, not sum
        if self.distance == 'burrows_delta':
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
            raise RuntimeError("Model has not been fitted. Call fit() first.")
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
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
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
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        vectors, labels = self._get_vectors_and_labels(level)
        return self._compute_distance_matrix(vectors), labels
    
    def hierarchical_clustering(
        self,
        documents: Optional[List[List[str]]] = None,
        labels: Optional[List[str]] = None,
        method: str = 'average',
        level: str = 'document',
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Perform hierarchical clustering.
        
        Two modes:
        - Unsupervised: Pass documents directly to cluster them (level param ignored)
        - Supervised: Omit documents to use fitted data at the specified level
        
        Returns:
            (linkage_matrix, labels) for scipy dendrogram
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        
        if method not in self.VALID_CLUSTERING_METHODS:
            raise ValueError(f"method must be one of {self.VALID_CLUSTERING_METHODS}, got '{method}'")
        
        if documents is not None:
            # Unsupervised mode
            vectors, doc_labels = self.transform(documents, labels)
        else:
            # Supervised mode - use fitted data
            if not self._is_fitted:
                raise RuntimeError("No documents provided and model not fitted. Call fit() first.")
            vectors, doc_labels = self._get_vectors_and_labels(level)
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 items for hierarchical clustering")
        
        dist_matrix = self._compute_distance_matrix(vectors)
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method=method)
        
        return linkage_matrix, doc_labels
    
    def plot(
        self,
        documents: Optional[List[List[str]]] = None,
        labels: Optional[List[str]] = None,
        method: str = 'pca',
        level: str = 'document',
        figsize: Tuple[int, int] = (10, 8),
        show_labels: bool = True,
        title: Optional[str] = None,
        colors: Optional[Dict[str, str]] = None,
        marker_size: int = 100,
        fontsize: int = 12,
        filename: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        """
        Create a 2D scatter plot of documents or authors.
        
        Two modes:
        - Unsupervised: Pass documents directly (level param ignored, uniform coloring)
        - Supervised: Omit documents to visualize fitted data (colored by author)
        
        Dimensionality reduction: 'pca', 'tsne', or 'mds'.
        """
        import matplotlib
        
        if method not in ('pca', 'tsne', 'mds'):
            raise ValueError(f"method must be 'pca', 'tsne', or 'mds', got '{method}'")
        
        is_unsupervised = documents is not None
        
        if is_unsupervised:
            vectors, doc_labels = self.transform(documents, labels)
            author_for_point = None
            unique_authors = None
        else:
            if not self._is_fitted:
                raise RuntimeError("No documents provided and model not fitted. Call fit() first.")
            
            vectors, doc_labels = self._get_vectors_and_labels(level)
            author_for_point = self.document_labels if level == 'document' else self.authors
            unique_authors = self.authors
        
        vectors = np.array(vectors)
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 items to create a plot")
        
        # Reduce to 2D
        coords, axis_labels = self._reduce_dimensions(vectors, method, random_state)
        
        # Handle 1D output (when only 2 items)
        if coords.ndim == 1 or coords.shape[1] == 1:
            coords = np.column_stack([coords.flatten(), np.zeros(len(vectors))])
        
        fig, ax = plt.subplots(figsize=figsize)
        
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
        else:
            mode_str = 'Documents' if is_unsupervised else ('Documents' if level == 'document' else 'Authors')
            ax.set_title(f'Stylometric {mode_str} Analysis ({method.upper()})', fontsize=fontsize + 4)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        plt.show()
    
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
        documents: Optional[List[List[str]]] = None,
        labels: Optional[List[str]] = None,
        method: str = 'average',
        level: str = 'document',
        orientation: str = 'top',
        figsize: Tuple[int, int] = (12, 8),
        fontsize: int = 10,
        filename: Optional[str] = None,
    ) -> None:
        """
        Visualize hierarchical clustering as a dendrogram.
        
        Two modes:
        - Unsupervised: Pass documents directly (level param ignored)
        - Supervised: Omit documents to use fitted data
        """
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        
        valid_orientations = ('top', 'bottom', 'left', 'right')
        if orientation not in valid_orientations:
            raise ValueError(f"orientation must be one of {valid_orientations}, got '{orientation}'")
        
        linkage_matrix, doc_labels = self.hierarchical_clustering(
            documents=documents,
            labels=labels,
            method=method,
            level=level,
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        scipy_dendrogram(
            linkage_matrix,
            labels=doc_labels,
            orientation=orientation,
            leaf_font_size=fontsize,
            ax=ax,
        )
        
        is_unsupervised = documents is not None
        mode_str = 'Unsupervised' if is_unsupervised else ('Document' if level == 'document' else 'Author')
        ax.set_title(f'Stylometric {mode_str} Clustering (method={method})', fontsize=fontsize + 4)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
