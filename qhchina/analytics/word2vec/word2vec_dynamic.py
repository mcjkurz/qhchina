"""
DynamicWord2Vec: Word2Vec with time-sliced embeddings and temporal regularization.

This module provides DynamicWord2Vec which maintains separate embedding matrices for each
time slice with temporal ℓ2 regularization to track smooth semantic change over time.

Unlike TempRefWord2Vec (which tags words as "word_period"), DynamicWord2Vec maintains
separate embedding matrices U[t] and V[t] for each time slice with shared vocabulary.
During training, each (center, context, time_id) tuple updates only that time slice's
embeddings, with temporal regularization pulling embeddings toward adjacent slices.

Reference:
    Yoon Kim, Yi-I Chiu, Kentaro Hanaki, Darshan Hegde, and Slav Petrov. 2014.
    Temporal Analysis of Language through Neural Language Models.
    In Proceedings of the ACL 2014 Workshop on Language Technologies and
    Computational Social Science.
"""

import logging
import pickle
import numpy as np
from collections import Counter
from collections.abc import Iterable
from .word2vec_base import Word2Vec
from .word2vec_utils import word2vec_c, TemporalSentenceIterator, SingleCorpusTemporalIterator
from ..vectors import cosine_similarity
from ...config import resolve_seed

logger = logging.getLogger("qhchina.analytics.dynamic_word2vec")

__all__ = [
    'DynamicWord2Vec',
]


def _filter_words(
    words: list[str],
    filters: dict | None,
) -> list[str]:
    """
    Apply word-level filters (stopwords, length, reference whitelist).

    This is a pre-filter applied once before the per-transition loop.
    Frequency-based filters (``vocab_top_n``, ``min_word_count``) are
    handled per-transition in :meth:`calculate_semantic_change`.

    Args:
        words: Candidate words to filter.
        filters: Dict with zero or more of the following keys:
            - ``min_word_length`` (int): Minimum character length.
            - ``stopwords`` (set): Words to exclude outright.
            - ``reference_words`` (list/set): Explicit whitelist.

    Returns:
        Filtered list of words (same order as input).
    """
    if not filters:
        return words

    valid_keys = {'min_word_count', 'min_word_length', 'stopwords', 'reference_words', 'vocab_top_n'}
    bad_keys = set(filters) - valid_keys
    if bad_keys:
        raise ValueError(f"Unknown filter key(s): {bad_keys}. Valid keys: {valid_keys}")

    ref_words = filters.get('reference_words')
    if ref_words is not None:
        ref_set = set(ref_words)
        words = [w for w in words if w in ref_set]

    min_len   = filters.get('min_word_length')
    stopwords = filters.get('stopwords') or set()

    result = []
    for word in words:
        if word in stopwords:
            continue
        if min_len is not None and len(word) < min_len:
            continue
        result.append(word)
    return result


class DynamicWord2Vec(Word2Vec):
    """
    Word2Vec with time-sliced embeddings for diachronic semantic analysis.

    Maintains separate embedding matrices $U^{(t)}$ and $V^{(t)}$
    for each time slice *t* with a shared vocabulary.  Two training modes are
    available:

    - **joint** (default): All periods train simultaneously with interleaved
      batches.  Temporal ℓ₂ regularization pulls adjacent slices together
      (see below).  Optional Procrustes alignment after training removes
      residual rotational drift between slices.
    - **sequential**: Each period is trained in order, initialising from the
      previous period's trained embeddings (Kim et al. 2014).  Alignment is
      implicit via the initialisation chain; no regularisation or Procrustes
      is needed.

    **Temporal regularization (joint mode):**
    The standard skip-gram loss is augmented with an ℓ₂ penalty that
    encourages embeddings in adjacent time slices to stay close:

    $$\\mathcal{L} = \\mathcal{L}_{\\text{SG}} + \\lambda \\sum_{t=1}^{T-1} \\| U^{(t)} - U^{(t-1)} \\|_F^2$$

    where $\\mathcal{L}_{\\text{SG}}$ is the skip-gram negative sampling
    objective, $\\lambda$ is ``temporal_lambda``, and $\\| \\cdot \\|_F$
    is the Frobenius norm.  When ``temporal_reg_V=True``, the same penalty
    is applied to the context matrix $V$.  Regularization is applied once
    per unique (word, time) pair per batch to avoid frequency-dependent
    regularization strength.

    **Architecture:**
    - U: Center/input embeddings [T, vocab_size, vector_size]
    - V: Context/output embeddings [T, vocab_size, vector_size]
    - Shared vocabulary across all time slices

    Args:
        sentences: Dictionary mapping time period labels to corpora. Each value must
            be an iterable of tokenized sentences (untagged).
            Format: ``{"label1": [["w1", "w2"], ...], "label2": LineSentenceFile("file.txt"), ...}``
        training_mode: ``"joint"`` (default) or ``"sequential"``.
        temporal_lambda: Regularization strength for joint mode (default: 0.1).
        temporal_reg_V: If True, also regularize V in joint mode (default: True).
        procrustes_align: Apply Procrustes alignment after joint training
            (default: True).  Ignored in sequential mode.
        sampling_strategy: How to sample from corpora during training:
            - "balanced" (default): Equal tokens from each corpus, stops at smallest corpus.
            - "proportional": Proportional tokens from each corpus, uses all data.
        **kwargs: Arguments passed to Word2Vec (vector_size, window, epochs, etc.).

    Example:
        from qhchina.analytics import DynamicWord2Vec, LineSentenceFile

        corpora = {
            "1800s": LineSentenceFile("corpus_1800s.txt"),
            "1900s": LineSentenceFile("corpus_1900s.txt"),
            "2000s": LineSentenceFile("corpus_2000s.txt"),
        }

        # Joint training with Procrustes alignment
        model = DynamicWord2Vec(
            sentences=corpora,
            training_mode="joint",
            temporal_lambda=0.1,
            vector_size=100,
            window=5,
            epochs=5,
        )
        model.train()

        # Query embeddings per period
        vec = model.get_vector("economy", time_label="2000s")
        drift = model.calculate_temporal_drift("economy")

        # Sequential training (Kim et al. 2014)
        model = DynamicWord2Vec(
            sentences=corpora,
            training_mode="sequential",
            epochs=5,
        )
        model.train()
    """

    def __init__(
        self,
        sentences: dict[str, Iterable[list[str]]],
        training_mode: str = "joint",
        temporal_lambda: float = 0.1,
        temporal_reg_V: bool = True,
        procrustes_align: bool = True,
        sampling_strategy: str = "balanced",
        _skip_init: bool = False,
        **kwargs
    ):
        if _skip_init:
            self.labels = []
            self.label2idx = {}
            self.num_time_slices = 0
            self.training_mode = training_mode
            self.temporal_lambda = temporal_lambda
            self.temporal_reg_V = temporal_reg_V
            self.procrustes_align = procrustes_align
            self._sampling_strategy = sampling_strategy
            self.period_vocab_counts = {}
            self._corpora = None
            self.U = None
            self.V = None
            super().__init__(_skip_init=True, **kwargs)
            return

        if not isinstance(sentences, dict):
            raise TypeError(
                f"sentences must be a dictionary mapping labels to corpora, "
                f"got {type(sentences).__name__}"
            )

        if not sentences:
            raise ValueError("sentences cannot be empty")

        if 'sg' in kwargs and kwargs['sg'] != 1:
            raise NotImplementedError("DynamicWord2Vec only supports Skip-gram (sg=1)")
        kwargs['sg'] = 1

        if training_mode not in ("joint", "sequential"):
            raise ValueError(
                f"training_mode must be 'joint' or 'sequential', got {training_mode!r}"
            )

        if not isinstance(temporal_lambda, (int, float)) or temporal_lambda < 0:
            raise ValueError("temporal_lambda must be a non-negative number")

        if sampling_strategy not in ("balanced", "proportional"):
            raise ValueError(
                f"sampling_strategy must be 'balanced' or 'proportional', got {sampling_strategy!r}"
            )

        # Reject shuffle (the streaming iterator handles interleaving and shuffling)
        if kwargs.get('shuffle') is True:
            raise ValueError(
                "shuffle=True is not supported for DynamicWord2Vec. "
                "The streaming iterator already interleaves and shuffles sentences across periods."
            )

        verbose = kwargs.get('verbose', False)

        self.labels = list(sentences.keys())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_time_slices = len(self.labels)
        self.training_mode = training_mode
        self.temporal_lambda = temporal_lambda
        self.temporal_reg_V = temporal_reg_V
        self.procrustes_align = procrustes_align
        self._sampling_strategy = sampling_strategy

        self._corpora = {}
        for label, corpus in sentences.items():
            self._corpora[label] = corpus

        self.U = None
        self.V = None

        if verbose:
            logger.info(f"Initializing DynamicWord2Vec ({training_mode} mode) with {self.num_time_slices} time slices:")
            for label, corpus in self._corpora.items():
                if hasattr(corpus, 'sentence_count'):
                    logger.info(f"  {label}: {corpus.sentence_count:,} documents, {corpus.token_count:,} tokens (file)")
                else:
                    logger.info(f"  {label}: {len(corpus):,} documents (in-memory)")
            if training_mode == "joint":
                logger.info(f"Temporal regularization: λ={temporal_lambda}, apply_to_V={temporal_reg_V}")
                logger.info(f"Procrustes alignment after training: {procrustes_align}")
            else:
                logger.info("Sequential training: each slice initialises from the previous")

        super().__init__(**kwargs)

    def build_vocab(self, sentences: Iterable[list[str]] | None = None) -> None:
        """
        Build vocabulary by iterating through all corpora.

        Creates a shared vocabulary across all time slices. Word counts are computed
        respecting the sampling strategy to match training data distribution.

        Args:
            sentences: Ignored. DynamicWord2Vec uses internal corpora instead.
                Accepted for API compatibility with the parent class.
        """
        self._count_words()
        self._filter_and_map_vocab()

    def _count_words(self, sentences: Iterable[list[str]] | None = None) -> None:
        """
        Count word occurrences from all corpora respecting the sampling strategy.

        With "balanced" strategy: counts only up to min_token_count per corpus.
        With "proportional" strategy: counts all tokens from each corpus.

        This ensures word frequencies, subsampling thresholds, and negative
        sampling distributions match the actual training data.

        Args:
            sentences: Ignored. DynamicWord2Vec uses internal corpora instead.
                Accepted for API compatibility with the parent class.

        Raises:
            ValueError: If corpora contain no words.
        """
        # Determine token counts for each corpus
        token_counts = {}
        for label, corpus in self._corpora.items():
            if hasattr(corpus, 'token_count'):
                token_counts[label] = corpus.token_count
            else:
                token_counts[label] = sum(len(sent) for sent in corpus)

        # Set token limits based on strategy
        if self._sampling_strategy == "balanced":
            min_count = min(token_counts.values())
            token_limits = {label: min_count for label in token_counts}
        else:  # proportional
            token_limits = token_counts

        self.period_vocab_counts = {}
        self.word_counts = Counter()

        for label, corpus in self._corpora.items():
            period_counter = Counter()
            tokens_counted = 0
            token_limit = token_limits[label]

            for sentence in corpus:
                remaining = token_limit - tokens_counted
                if remaining <= 0:
                    break

                # Truncate sentence if it would exceed the limit
                if len(sentence) > remaining:
                    sentence = sentence[:remaining]

                period_counter.update(sentence)
                tokens_counted += len(sentence)

            self.period_vocab_counts[label] = period_counter
            self.word_counts.update(period_counter)

        if not self.word_counts:
            raise ValueError("Corpora contain no words.")

        if self.verbose:
            logger.info("Word counts per period:")
            for label, counter in self.period_vocab_counts.items():
                logger.info(
                    f"  {label}: {len(counter):,} unique tokens, "
                    f"{sum(counter.values()):,} total tokens"
                )

    def _initialize_vectors(self) -> None:
        """
        Initialize 3D embedding tensors U and V.

        U: Center word embeddings [T, vocab_size, vector_size]
        V: Context word embeddings [T, vocab_size, vector_size]

        All time slices are initialized with the same base embeddings to maintain
        a shared coordinate system. This allows temporal regularization to work
        effectively.
        """
        vocab_size = len(self.vocab)
        T = self.num_time_slices

        if self.verbose:
            logger.info(
                f"Initializing 3D embedding tensors: "
                f"U[{T}, {vocab_size:,}, {self.vector_size}], "
                f"V[{T}, {vocab_size:,}, {self.vector_size}]"
            )

        # Resolve seed consistently with model-level RNG behavior
        init_rng = np.random.default_rng(seed=resolve_seed(self.seed))

        # Sample base embeddings once
        base_U = init_rng.random((vocab_size, self.vector_size), dtype=self.dtype)
        base_U *= 2.0
        base_U -= 1.0
        base_U /= self.vector_size

        # Replicate to all time slices
        self.U = np.tile(base_U, (T, 1, 1))  # [T, vocab_size, vector_size]

        # Output matrix starts at zero for stable negative sampling convergence
        self.V = np.zeros((T, vocab_size, self.vector_size), dtype=self.dtype)

        # Also set W and W_prime for compatibility (will be replaced during training)
        # We set them to the first time slice by default
        self.W = self.U[0]
        self.W_prime = self.V[0]

        # Work buffers reused across batches
        self._work = np.zeros(self.vector_size, dtype=self.dtype)
        self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)

    def _get_thread_working_mem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Allocate private work buffer for a worker thread.

        DynamicWord2Vec only needs the work buffer (Skip-gram only, no neu1
        or context_buffer needed). Returns (work, None, None) to match parent.
        """
        work = np.zeros(self.vector_size, dtype=self.dtype)
        return work, None, None

    def _train_batch_worker(
        self,
        batch: list,
        sample_ints: np.ndarray,
        alpha: float,
        random_seed: int,
        calculate_loss: bool,
        work: np.ndarray,
        neu1: np.ndarray,
        context_buffer: np.ndarray,
    ) -> tuple[float, int, int]:
        """
        Train on a single batch with temporal information using provided work buffers.

        This method is called by worker threads and uses thread-private buffers.
        The batch contains TemporalSentence objects with time_idx attribute.

        Args:
            batch: List of TemporalSentence objects.
            sample_ints: Subsampling thresholds array.
            alpha: Learning rate for this batch.
            random_seed: Random seed for this batch.
            calculate_loss: Whether to compute loss.
            work: Thread-private work buffer.
            neu1: Unused (temporal training is Skip-gram only).
            context_buffer: Unused (temporal training is Skip-gram only).

        Returns:
            Tuple of (batch_loss, batch_examples, batch_words).
        """
        # Extract sentences and time indices from TemporalSentence objects
        # TemporalSentence is a list subclass, so we can use it directly as tokens
        batch_with_time = [(list(sent), sent.time_idx) for sent in batch]

        # Flatten U and V for Cython
        U_flat = self.U.reshape(-1)
        V_flat = self.V.reshape(-1)

        batch_loss, batch_examples, batch_words, _ = word2vec_c.train_batch_dynamic(
            U_flat,
            V_flat,
            self.num_time_slices,
            len(self.vocab),
            batch_with_time,
            self.vocab,
            sample_ints,
            self._alias_prob,
            self._alias_index,
            work,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            alpha,
            self.negative,
            self.temporal_lambda,
            self.temporal_reg_V,
            random_seed,
            calculate_loss,
        )
        return batch_loss, batch_examples, batch_words

    def train(self) -> float | None:
        """
        Train the DynamicWord2Vec model.

        Dispatches to joint or sequential training based on ``training_mode``.

        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        if not self.vocab:
            self.build_vocab()

        if not self.vocab:
            raise ValueError(
                "Cannot train with empty vocabulary. "
                "Try reducing min_word_count or providing more training data."
            )

        if self.U is None or self.V is None:
            self._initialize_vectors()

        if not hasattr(self, '_work') or self._work is None:
            self._work = np.zeros(self.vector_size, dtype=self.dtype)
            self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)

        self._build_alias_table()

        if self.training_mode == "sequential":
            return self._train_sequential()
        else:
            return self._train_joint()

    def _train_joint(self) -> float | None:
        """Joint training: interleaved batches from all periods with temporal regularization."""
        self._sentences = TemporalSentenceIterator(
            corpora=self._corpora,
            label2idx=self.label2idx,
            token_budget=self.batch_size,
            seed=self.seed,
            strategy=self._sampling_strategy,
        )

        result = super().train()

        if self.procrustes_align:
            self._align_procrustes()

        return result

    def _train_sequential(self) -> float | None:
        """Sequential training: train each slice in order, initialising from the previous.

        Each time slice is trained independently using the parent's training
        loop.  After slice *t* finishes, its trained ``U[t]`` and ``V[t]`` are
        copied to ``U[t+1]`` and ``V[t+1]`` as initialisation.  This keeps
        the embedding spaces naturally aligned without regularisation.

        Only one corpus is streamed at a time, so memory usage is constant.
        """
        # Determine per-corpus token limits (for balanced strategy)
        token_counts = {}
        for label, corpus in self._corpora.items():
            if hasattr(corpus, 'token_count'):
                token_counts[label] = corpus.token_count
            else:
                token_counts[label] = sum(len(sent) for sent in corpus)

        if self._sampling_strategy == "balanced":
            min_count = min(token_counts.values())
            token_limits = {label: min_count for label in self.labels}
        else:
            token_limits = {label: None for label in self.labels}

        # Save original settings that we'll temporarily override per-slice
        saved_lambda = self.temporal_lambda
        saved_corpus_word_count = self.corpus_word_count

        # No temporal regularization in sequential mode
        self.temporal_lambda = 0.0

        total_loss = 0.0
        total_examples = 0

        for t, label in enumerate(self.labels):
            if self.verbose:
                logger.info(f"Sequential training: slice {t+1}/{self.num_time_slices} ({label})")

            # Copy from previous slice to current as initialisation (skip t=0)
            if t > 0:
                self.U[t] = self.U[t - 1].copy()
                self.V[t] = self.V[t - 1].copy()

            # Set corpus_word_count to this period's count for correct progress bar / LR decay
            period_tokens = sum(self.period_vocab_counts.get(label, {}).values())
            self.corpus_word_count = period_tokens

            self._sentences = SingleCorpusTemporalIterator(
                corpus=self._corpora[label],
                time_idx=self.label2idx[label],
                token_limit=token_limits[label],
            )

            result = super().train()

            if result is not None:
                total_loss += result * period_tokens
                total_examples += period_tokens

        # Restore original settings
        self.temporal_lambda = saved_lambda
        self.corpus_word_count = saved_corpus_word_count

        if self.calculate_loss and total_examples > 0:
            return total_loss / total_examples
        return None

    def _align_procrustes(self) -> None:
        """Align embedding slices via orthogonal Procrustes (post-training).

        For each adjacent pair (t, t+1), finds the orthogonal matrix Q that
        minimises ``||U[t+1] Q - U[t]||_F`` and applies it to ``U[t+1]``
        (and ``V[t+1]``).  This removes global rotational drift between slices
        that SGD introduces, making cross-slice similarity comparisons
        meaningful.

        Alignment is performed sequentially: slice 0 is the anchor,
        slice 1 is aligned to 0, slice 2 to the (already aligned) 1, etc.
        """
        T = self.num_time_slices
        if T < 2:
            return

        if self.verbose:
            logger.info("Applying Procrustes alignment across time slices...")

        for t in range(T - 1):
            # Find Q that minimises ||U[t+1] Q - U[t]||_F
            # Solution: Q = V Uᵀ  where  U[t]ᵀ U[t+1] = U Σ Vᵀ  (SVD)
            M = self.U[t].T @ self.U[t + 1]  # [d, d]
            U_svd, _, Vt_svd = np.linalg.svd(M)
            Q = (U_svd @ Vt_svd).T  # Orthogonal rotation [d, d]

            self.U[t + 1] = self.U[t + 1] @ Q
            self.V[t + 1] = self.V[t + 1] @ Q

        # Refresh W / W_prime aliases
        self.W = self.U[0]
        self.W_prime = self.V[0]

        if self.verbose:
            logger.info("Procrustes alignment complete.")

    def get_vector(self, word: str, time_label: str, normalize: bool = False) -> np.ndarray:
        """
        Get the vector for a word at a specific time period.

        Args:
            word: Input word.
            time_label: Time period label (must be one of the labels from initialization).
            normalize: If True, return the normalized vector (unit length).

        Returns:
            Word vector as numpy array of shape (vector_size,).

        Raises:
            KeyError: If word is not in vocabulary or time_label is invalid.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")

        if time_label not in self.label2idx:
            raise KeyError(
                f"Time label '{time_label}' not found. "
                f"Available labels: {list(self.label2idx.keys())}"
            )

        t = self.label2idx[time_label]
        word_idx = self.vocab[word]
        vector = self.U[t, word_idx]

        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm

        return vector

    def get_all_time_vectors(self, word: str, normalize: bool = False) -> np.ndarray:
        """
        Get vectors for a word across all time periods.

        Args:
            word: Input word.
            normalize: If True, normalize each vector independently.

        Returns:
            Array of shape [T, vector_size] containing the word's embedding
            at each time slice.

        Raises:
            KeyError: If word is not in vocabulary.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")

        word_idx = self.vocab[word]
        vectors = self.U[:, word_idx, :]  # [T, vector_size]

        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
            vectors = vectors / norms

        return vectors

    def most_similar(
        self,
        word: str,
        time_label: str,
        topn: int = 10,
        cross_space: bool = False
    ) -> list[tuple[str, float]]:
        """
        Find the topn most similar words to the given word at a specific time period.

        Args:
            word: Input word.
            time_label: Time period label.
            topn: Number of similar words to return.
            cross_space: If False (default), compare U vs U (second-order similarity).
                If True, compare U vs V (first-order similarity).

        Returns:
            List of (word, similarity) tuples sorted by descending similarity.

        Raises:
            KeyError: If word is not in vocabulary or time_label is invalid.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")

        if time_label not in self.label2idx:
            raise KeyError(
                f"Time label '{time_label}' not found. "
                f"Available labels: {list(self.label2idx.keys())}"
            )

        t = self.label2idx[time_label]
        word_idx = self.vocab[word]
        word_vec = self.U[t, word_idx].reshape(1, -1)

        if cross_space:
            sim = cosine_similarity(word_vec, self.V[t]).flatten()
        else:
            sim = cosine_similarity(word_vec, self.U[t]).flatten()

        k = min(topn + 1, len(sim))
        top_candidates = np.argpartition(-sim, k)[:k]
        top_sorted = top_candidates[np.argsort(-sim[top_candidates])]
        return [(self.index2word[i], float(sim[i])) for i in top_sorted if i != word_idx][:topn]

    def calculate_temporal_drift(self, word: str) -> np.ndarray:
        """
        Calculate temporal drift as cosine distances between adjacent time slices.

        Args:
            word: Input word.

        Returns:
            Array of shape [T-1] where element i is the cosine distance between
            the word's embedding at time i and time i+1.

        Raises:
            KeyError: If word is not in vocabulary.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")

        word_idx = self.vocab[word]
        vectors = self.U[:, word_idx, :]  # [T, vector_size]

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized_vecs = vectors / norms

        # Calculate cosine distances between adjacent slices
        drifts = []
        for i in range(len(self.labels) - 1):
            cos_sim = np.dot(normalized_vecs[i], normalized_vecs[i+1])
            cos_dist = 1.0 - cos_sim
            drifts.append(cos_dist)

        return np.array(drifts, dtype=np.float32)

    def calculate_semantic_change(
        self,
        word: str,
        filters: dict | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Calculate semantic change by comparing similarity shifts across time periods.

        For each transition (t -> t+1), computes how the target word's similarity
        to other words changes. Positive values indicate words that became more
        similar, negative values indicate words that became less similar.

        Frequency-based filters (``vocab_top_n``, ``min_word_count``) are applied
        **per transition** using only the two adjacent periods, so that only words
        with meaningful training signal in both slices are compared.

        Args:
            word: Target word to analyze.
            filters: Optional dict to restrict which reference words are considered.
                Supported keys:

                - ``vocab_top_n`` (int): For each transition, take the top-N most
                  frequent words from each of the two adjacent periods and use
                  their union (at most 2N words per transition).
                - ``min_word_count`` (int): Minimum occurrences a word must have
                  in **both** adjacent periods to be included in that transition.
                - ``min_word_length`` (int): Minimum character length of a word.
                - ``stopwords`` (set): Words to exclude from the reference set.
                - ``reference_words`` (list/set): Explicit whitelist of reference words.

        Returns:
            Dict mapping transition names (e.g., "宋_to_明") to lists of
            (word, change) tuples sorted by descending change score.

        Raises:
            KeyError: If word is not in vocabulary.

        Example:
            changes = model.calculate_semantic_change(
                "民",
                filters={"vocab_top_n": 500, "min_word_length": 2},
            )
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")

        # Word-level filters (stopwords, length, reference whitelist)
        candidate_words = [w for w in self.vocab if w != word]
        candidate_words = _filter_words(candidate_words, filters)
        candidate_set = set(candidate_words)

        if not candidate_set:
            raise ValueError("No reference words remain after applying filters.")

        # Extract per-transition frequency filters
        vocab_top_n = (filters or {}).get('vocab_top_n')
        min_wc = (filters or {}).get('min_word_count', 0)

        if vocab_top_n is not None:
            if not isinstance(vocab_top_n, int) or vocab_top_n <= 0:
                raise ValueError("vocab_top_n must be a positive integer")
        if not isinstance(min_wc, int) or min_wc < 0:
            raise ValueError("min_word_count must be a non-negative integer")

        word_idx = self.vocab[word]

        results = {}
        for i in range(len(self.labels) - 1):
            from_label = self.labels[i]
            to_label   = self.labels[i + 1]
            transition = f"{from_label}_to_{to_label}"

            from_counts = self.period_vocab_counts.get(from_label, {})
            to_counts   = self.period_vocab_counts.get(to_label, {})

            # Build per-transition word set from the two adjacent periods
            if vocab_top_n is not None:
                top_from = {w for w, _ in Counter(from_counts).most_common(vocab_top_n)}
                top_to   = {w for w, _ in Counter(to_counts).most_common(vocab_top_n)}
                eligible = (top_from | top_to) & candidate_set
            else:
                eligible = candidate_set

            # Keep only words that meet min_word_count in both adjacent periods
            transition_words = [
                w for w in candidate_words if w in eligible
                and from_counts.get(w, 0) >= max(min_wc, 1)
                and to_counts.get(w, 0) >= max(min_wc, 1)
            ]

            if not transition_words:
                results[transition] = []
                continue

            ref_indices = [self.vocab[w] for w in transition_words]

            from_vec = self.U[i,     word_idx].reshape(1, -1)
            to_vec   = self.U[i + 1, word_idx].reshape(1, -1)

            from_ref = self.U[i,     ref_indices, :]
            to_ref   = self.U[i + 1, ref_indices, :]

            from_sims   = cosine_similarity(from_vec, from_ref)[0]
            to_sims     = cosine_similarity(to_vec,   to_ref)[0]
            sim_changes = to_sims - from_sims

            word_changes = [(transition_words[j], float(sim_changes[j])) for j in range(len(transition_words))]
            word_changes.sort(key=lambda x: x[1], reverse=True)
            results[transition] = word_changes

        return results

    def get_time_labels(self) -> list[str]:
        """
        Get the list of time period labels.

        Returns:
            List of time period labels in temporal order.
        """
        return self.labels.copy()

    def save(self, path: str) -> None:
        """
        Save the DynamicWord2Vec model to a file.

        Saves all temporal embeddings, labels, and configuration parameters.

        Args:
            path: Path to save the model.
        """
        model_data = {
            'vocab': self.vocab,
            'index2word': self.index2word,
            'word_counts': dict(self.word_counts),
            'corpus_word_count': self.corpus_word_count,
            'total_corpus_tokens': self.total_corpus_tokens,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_word_count': self.min_word_count,
            'negative': self.negative,
            'ns_exponent': self.ns_exponent,
            'cbow_mean': self.cbow_mean,
            'sg': self.sg,
            'sample': self.sample,
            'shrink_windows': self.shrink_windows,
            'max_vocab_size': self.max_vocab_size,
            'U': self.U,
            'V': self.V,
            'labels': self.labels,
            'label2idx': self.label2idx,
            'num_time_slices': self.num_time_slices,
            'training_mode': self.training_mode,
            'temporal_lambda': self.temporal_lambda,
            'temporal_reg_V': self.temporal_reg_V,
            'procrustes_align': self.procrustes_align,
            '_sampling_strategy': self._sampling_strategy,
            'period_vocab_counts': {
                label: dict(counter)
                for label, counter in self.period_vocab_counts.items()
            },
            'model_type': 'DynamicWord2Vec'
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            logger.info(f"DynamicWord2Vec model saved to {path}")
            logger.info(f"Saved data includes:")
            logger.info(f"  - Vocabulary: {len(self.vocab)} words")
            logger.info(f"  - Time slices: {self.num_time_slices} ({', '.join(self.labels)})")
            logger.info(f"  - Embedding shape: U{self.U.shape}, V{self.V.shape}")

    @classmethod
    def load(cls, path: str) -> 'DynamicWord2Vec':
        """
        Load a DynamicWord2Vec model from a file.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded DynamicWord2Vec model.

        Raises:
            ValueError: If the file doesn't contain DynamicWord2Vec data.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Check if this is a DynamicWord2Vec model
        if model_data.get('model_type') != 'DynamicWord2Vec':
            raise ValueError(
                "The loaded file does not contain a DynamicWord2Vec model. "
                "Use Word2Vec.load() for regular Word2Vec models or "
                "TempRefWord2Vec.load() for TempRefWord2Vec models."
            )

        # Get base model parameters with defaults
        shrink_windows = model_data.get('shrink_windows', False)
        sample = model_data.get('sample', 1e-3)
        max_vocab_size = model_data.get('max_vocab_size', None)
        training_mode = model_data.get('training_mode', 'joint')
        temporal_lambda = model_data.get('temporal_lambda', 0.1)
        temporal_reg_V = model_data.get('temporal_reg_V', True)
        procrustes_align = model_data.get('procrustes_align', True)
        sampling_strategy = model_data.get('_sampling_strategy', 'balanced')

        # Create model instance with _skip_init
        model = cls(
            sentences={},  # Empty dict - won't be used with _skip_init
            training_mode=training_mode,
            temporal_lambda=temporal_lambda,
            temporal_reg_V=temporal_reg_V,
            procrustes_align=procrustes_align,
            sampling_strategy=sampling_strategy,
            vector_size=model_data['vector_size'],
            window=model_data['window'],
            min_word_count=model_data['min_word_count'],
            negative=model_data['negative'],
            ns_exponent=model_data['ns_exponent'],
            cbow_mean=model_data['cbow_mean'],
            sg=model_data['sg'],
            sample=sample,
            shrink_windows=shrink_windows,
            max_vocab_size=max_vocab_size,
            _skip_init=True,
        )

        # Restore saved model state
        model.labels = model_data['labels']
        model.label2idx = model_data['label2idx']
        model.num_time_slices = model_data['num_time_slices']
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.U = model_data['U']
        model.V = model_data['V']

        # Set W and W_prime to first time slice for compatibility
        model.W = model.U[0]
        model.W_prime = model.V[0]

        # Restore corpus statistics
        model.corpus_word_count = model_data.get('corpus_word_count', sum(model.word_counts.values()))
        model.total_corpus_tokens = model_data.get('total_corpus_tokens', model.corpus_word_count)

        # Restore period vocab counts
        model.period_vocab_counts = {
            label: Counter(counts_dict)
            for label, counts_dict in model_data.get('period_vocab_counts', {}).items()
        }

        if model.verbose:
            logger.info(f"DynamicWord2Vec model loaded from {path}")
            logger.info(f"Restored data includes:")
            logger.info(f"  - Vocabulary: {len(model.vocab)} words")
            logger.info(f"  - Time slices: {model.num_time_slices} ({', '.join(model.labels)})")
            logger.info(f"  - Embedding shape: U{model.U.shape}, V{model.V.shape}")

        return model
