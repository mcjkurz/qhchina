"""
Word2Vec implementation for learning word embeddings.

Provides CBOW and Skip-gram architectures for training word vectors from text corpora.
"""

import logging
import pickle
import numpy as np
import threading
from collections import Counter, deque
from collections.abc import Callable, Iterable
from queue import Queue
from tqdm.auto import tqdm
import time
from ..vectors import cosine_similarity
from .utils import word2vec_c
from ...config import get_rng, resolve_seed
from ...helpers.texts import iter_batches

logger = logging.getLogger("qhchina.analytics.word2vec")


class Word2Vec:
    """
    Word2Vec model for learning word embeddings from text.
    
    Supports two training algorithms:
    - Skip-gram (sg=1): Predicts context words from center word. Generally better for 
      infrequent words and smaller datasets.
    - CBOW (sg=0): Predicts center word from context words. Faster to train.
    
    Training does not start automatically. Call ``train()`` explicitly after initialization
    to begin training.
    
    Args:
        sentences: Iterable of tokenized sentences, where each sentence is a list of
            string tokens. Must be restartable (can be iterated multiple times).
            For streaming from a file, use ``LineSentenceFile``.
        vector_size (int): Dimensionality of the word vectors (default: 100).
        window (int): Maximum distance between the current and predicted word (default: 5).
        min_word_count (int): Ignores all words with frequency lower than this (default: 5).
        negative (int): Number of negative samples (default: 5).
        ns_exponent (float): Exponent for negative sampling distribution (default: 0.75).
        cbow_mean (bool): If True, use mean of context word vectors, else use sum (default: True).
        sg (int): Training algorithm: 1 for skip-gram; 0 for CBOW (default: 0).
        seed (int, optional): Seed for random number generator. If None, uses global seed setting.
        alpha (float): Initial learning rate (default: 0.025).
        min_alpha (float, optional): Minimum learning rate. If None, learning rate remains constant.
        sample (float): Threshold for subsampling frequent words. Default is 1e-3, set to 0 to disable.
        shrink_windows (bool): If True, randomly vary the effective window size during training 
            (default: True).
        max_vocab_size (int, optional): Maximum vocabulary size. None means no limit.
        verbose (bool): If True, log progress information during training (default: False).
        epochs (int): Number of training iterations over the corpus. Must be specified explicitly.
        batch_size (int): Number of words per training batch (default: 10240).
        workers (int): Number of worker threads for parallel training (default: 1).
        callbacks (list of callable, optional): Callback functions to call after each epoch.
        calculate_loss (bool): Whether to calculate and return the final loss (default: True).
        total_examples (int, optional): Total number of training examples per epoch. When provided 
            along with ``min_alpha``, uses this exact value for learning rate decay calculation.
        shuffle (bool, optional): Whether to shuffle sentences before each epoch. 
            Defaults to True if sentences is a list, False otherwise.
    
    Example:
        from qhchina.analytics import Word2Vec, LineSentenceFile
        
        # From a list of tokenized sentences
        sentences = [['我', '喜欢', '学习'], ['他', '喜欢', '运动']]
        model = Word2Vec(sentences, vector_size=100, window=5, min_word_count=1, epochs=5)
        model.train()
        
        # From a text file (memory-efficient for large corpora)
        # File format: one sentence per line, tokens separated by spaces
        sentences = LineSentenceFile("corpus.txt")
        model = Word2Vec(sentences, vector_size=100, epochs=5)
        model.train()
        
        # Get word vector
        vector = model['喜欢']
        
        # Find similar words
        similar = model.most_similar('喜欢', topn=5)
    """
    
    def __init__(
        self,
        sentences: Iterable[list[str]] | None = None,
        vector_size: int = 100,
        window: int = 5,
        min_word_count: int = 5,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_mean: bool = True,
        sg: int = 0,
        seed: int | None = None,
        alpha: float = 0.025,
        min_alpha: float | None = None,
        sample: float = 1e-3,
        shrink_windows: bool = True,
        max_vocab_size: int | None = None,
        verbose: bool = False,
        epochs: int | None = None,
        batch_size: int = 10240,
        workers: int = 1,
        callbacks: list[Callable] | None = None,
        calculate_loss: bool = True,
        total_examples: int | None = None,
        shuffle: bool | None = None,
        _skip_init: bool = False,
    ):
        # _skip_init is used by load() to create an empty shell that will be populated
        # with saved state. This avoids unnecessary initialization and training.
        if _skip_init:
            self.verbose = verbose
            self.vector_size = vector_size
            self.window = window
            self.min_word_count = min_word_count
            self.negative = negative
            self.ns_exponent = ns_exponent
            self.cbow_mean = cbow_mean
            self.sg = sg
            self.seed = seed
            self.alpha = alpha
            self._initial_alpha = alpha  # Store original alpha for reset_lr=True
            self.min_alpha = min_alpha
            self.sample = sample
            self.shrink_windows = shrink_windows
            self.max_vocab_size = max_vocab_size
            self.dtype = np.float32
            self.epochs = epochs
            self.batch_size = batch_size
            self.workers = workers
            self.callbacks = callbacks
            self.calculate_loss = calculate_loss
            self.total_examples_hint = total_examples
            self.shuffle = shuffle
            # NOTE: _sentences=None is intentional for loaded models (no corpus attached).
            # Keep this branch in sync with the normal init path below when adding attributes.
            self._sentences = None
            self.vocab = {}
            self.index2word = []
            self.word_counts = Counter()
            self.corpus_word_count = 0
            self.total_corpus_tokens = 0
            self.W = None
            self.W_prime = None
            # Use effective_seed consistently with the normal initialization path
            effective_seed = resolve_seed(seed)
            self._rng = get_rng(effective_seed)
            return
        
        self.verbose = verbose
        self.vector_size = vector_size
        self.window = window
        self.min_word_count = min_word_count
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.sg = sg
        self.seed = seed
        self.alpha = alpha
        self._initial_alpha = alpha  # Store original alpha for reset_lr=True
        self.min_alpha = min_alpha
        self.sample = sample  # Threshold for subsampling
        self.shrink_windows = shrink_windows  # Dynamic window size
        self.max_vocab_size = max_vocab_size  # Maximum vocabulary size

        # Validate core hyperparameters early to prevent invalid model states.
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        if not isinstance(negative, int) or negative <= 0:
            raise ValueError("negative must be a positive integer")
        if epochs is None:
            raise ValueError("epochs must be specified explicitly (e.g. epochs=1)")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(min_word_count, int) or min_word_count < 0:
            raise ValueError("min_word_count must be a non-negative integer")
        
        # Set dtype for weight matrices (float32 for Cython BLAS compatibility)
        self.dtype = np.float32
        
        # Use isolated RNG to avoid affecting global state
        effective_seed = resolve_seed(seed)
        self._rng = get_rng(effective_seed)
        
        self._sentences = sentences
        
        # Initialize vocabulary structures
        self.vocab = {}  # word -> index (direct mapping)
        self.index2word = []  # index -> word
        self.word_counts = Counter()  # word -> count
        self.corpus_word_count = 0  # Token count for words IN vocabulary
        self.total_corpus_tokens = 0  # Total token count (including OOV words)
        
        # These will be initialized in _initialize_vectors
        self.W = None  # Input word embeddings
        self.W_prime = None  # Output word embeddings (for negative sampling)
        
        # Training configuration
        self.epochs = epochs
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self.callbacks = callbacks
        self.calculate_loss = calculate_loss
        self.total_examples_hint = total_examples
        self.shuffle = shuffle

    def build_vocab(self, sentences: Iterable[list[str]]) -> None:
        """
        Build vocabulary from sentences.
        
        Args:
            sentences: Iterable of tokenized sentences (each sentence is a list of words).
        
        Raises:
            ValueError: If sentences is empty or contains no words.
        """
        self._count_words(sentences)
        self._filter_and_map_vocab()
    
    def _expand_vocab(self, sentences: Iterable[list[str]]) -> int:
        """
        Expand vocabulary with new words from sentences.
        
        Counts words in the new sentences, merges counts with existing word_counts,
        and adds new words that meet the min_word_count threshold. Existing words
        are not removed, only updated counts and new additions.
        
        New word embeddings are initialized using the mean of existing embeddings
        (based on Hewitt 2021) to avoid breaking the pretrained distribution.
        
        Args:
            sentences: Iterable of tokenized sentences.
        
        Returns:
            Number of new words added to the vocabulary.
        
        Raises:
            ValueError: If model has no existing vocabulary or vectors.
        """
        if not self.vocab or self.W is None:
            raise ValueError(
                "Cannot expand vocabulary on an uninitialized model. "
                "Train the model first or use build_vocab()."
            )
        
        # Count words in new sentences
        new_counts = Counter()
        total_new_tokens = 0
        for sentence in sentences:
            new_counts.update(sentence)
            total_new_tokens += len(sentence)
        
        if not new_counts:
            return 0
        
        # Merge counts with existing
        old_vocab_size = len(self.vocab)
        self.word_counts.update(new_counts)
        self.total_corpus_tokens += total_new_tokens
        
        # Find genuinely new words that now meet threshold
        new_words = []
        for word, count in new_counts.items():
            if word not in self.vocab and self.word_counts[word] >= self.min_word_count:
                new_words.append(word)
        
        if not new_words:
            # Update corpus_word_count for existing vocab words
            self.corpus_word_count += sum(
                new_counts[word] for word in new_counts if word in self.vocab
            )
            return 0
        
        # Initialize new embeddings using mean of existing (Hewitt 2021)
        mu = np.mean(self.W, axis=0)
        
        # Compute covariance for optional noise
        centered = self.W - mu
        sigma = (centered.T @ centered) / len(self.W)
        
        # Sample new vectors from N(mu, sigma * 1e-5) for small noise
        # This bounds KL divergence while adding some variation
        n_new = len(new_words)
        try:
            new_vectors = self._rng.multivariate_normal(
                mu, sigma * 1e-5, size=n_new
            ).astype(self.dtype)
        except np.linalg.LinAlgError:
            # Fallback if covariance is singular: use mean with small uniform noise
            new_vectors = np.tile(mu, (n_new, 1)).astype(self.dtype)
            noise = (self._rng.random((n_new, self.vector_size)) - 0.5) * 0.01
            new_vectors += noise.astype(self.dtype)
        
        # Expand W matrix
        self.W = np.vstack([self.W, new_vectors])
        
        # Expand W_prime matrix (zeros for new words, same as initial training)
        if self.W_prime is not None:
            new_W_prime = np.zeros((n_new, self.vector_size), dtype=self.dtype)
            self.W_prime = np.vstack([self.W_prime, new_W_prime])
        
        # Update vocabulary mappings
        for word in new_words:
            self.vocab[word] = len(self.index2word)
            self.index2word.append(word)
        
        # Update corpus_word_count (existing vocab words + new words)
        self.corpus_word_count += sum(
            new_counts[word] for word in new_counts if word in self.vocab
        )
        
        if self.verbose:
            logger.info(
                f"Vocabulary expanded: {old_vocab_size:,} -> {len(self.vocab):,} words "
                f"(+{n_new:,} new)"
            )
        
        return n_new
    
    def _count_words(self, sentences: Iterable[list[str]]) -> None:
        """
        Count word occurrences from sentences.
        
        Populates self.word_counts with word frequencies. Override in subclasses
        for custom counting logic (e.g., balanced counting from multiple sources).
        
        Args:
            sentences: Iterable of tokenized sentences.
        
        Raises:
            ValueError: If sentences is empty or contains no words.
        """
        self.word_counts = Counter()
        sentence_count = 0
        
        for sentence in sentences:
            self.word_counts.update(sentence)
            sentence_count += 1
        
        if sentence_count == 0:
            raise ValueError("sentences cannot be empty")
        
        if not self.word_counts:
            raise ValueError("sentences contains no words. Provide non-empty tokenized sentences.")
    
    def _filter_and_map_vocab(self) -> None:
        """
        Filter words and create vocabulary mappings.
        
        Filters words by min_word_count and max_vocab_size, then creates
        vocab (word->index) and index2word (index->word) mappings.
        
        Uses Gensim-compatible ordering: primary sort by descending count,
        secondary sort by descending original index for ties.
        
        Requires self.word_counts to be populated first via _count_words().
        """
        # Filter words by min_word_count
        retained_words = {
            word for word, count in self.word_counts.items() 
            if count >= self.min_word_count
        }
        
        # If max_vocab_size is set, keep only the most frequent words
        if self.max_vocab_size is not None and len(retained_words) > self.max_vocab_size:
            top_words = [word for word, _ in self.word_counts.most_common(self.max_vocab_size)]
            retained_words = {word for word in top_words if word in retained_words}
        
        # Create mappings with Gensim-compatible ordering
        # Sort by: descending count, then descending original index for ties
        word_to_original_idx = {word: i for i, word in enumerate(self.word_counts.keys())}
        words_with_indices = [
            (word, self.word_counts[word], word_to_original_idx[word]) 
            for word in retained_words
        ]
        words_sorted = sorted(words_with_indices, key=lambda x: (-x[1], -x[2]))
        
        self.vocab = {}
        self.index2word = []
        for word, count, _ in words_sorted:
            self.vocab[word] = len(self.index2word)
            self.index2word.append(word)
        
        # Compute token counts
        self.total_corpus_tokens = sum(self.word_counts.values())
        self.corpus_word_count = sum(self.word_counts[word] for word in self.vocab)
        
        if self.verbose:
            logger.info(
                f"Vocabulary built: {len(self.vocab):,} words, "
                f"{self.corpus_word_count:,} tokens in vocab, "
                f"{self.total_corpus_tokens:,} total tokens"
            )

    def _estimate_example_count(self, sample_ints: np.ndarray) -> int:
        """
        Estimate total training examples per epoch for learning rate decay scheduling.
        
        Args:
            sample_ints: Pre-computed subsampling thresholds from _compute_sample_ints().
        
        Returns:
            Estimated number of training examples per epoch.
        """
        # Vocabulary coverage: probability that a random word position is in vocabulary
        if self.total_corpus_tokens > 0:
            vocab_coverage = self.corpus_word_count / self.total_corpus_tokens
        else:
            vocab_coverage = 1.0
        
        # Edge factor: accounts for sentence boundary effects where context windows
        # are truncated. This is the only remaining heuristic (~5% loss estimate).
        edge_factor = 0.95
        
        # Calculate effective word count after subsampling
        # Convert sample_ints back to keep probabilities: keep_prob = sample_ints / (2^32 - 1)
        if self.sample > 0 and sample_ints is not None and self.corpus_word_count > 0:
            # Expected retained words = sum of (count * keep_probability)
            max_uint32 = 4294967295.0
            effective_words = sum(
                self.word_counts[word] * (sample_ints[idx] / max_uint32)
                for word, idx in self.vocab.items()
            )
            # Average keep probability for context words (same distribution)
            avg_keep_prob = effective_words / self.corpus_word_count
        else:
            effective_words = self.corpus_word_count
            avg_keep_prob = 1.0
        
        # Average window size: if shrink_windows, uniform[1, window] has mean (window+1)/2
        avg_window = (self.window + 1) / 2.0 if self.shrink_windows else self.window
        
        if self.sg:  # Skip-gram
            # Each center word pairs with ~2*window context words (left + right)
            # Context words must: (1) be in vocabulary, (2) survive subsampling
            # Edge factor accounts for truncated windows at sentence boundaries
            estimated = int(effective_words * 2 * avg_window * vocab_coverage * avg_keep_prob * edge_factor)
        else:  # CBOW
            # Each center word generates 1 example (context words as input)
            # Edge factor accounts for sentence boundaries
            # Additional factor for cases where all context words are subsampled away
            context_survival = max(0.5, vocab_coverage * avg_keep_prob)
            estimated = int(effective_words * edge_factor * context_survival)
        
        return max(1, estimated)

    def _initialize_vectors(self) -> None:
        """Initialize embedding matrices and work buffers.
        
        Matrix roles in skip-gram with negative sampling:
          - W (syn0): Center word embeddings (input layer), returned for queries
          - W_prime (syn1neg): Context word embeddings (output layer), discarded
        """
        vocab_size = len(self.vocab)
        if self.verbose:
            logger.info(f"Initializing vectors: 2 matrices of shape ({vocab_size:,}, {self.vector_size})...")
        
        # Resolve seed consistently with model-level RNG behavior.
        init_rng = np.random.default_rng(seed=resolve_seed(self.seed))
        self.W = init_rng.random((vocab_size, self.vector_size), dtype=self.dtype)
        self.W *= 2.0
        self.W -= 1.0
        self.W /= self.vector_size
        
        # Output matrix starts at zero for stable negative sampling convergence
        self.W_prime = np.zeros((vocab_size, self.vector_size), dtype=self.dtype)
        
        # Work buffers reused across batches (avoids per-batch allocation)
        self._work = np.zeros(self.vector_size, dtype=self.dtype)
        self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)

    def _build_alias_table(self) -> None:
        """Build alias table for O(1) negative sampling using Vose's algorithm.
        
        Creates two arrays:
        - _alias_prob[i]: uint32 threshold for choosing word i vs its alias
        - _alias_index[i]: uint32 fallback word index when prob test fails
        
        Sampling is O(1): pick a random column i, then compare a random value
        against _alias_prob[i] to return either i or _alias_index[i].
        """
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            self._alias_prob = np.array([], dtype=np.uint32)
            self._alias_index = np.array([], dtype=np.uint32)
            return
        
        weights = np.array([
            self.word_counts[word] ** self.ns_exponent
            for word in self.index2word
        ], dtype=np.float64)
        
        self._build_alias_from_weights(weights)
    
    def _build_alias_from_weights(self, weights: np.ndarray) -> None:
        """Build alias table from a weight array using Vose's algorithm.
        
        Shared implementation used by both Word2Vec and TempRefWord2Vec.
        Zero-weight entries are handled correctly (they become pure aliases
        to other words).
        """
        n = len(weights)
        total = weights.sum()
        if total == 0:
            self._alias_prob = np.zeros(n, dtype=np.uint32)
            self._alias_index = np.zeros(n, dtype=np.uint32)
            return
        
        # Normalize so that sum of scaled probs = n
        # prob_scaled[i] = weights[i] / total * n
        prob_scaled = weights * (n / total)
        
        prob = np.zeros(n, dtype=np.uint32)
        alias = np.zeros(n, dtype=np.uint32)
        
        # Vose's algorithm: partition into "small" (< 1.0) and "large" (>= 1.0)
        small = []
        large = []
        for i in range(n):
            if prob_scaled[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Fill the alias table
        while small and large:
            s = small.pop()
            l = large.pop()
            
            # prob[s] = prob_scaled[s] * UINT32_MAX, so comparison against
            # a uniform uint32 gives the correct accept probability
            prob[s] = min(int(prob_scaled[s] * 4294967296.0), 4294967295)
            alias[s] = l
            
            prob_scaled[l] = (prob_scaled[l] + prob_scaled[s]) - 1.0
            if prob_scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        
        # Remaining entries have probability ~1.0 (always accepted)
        for l in large:
            prob[l] = 4294967295  # UINT32_MAX
            alias[l] = l
        for s in small:
            prob[s] = 4294967295
            alias[s] = s
        
        self._alias_prob = prob
        self._alias_index = alias
    
    def _get_random_state(self) -> int:
        """Get a fresh random seed for training each batch.
        
        Each batch gets a fresh seed derived from the model's NumPy RNG.
        Reusing the LCG state across batches causes training divergence.
        """
        # Combine two 24-bit randints into a 48-bit seed for the Cython LCG
        high = self._rng.randint(0, 2**24)
        low = self._rng.randint(0, 2**24)
        return (2**24) * high + low
    
    def _set_random_state(self, state: int) -> None:
        """No-op: LCG state from Cython is discarded; each batch gets fresh seed."""
        pass
    
    def _compute_sample_ints(self) -> np.ndarray:
        """
        Compute subsampling thresholds as uint32 integers for fast comparison in Cython.
        
        Returns numpy array where sample_ints[word_id] is the threshold value.
        In Cython, a word is kept if sample_ints[word_id] >= random_uint32.
        
        Higher keep probability → higher threshold → more likely to be kept.
        """
        sample_ints = np.zeros(len(self.vocab), dtype=np.uint32)
        
        if self.sample <= 0:
            # No subsampling - thresholds don't matter because use_subsampling=False
            # in the Cython call, so this array won't be used for comparisons.
            return sample_ints
        
        total_words = self.corpus_word_count
        
        # Guard against empty vocabulary (all words filtered)
        if total_words == 0:
            return sample_ints
        
        # Build counts array ordered by vocab index
        V = len(self.vocab)
        counts = np.empty(V, dtype=np.float64)
        for word, idx in self.vocab.items():
            counts[idx] = self.word_counts[word]
        
        # Google/Gensim formula: keep_prob = sqrt(t/f) + t/f
        word_freq = counts / total_words
        t_over_f = self.sample / word_freq
        keep_prob = np.sqrt(t_over_f) + t_over_f
        np.clip(keep_prob, 0.0, 1.0, out=keep_prob)
        
        # Convert to uint32 thresholds: word is kept if threshold >= random
        sample_ints = (keep_prob * 4294967295.0).clip(0, 4294967295).astype(np.uint32)
        
        return sample_ints
    
    def _get_thread_working_mem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Allocate private work buffers for a worker thread.
        
        Each worker thread needs its own work buffers to avoid race conditions.
        These buffers are used for gradient accumulation during training.
        
        Returns:
            Tuple of (work, neu1, context_buffer) numpy arrays.
        """
        work = np.zeros(self.vector_size, dtype=self.dtype)
        neu1 = np.zeros(self.vector_size, dtype=self.dtype)
        context_buffer = np.zeros(2 * self.window, dtype=np.uint32)
        return work, neu1, context_buffer
    
    def _train_batch_worker(
        self,
        batch: list[list[str]],
        sample_ints: np.ndarray,
        alpha: float,
        random_seed: int,
        calculate_loss: bool,
        work: np.ndarray,
        neu1: np.ndarray,
        context_buffer: np.ndarray,
    ) -> tuple[float, int, int]:
        """
        Train on a single batch using provided work buffers.
        
        This method is called by worker threads and uses thread-private buffers
        instead of the instance buffers. Override in subclasses for different
        training logic (e.g., TempRefWord2Vec uses temporal training).
        
        Args:
            batch: List of tokenized sentences.
            sample_ints: Subsampling thresholds array.
            alpha: Learning rate for this batch.
            random_seed: Random seed for this batch (generated by producer thread).
            calculate_loss: Whether to compute loss.
            work: Thread-private work buffer.
            neu1: Thread-private neu1 buffer.
            context_buffer: Thread-private context index buffer (CBOW only).
        
        Returns:
            Tuple of (batch_loss, batch_examples, batch_words).
        """
        batch_loss, batch_examples, batch_words, _ = word2vec_c.train_batch(
            self.W,
            self.W_prime,
            batch,
            self.vocab,
            sample_ints,
            self._alias_prob,
            self._alias_index,
            work,
            neu1,
            context_buffer,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            alpha,
            self.sg,
            self.negative,
            self.cbow_mean,
            random_seed,
            calculate_loss,
        )
        return batch_loss, batch_examples, batch_words
    
    def _worker_loop(
        self,
        job_queue: Queue,
        progress_queue: Queue,
        sample_ints: np.ndarray,
        calculate_loss: bool,
    ) -> None:
        """
        Worker thread main loop - processes training jobs from the queue.
        
        Each worker pulls batches from the job queue, trains on them using
        thread-private buffers, and reports progress back via the progress queue.
        
        Args:
            job_queue: Queue containing (batch, alpha, seed) tuples or None (poison pill).
            progress_queue: Queue for reporting (loss, examples, words) results.
            sample_ints: Subsampling thresholds array (shared, read-only).
            calculate_loss: Whether to compute and report loss values.
        """
        work, neu1, context_buffer = self._get_thread_working_mem()
        
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break
            
            batch, alpha, seed = job
            
            batch_loss, batch_examples, batch_words = self._train_batch_worker(
                batch, sample_ints, alpha, seed, calculate_loss, work, neu1, context_buffer
            )
            
            progress_queue.put((batch_loss, batch_examples, batch_words))
    
    def _job_producer(
        self,
        sentences: Iterable[list[str]],
        job_queue: Queue,
        start_alpha: float,
        min_alpha_val: float,
        decay_alpha: bool,
        total_words_all_epochs: int,
        epoch: int,
        epochs: int,
    ) -> None:
        """
        Producer thread - batches sentences and puts jobs into the queue.
        
        This runs in a separate thread to enable pipelining: the producer prepares
        the next batch while workers train on current batches. The producer also
        computes the learning rate for each batch based on training progress.
        
        Args:
            sentences: Iterable of tokenized sentences.
            job_queue: Queue to put (batch, alpha, seed) jobs into.
            start_alpha: Initial learning rate.
            min_alpha_val: Minimum learning rate (for decay).
            decay_alpha: Whether to decay learning rate.
            total_words_all_epochs: Total words across all epochs (for decay calculation).
            epoch: Current epoch number (0-indexed).
            epochs: Total number of epochs.
        """
        words_produced = 0
        base_words = epoch * self.corpus_word_count
        
        for batch in iter_batches(sentences, batch_words=self.batch_size, max_length=None):
            if decay_alpha and total_words_all_epochs > 0:
                global_words = base_words + words_produced
                progress = min(global_words / total_words_all_epochs, 1.0)
                batch_alpha = start_alpha + (min_alpha_val - start_alpha) * progress
            else:
                batch_alpha = start_alpha
            
            batch_seed = self._get_random_state()
            job_queue.put((batch, batch_alpha, batch_seed))
            words_produced += sum(len(s) for s in batch)
        
        for _ in range(self.workers):
            job_queue.put(None)
    
    def _train_epoch_threaded(
        self,
        sentences: Iterable[list[str]],
        sample_ints: np.ndarray,
        start_alpha: float,
        min_alpha_val: float,
        decay_alpha: bool,
        total_words_all_epochs: int,
        epoch: int,
        epochs: int,
        calculate_loss: bool,
        progress_bar,
        recent_losses: list,
    ) -> tuple[float, int, int]:
        """
        Train one epoch using producer-consumer threading model.
        
        This method spawns worker threads and a producer thread, then monitors
        progress from the main thread. Even with workers=1, this enables
        pipelining where batch preparation overlaps with training.
        
        Args:
            sentences: Iterable of tokenized sentences.
            sample_ints: Subsampling thresholds array.
            start_alpha: Initial learning rate.
            min_alpha_val: Minimum learning rate.
            decay_alpha: Whether to decay learning rate.
            total_words_all_epochs: Total words across all epochs.
            epoch: Current epoch number (0-indexed).
            epochs: Total number of epochs.
            calculate_loss: Whether to compute loss.
            progress_bar: tqdm progress bar or None.
            recent_losses: List for tracking recent loss values (modified in place).
        
        Returns:
            Tuple of (epoch_loss, epoch_examples, epoch_words).
        """
        queue_factor = 2
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)
        
        worker_threads = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue, sample_ints, calculate_loss),
            )
            for _ in range(self.workers)
        ]
        
        producer_thread = threading.Thread(
            target=self._job_producer,
            args=(
                sentences, job_queue, start_alpha, min_alpha_val,
                decay_alpha, total_words_all_epochs, epoch, epochs,
            ),
        )
        
        for t in worker_threads:
            t.daemon = True
            t.start()
        producer_thread.daemon = True
        producer_thread.start()
        
        epoch_loss = 0.0
        epoch_examples = 0
        epoch_words = 0
        unfinished_workers = self.workers
        
        while unfinished_workers > 0:
            result = progress_queue.get()
            if result is None:
                unfinished_workers -= 1
                continue
            
            batch_loss, batch_examples, batch_words = result
            epoch_loss += batch_loss
            epoch_examples += batch_examples
            epoch_words += batch_words
            
            if batch_examples > 0 and batch_loss > 0:
                batch_avg_loss = batch_loss / batch_examples
                recent_losses.append(batch_avg_loss)
            
            if progress_bar is not None:
                recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                if decay_alpha:
                    global_words = epoch * self.corpus_word_count + epoch_words
                    progress = min(global_words / total_words_all_epochs, 1.0) if total_words_all_epochs > 0 else 0
                    current_alpha = start_alpha + (min_alpha_val - start_alpha) * progress
                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}"
                else:
                    postfix_str = f"loss={recent_avg:.6f}"
                progress_bar.set_postfix_str(postfix_str)
                progress_bar.update(batch_words)
        
        producer_thread.join()
        for t in worker_threads:
            t.join()
        
        return epoch_loss, epoch_examples, epoch_words

    def train(
        self,
        sentences: Iterable[list[str]] | None = None,
        epochs: int | None = None,
        update_vocab: bool = False,
        reset_lr: bool = True,
    ) -> float | None:
        """
        Train word2vec model on sentences.
        
        Processes sentences in batches using Cython-accelerated training.
        This approach is memory-efficient and works with both lists and iterables.
        
        This method supports incremental training: call it multiple times with
        new data and ``update_vocab=True`` to expand the vocabulary and continue
        training.
        
        Args:
            sentences: Iterable of tokenized sentences. If None, uses sentences
                provided at initialization.
            epochs: Number of training epochs. If None, uses epochs from initialization.
            update_vocab: If True, expand vocabulary with new words from sentences.
                New words are initialized using the mean of existing embeddings
                (Hewitt 2021) to preserve the pretrained distribution. Only
                effective when the model already has a vocabulary.
            reset_lr: If True (default), reset learning rate to ``_initial_alpha``
                for this training run. If False, continue from current ``alpha``
                (useful for true continuation of a training run).
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        
        Raises:
            ValueError: If no sentences are available (neither passed nor at init).
        
        Example:
            # Initial training
            model = Word2Vec(sentences, epochs=5)
            model.train()
            
            # Continue training on same data
            model.train(epochs=3, reset_lr=False)
            
            # Incremental training with new data
            model.train(new_sentences, epochs=5, update_vocab=True)
        """
        # Resolve sentences source
        if sentences is None:
            sentences = self._sentences
        if sentences is None:
            raise ValueError(
                "No sentences provided. Pass sentences to train() or "
                "provide them at Word2Vec() initialization."
            )
        
        # Resolve epochs
        if epochs is None:
            epochs = self.epochs
        if epochs is None:
            raise ValueError("epochs must be specified either at init or in train()")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        
        # Handle learning rate reset
        if reset_lr:
            # Reset to initial alpha (stored at init)
            if hasattr(self, '_initial_alpha') and self._initial_alpha is not None:
                self.alpha = self._initial_alpha
            elif self.alpha is None:
                self.alpha = 0.025
        
        # Handle missing alpha
        if self.alpha is None:
            logger.warning("No initial learning rate (alpha) provided. Using default value of 0.025 with no decay.")
            self.alpha = 0.025
            self.min_alpha = None

        # Determine if we should decay the learning rate based on min_alpha
        decay_alpha = self.min_alpha is not None
        
        # Handle vocabulary: expand, build from scratch, or use existing
        if update_vocab and self.vocab:
            # Expand existing vocabulary with new words
            # Note: this consumes the iterator once for counting
            new_words_added = self._expand_vocab(sentences)
            if self.verbose and new_words_added > 0:
                logger.info(f"Added {new_words_added:,} new words to vocabulary")
        elif not self.vocab:
            # Build vocabulary from scratch
            self.build_vocab(sentences)
        
        # Initialize vectors if needed
        if self.W is None or self.W_prime is None:
            self._initialize_vectors()
        
        # Ensure W_prime exists for training (may be None if loaded from external format)
        if self.W_prime is None:
            self.W_prime = np.zeros((len(self.vocab), self.vector_size), dtype=self.dtype)
            if self.verbose:
                logger.info("Initialized W_prime for training (was None from imported vectors)")
        
        # Ensure work buffers exist (needed if continuing training on loaded model)
        if not hasattr(self, '_work') or self._work is None:
            self._work = np.zeros(self.vector_size, dtype=self.dtype)
            self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)
        
        # Build alias table for O(1) negative sampling
        # Must be rebuilt after vocab expansion
        self._build_alias_table()
        
        # Compute sample_ints for subsampling (needed for both training and example estimation)
        sample_ints = self._compute_sample_ints()
        
        # Read training configuration from instance attributes
        batch_size = self.batch_size
        callbacks = self.callbacks
        calculate_loss = self.calculate_loss
        total_examples = self.total_examples_hint
        
        # Setup for loss calculation
        total_loss = 0.0
        examples_processed_total = 0
        total_example_count = 0
        recent_losses = deque(maxlen=100)
        
        # Estimate total examples if needed for learning rate decay
        examples_per_epoch = None
        if decay_alpha:
            if total_examples is not None:
                examples_per_epoch = total_examples
                total_example_count = total_examples * epochs
            else:
                examples_per_epoch = self._estimate_example_count(sample_ints)
                total_example_count = examples_per_epoch * epochs
        
        # Track progress across all epochs for learning rate decay
        total_words_all_epochs = self.corpus_word_count * epochs
        global_words_processed = 0
        
        start_alpha = self.alpha
        min_alpha_val = self.min_alpha if self.min_alpha else start_alpha
        
        if self.verbose:
            logger.info(f"Starting training: {epochs} epoch(s), batch_size={batch_size:,}, alpha={self.alpha}")
        
        # Determine shuffle behavior: default to True for lists, False for iterables
        is_list = isinstance(sentences, list)
        shuffle = self.shuffle if self.shuffle is not None else is_list
        
        # Validate shuffle parameter with non-list iterables
        if shuffle and not is_list:
            raise ValueError(
                "shuffle=True requires sentences to be a list. "
                "For file-based or streaming iterables, pre-shuffle your data "
                "or set shuffle=False."
            )
        
        if self.verbose:
            logger.info(f"Training with {self.workers} worker(s) using producer-consumer threading")
        
        # Training loop for each epoch
        for epoch in range(epochs):
            # Shuffle sentences at the start of each epoch if enabled
            if shuffle:
                self._rng.shuffle(sentences)
            
            epoch_start_time = time.time()
            
            # Progress bar for streaming (we know total words from vocab building)
            if calculate_loss:
                bar_format = '{l_bar}{bar}| {percentage:.1f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                progress_bar = tqdm(
                    desc=f"Epoch {epoch+1}/{epochs}",
                    total=self.corpus_word_count,
                    bar_format=bar_format,
                    unit=" tokens",
                    unit_scale=True,
                    mininterval=0.5
                )
            else:
                progress_bar = None
            
            # Train epoch using threaded producer-consumer model
            epoch_loss, epoch_examples, epoch_words = self._train_epoch_threaded(
                sentences=sentences,
                sample_ints=sample_ints,
                start_alpha=start_alpha,
                min_alpha_val=min_alpha_val,
                decay_alpha=decay_alpha,
                total_words_all_epochs=total_words_all_epochs,
                epoch=epoch,
                epochs=epochs,
                calculate_loss=calculate_loss,
                progress_bar=progress_bar,
                recent_losses=recent_losses,
            )
            
            # Close progress bar
            if progress_bar is not None:
                if progress_bar.total is not None:
                    remaining = max(0, progress_bar.total - progress_bar.n)
                    if remaining > 0:
                        progress_bar.update(remaining)
                progress_bar.close()
            
            # Add epoch loss to total
            if calculate_loss:
                total_loss += epoch_loss
                examples_processed_total += epoch_examples
            
            # Update instance alpha to reflect current learning rate (for callbacks)
            if decay_alpha:
                # Calculate what alpha should be at end of this epoch
                epoch_progress = (epoch + 1) / epochs
                self.alpha = start_alpha + (min_alpha_val - start_alpha) * epoch_progress
            
            # Call any registered callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)
            
            if self.verbose:
                elapsed = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch+1}/{epochs} completed: {epoch_examples:,} examples, "
                          f"{epoch_words:,} words in {elapsed:.1f}s")
        
        # Update the instance alpha to reflect the final learning rate
        if decay_alpha:
            self.alpha = self.min_alpha
        
        # Calculate and return the final average loss if requested
        if calculate_loss and examples_processed_total > 0:
            final_avg_loss = total_loss / examples_processed_total
            if self.verbose:
                logger.info(f"Training completed. Final average loss: {final_avg_loss:.6f}")
            return final_avg_loss
        
        return None

    def get_vector(self, word: str, normalize: bool = False) -> np.ndarray:
        """
        Get the vector for a word.
        
        Args:
            word: Input word.
            normalize: If True, return the normalized vector (unit length).
        
        Returns:
            Word vector as numpy array of shape (vector_size,).
        
        Raises:
            KeyError: If word is not in vocabulary.
        """
        if word in self.vocab:
            vector = self.W[self.vocab[word]]
            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    return vector / norm
            return vector
        else:
            raise KeyError(f"Word '{word}' not in vocabulary")
    
    def __getitem__(self, word: str) -> np.ndarray:
        """
        Dictionary-like access to word vectors.
        
        Args:
            word: Input word.
        
        Returns:
            Word vector or raises KeyError if word is not in vocabulary.
        """
        
        return self.get_vector(word)
    
    def __contains__(self, word: str) -> bool:
        """
        Check if a word is in the vocabulary using the 'in' operator.
        
        Args:
            word: Word to check.
        
        Returns:
            True if the word is in the vocabulary, False otherwise.
        """
        return word in self.vocab
    
    def most_similar(
        self, 
        word: str, 
        topn: int = 10,
        cross_space: bool = False
    ) -> list[tuple[str, float]]:
        """
        Find the topn most similar words to the given word.
        
        Args:
            word: Input word.
            topn: Number of similar words to return.
            cross_space: If False (default), compare W vs W (second-order similarity).
                If True, compare W vs W_prime (first-order similarity based on
                direct co-occurrence patterns).
        
        Returns:
            List of (word, similarity) tuples sorted by descending similarity.
        
        Raises:
            KeyError: If word is not in vocabulary.
        """
        if word not in self.vocab:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        word_idx = self.vocab[word]
        word_vec = self.W[word_idx].reshape(1, -1)
        
        if cross_space:
            sim = cosine_similarity(word_vec, self.W_prime).flatten()
        else:
            sim = cosine_similarity(word_vec, self.W).flatten()
        
        k = min(topn + 1, len(sim))
        top_candidates = np.argpartition(-sim, k)[:k]
        top_sorted = top_candidates[np.argsort(-sim[top_candidates])]
        return [(self.index2word[i], float(sim[i])) for i in top_sorted if i != word_idx][:topn]

    def similarity(
        self, 
        word1: str, 
        word2: str,
        cross_space: bool = False
    ) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word (always from W).
            word2: Second word (from W or W_prime depending on cross_space).
            cross_space: If False (default), compare W[word1] vs W[word2].
                If True, compare W[word1] vs W_prime[word2].
        
        Returns:
            Cosine similarity between the two words (float between -1 and 1).
        
        Raises:
            KeyError: If either word is not in the vocabulary.
        """
        if word1 not in self.vocab:
            raise KeyError(f"Word '{word1}' not found in vocabulary")
        if word2 not in self.vocab:
            raise KeyError(f"Word '{word2}' not found in vocabulary")
        
        word1_vec = self.W[self.vocab[word1]]
        
        if cross_space:
            word2_vec = self.W_prime[self.vocab[word2]]
        else:
            word2_vec = self.W[self.vocab[word2]]
        
        return float(cosine_similarity(word1_vec, word2_vec))

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Saves all model parameters, vocabulary, and trained vectors. Training-specific
        parameters (alpha, min_alpha, epochs, etc.) are not saved as they are only
        needed during training, not inference.
        
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
            'W': self.W,
            'W_prime': self.W_prime
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def export(self, path: str, format: str = "word2vec", binary: bool = True) -> None:
        """
        Export word vectors to external format for interoperability.
        
        Exports only the input embeddings (W matrix). Output embeddings (W_prime)
        and word counts are not exported as external formats don't support them.
        
        Args:
            path: Output file path.
            format: Export format, one of:
                - "word2vec": Google word2vec C format (default). Compatible with
                  gensim's ``KeyedVectors.load_word2vec_format()``.
                - "glove": GloVe text format. No header, space-separated values.
                - "gensim": Gensim's native KeyedVectors format. Requires gensim.
            binary: For word2vec format only. If True (default), write vectors as
                binary floats. If False, write as text. Ignored for other formats.
        
        Example:
            # Export to word2vec binary format
            model.export("vectors.bin", format="word2vec", binary=True)
            
            # Export to text format for inspection
            model.export("vectors.txt", format="word2vec", binary=False)
            
            # Export to GloVe format
            model.export("vectors.glove.txt", format="glove")
            
            # Load in gensim
            from gensim.models import KeyedVectors
            kv = KeyedVectors.load_word2vec_format("vectors.bin", binary=True)
        
        Raises:
            ValueError: If format is not recognized.
            ImportError: If format="gensim" and gensim is not installed.
        """
        if self.W is None:
            raise ValueError("No vectors to export. Train the model first.")
        
        format = format.lower()
        
        if format == "word2vec":
            self._export_word2vec_format(path, binary=binary)
        elif format == "glove":
            self._export_glove_format(path)
        elif format == "gensim":
            self._export_gensim_format(path)
        else:
            raise ValueError(
                f"Unknown format '{format}'. "
                "Supported formats: 'word2vec', 'glove', 'gensim'."
            )
    
    def _export_word2vec_format(self, path: str, binary: bool = True) -> None:
        """Export to Google word2vec C format."""
        vocab_size = len(self.vocab)
        with open(path, 'wb') as f:
            header = f"{vocab_size} {self.vector_size}\n"
            f.write(header.encode('utf-8'))
            
            for word in self.index2word:
                vec = self.W[self.vocab[word]]
                if binary:
                    f.write(word.encode('utf-8') + b' ')
                    f.write(vec.astype(np.float32).tobytes())
                    f.write(b'\n')
                else:
                    vec_str = ' '.join(f'{x:.6f}' for x in vec)
                    line = f"{word} {vec_str}\n"
                    f.write(line.encode('utf-8'))
    
    def _export_glove_format(self, path: str) -> None:
        """Export to GloVe text format (no header)."""
        with open(path, 'w', encoding='utf-8') as f:
            for word in self.index2word:
                vec = self.W[self.vocab[word]]
                vec_str = ' '.join(f'{x:.6f}' for x in vec)
                f.write(f"{word} {vec_str}\n")
    
    def _export_gensim_format(self, path: str) -> None:
        """Export to gensim's native KeyedVectors format."""
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required for format='gensim'. "
                "Install it with: pip install gensim"
            )
        
        kv = KeyedVectors(vector_size=self.vector_size)
        kv.add_vectors(self.index2word, self.W)
        kv.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'Word2Vec':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from.
        
        Returns:
            Loaded Word2Vec model.
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Get values with defaults if not found
        shrink_windows = model_data.get('shrink_windows', False)
        sample = model_data.get('sample', 1e-3)
        max_vocab_size = model_data.get('max_vocab_size', None)
        
        # Create model with _skip_init to avoid unnecessary initialization
        model = cls(
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
        
        # Restore saved state
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
        # Restore corpus statistics (for potential further training or analysis)
        model.corpus_word_count = model_data.get('corpus_word_count', sum(model.word_counts.values()))
        model.total_corpus_tokens = model_data.get('total_corpus_tokens', model.corpus_word_count)
        
        return model
    
    @classmethod
    def load_vectors(
        cls,
        path: str,
        format: str = "word2vec",
        binary: bool = True,
    ) -> 'Word2Vec':
        """
        Load word vectors from external format.
        
        Creates a Word2Vec model from externally-trained vectors (e.g., from gensim,
        original word2vec, or GloVe). The loaded model supports inference operations
        (similarity queries, vector access) but lacks output embeddings (W_prime)
        and word counts needed for training.
        
        To continue training on a loaded model, call ``train()`` with a corpus.
        Use ``update_vocab=True`` if you want to add new words; otherwise the
        existing vocabulary is preserved. Missing structures (W_prime, word_counts)
        are initialized automatically.
        
        Args:
            path: Path to the vectors file.
            format: Input format, one of:
                - "word2vec": Google word2vec C format (default). Compatible with
                  gensim's ``save_word2vec_format()``.
                - "glove": GloVe text format. No header, space-separated values.
                - "gensim": Gensim's native KeyedVectors format.
            binary: For word2vec format only. If True (default), expect binary floats.
                If False, expect text format. Ignored for other formats.
        
        Returns:
            Word2Vec model with loaded vectors. The model has:
            - W: Input embeddings loaded from file
            - W_prime: None (not available in external formats)
            - word_counts: Empty (not available in external formats)
        
        Example:
            # Load word2vec binary format
            model = Word2Vec.load_vectors("vectors.bin", format="word2vec", binary=True)
            
            # Load GloVe format
            model = Word2Vec.load_vectors("glove.txt", format="glove")
            
            # Use for similarity queries
            similar = model.most_similar("king", topn=10)
            
            # Enable training by providing a corpus
            model.train(new_sentences, epochs=5, update_vocab=True)
        
        Raises:
            ValueError: If format is not recognized or file is malformed.
            ImportError: If format="gensim" and gensim is not installed.
        """
        format = format.lower()
        
        if format == "word2vec":
            return cls._load_word2vec_format(path, binary=binary)
        elif format == "glove":
            return cls._load_glove_format(path)
        elif format == "gensim":
            return cls._load_gensim_format(path)
        else:
            raise ValueError(
                f"Unknown format '{format}'. "
                "Supported formats: 'word2vec', 'glove', 'gensim'."
            )
    
    @classmethod
    def _load_word2vec_format(cls, path: str, binary: bool = True) -> 'Word2Vec':
        """Load from Google word2vec C format."""
        vocab = {}
        index2word = []
        vectors = []
        
        with open(path, 'rb') as f:
            header = f.readline().decode('utf-8').strip()
            parts = header.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid word2vec header: expected '<vocab_size> <vector_size>', "
                    f"got '{header}'"
                )
            vocab_size, vector_size = int(parts[0]), int(parts[1])
            
            for _ in range(vocab_size):
                word_bytes = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise ValueError("Unexpected end of file while reading word")
                    word_bytes.append(ch)
                word = b''.join(word_bytes).decode('utf-8')
                
                if binary:
                    vec_bytes = f.read(vector_size * 4)
                    if len(vec_bytes) != vector_size * 4:
                        raise ValueError(f"Unexpected end of file while reading vector for '{word}'")
                    vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                    # Skip newline if present
                    ch = f.read(1)
                    if ch and ch != b'\n':
                        f.seek(-1, 1)
                else:
                    line_rest = f.readline().decode('utf-8').strip()
                    vec_parts = line_rest.split()
                    if len(vec_parts) != vector_size:
                        raise ValueError(
                            f"Vector dimension mismatch for '{word}': "
                            f"expected {vector_size}, got {len(vec_parts)}"
                        )
                    vec = np.array([float(x) for x in vec_parts], dtype=np.float32)
                
                vocab[word] = len(index2word)
                index2word.append(word)
                vectors.append(vec)
        
        W = np.vstack(vectors).astype(np.float32)
        return cls._create_from_vectors(W, vocab, index2word, vector_size)
    
    @classmethod
    def _load_glove_format(cls, path: str) -> 'Word2Vec':
        """Load from GloVe text format (no header)."""
        vocab = {}
        index2word = []
        vectors = []
        vector_size = None
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                vec_parts = parts[1:]
                
                if vector_size is None:
                    vector_size = len(vec_parts)
                elif len(vec_parts) != vector_size:
                    raise ValueError(
                        f"Inconsistent vector dimension at line {line_num}: "
                        f"expected {vector_size}, got {len(vec_parts)}"
                    )
                
                vec = np.array([float(x) for x in vec_parts], dtype=np.float32)
                vocab[word] = len(index2word)
                index2word.append(word)
                vectors.append(vec)
        
        if not vectors:
            raise ValueError("No vectors found in file")
        
        W = np.vstack(vectors).astype(np.float32)
        return cls._create_from_vectors(W, vocab, index2word, vector_size)
    
    @classmethod
    def _load_gensim_format(cls, path: str) -> 'Word2Vec':
        """Load from gensim's native KeyedVectors format."""
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required for format='gensim'. "
                "Install it with: pip install gensim"
            )
        
        kv = KeyedVectors.load(path)
        
        vocab = {word: idx for idx, word in enumerate(kv.index_to_key)}
        index2word = list(kv.index_to_key)
        W = kv.vectors.astype(np.float32)
        vector_size = kv.vector_size
        
        return cls._create_from_vectors(W, vocab, index2word, vector_size)
    
    @classmethod
    def _create_from_vectors(
        cls,
        W: np.ndarray,
        vocab: dict,
        index2word: list,
        vector_size: int,
    ) -> 'Word2Vec':
        """Create a Word2Vec model from loaded vectors."""
        model = cls(
            vector_size=vector_size,
            _skip_init=True,
        )
        
        model.vocab = vocab
        model.index2word = index2word
        model.W = W
        model.W_prime = None  # Not available from external formats
        model.word_counts = Counter()  # Not available from external formats
        model.corpus_word_count = 0
        model.total_corpus_tokens = 0
        
        return model


__all__ = [
    'Word2Vec',
]