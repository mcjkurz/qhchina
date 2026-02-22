"""
Word2Vec implementation for learning word embeddings.

Provides CBOW and Skip-gram architectures for training word vectors from text corpora.
Includes support for temporal semantic change analysis via TempRefWord2Vec.
"""

import logging
import numpy as np
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from queue import Queue
from tqdm.auto import tqdm
import time
from .vectors import cosine_similarity
from .word2vec_utils import LineSentenceFile, word2vec_c
from ..config import get_rng, resolve_seed

logger = logging.getLogger("qhchina.analytics.word2vec")


class Word2Vec:
    """
    Word2Vec model for learning word embeddings from text.
    
    Supports two training algorithms:
    - Skip-gram (sg=1): Predicts context words from center word. Generally better for 
      infrequent words and smaller datasets.
    - CBOW (sg=0): Predicts center word from context words. Faster to train.
    
    Training does NOT start automatically. Call ``train()`` explicitly after initialization
    to begin training.
    
    Args:
        sentences: Tokenized sentences for training. Can be:
            - An iterable of sentences (each sentence is a list of tokens)
            - A file path (str) to a corpus file (created via ``Corpus.save("file.txt")``)
            Note: Iterables must be restartable (can be iterated multiple times).
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
        epochs (int): Number of training iterations over the corpus (default: 1).
        batch_size (int): Number of words per training batch (default: 10240).
        workers (int): Number of worker threads for parallel training (default: 1).
        callbacks (list of callable, optional): Callback functions to call after each epoch.
        calculate_loss (bool): Whether to calculate and return the final loss (default: True).
        total_examples (int, optional): Total number of training examples per epoch. When provided 
            along with ``min_alpha``, uses this exact value for learning rate decay calculation.
        shuffle (bool, optional): Whether to shuffle sentences before each epoch. 
            Defaults to True if sentences is a list, False otherwise.
    
    Example:
        from qhchina.analytics.word2vec import Word2Vec
        
        # Prepare corpus as list of tokenized sentences
        sentences = [['我', '喜欢', '学习'], ['他', '喜欢', '运动']]
        
        # Initialize the model with sentences
        model = Word2Vec(sentences, vector_size=100, window=5, min_word_count=1, epochs=5)
        
        # Explicitly start training
        model.train()
        
        # Or train from a corpus file (memory-efficient for large corpora)
        corpus = Corpus(sentences)
        corpus.save("corpus.txt")
        model = Word2Vec("corpus.txt", vector_size=100, epochs=5)
        model.train()
        
        # Get word vector
        vector = model['喜欢']
        
        # Find similar words
        similar = model.most_similar('喜欢', topn=5)
    """
    
    def __init__(
        self,
        sentences: Iterable[list[str]] | str | None = None,
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
        epochs: int = 1,
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
            self._sentences = None
            # Initialize empty structures - will be populated by load()
            self.vocab = {}
            self.index2word = []
            self.word_counts = Counter()
            self.corpus_word_count = 0
            self.total_corpus_tokens = 0
            self.W = None
            self.W_prime = None
            self.noise_distribution = None
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
        self.min_alpha = min_alpha
        self.sample = sample  # Threshold for subsampling
        self.shrink_windows = shrink_windows  # Dynamic window size
        self.max_vocab_size = max_vocab_size  # Maximum vocabulary size
        
        # Set dtype for weight matrices (float32 for Cython BLAS compatibility)
        self.dtype = np.float32
        
        # Use isolated RNG to avoid affecting global state
        effective_seed = resolve_seed(seed)
        self._rng = get_rng(effective_seed)
        
        # Store sentences for training (normalize file path to LineSentenceFile)
        if isinstance(sentences, str):
            self._sentences = LineSentenceFile(sentences)
        else:
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
        self.noise_distribution = None  # For negative sampling
        
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
        
        # Use default_rng (PCG64) for reproducible initialization
        init_rng = np.random.default_rng(seed=self.seed)
        self.W = init_rng.random((vocab_size, self.vector_size), dtype=self.dtype)
        self.W *= 2.0
        self.W -= 1.0
        self.W /= self.vector_size
        
        # Output matrix starts at zero for stable negative sampling convergence
        self.W_prime = np.zeros((vocab_size, self.vector_size), dtype=self.dtype)
        
        # Work buffers reused across batches (avoids per-batch allocation)
        self._work = np.zeros(self.vector_size, dtype=self.dtype)
        self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)

    def _prepare_noise_distribution(self) -> None:
        """
        Prepare noise distribution for negative sampling.
        
        More frequent words have higher probability of being selected.
        Applies ns_exponent smoothing (typically 0.75) to prevent extremely
        common words from dominating the negative samples.
        """
        if len(self.vocab) == 0:
            self.noise_distribution = np.array([], dtype=self.dtype)
            return
        
        word_counts = np.array([self.word_counts[word] for word in self.vocab])
        
        # Apply the exponent to smooth the distribution
        noise_dist = word_counts ** self.ns_exponent
        
        # Normalize to get a probability distribution
        total = np.sum(noise_dist)
        if total == 0:
            # Fallback to uniform distribution if all counts are zero
            noise_dist_normalized = np.ones(len(self.vocab), dtype=self.dtype) / len(self.vocab)
        else:
            noise_dist_normalized = noise_dist / total
        
        # Explicitly cast to the correct dtype (float32 or float64)
        self.noise_distribution = noise_dist_normalized.astype(self.dtype)

    def _build_cum_table(self) -> np.ndarray:
        """
        Build cumulative distribution table for negative sampling.
        
        The cumulative table maps uniform random values to word indices,
        with more frequent words having larger ranges.
        
        Returns:
            np.ndarray[uint32]: Cumulative distribution table
            
        The model keeps a reference to this array to prevent garbage collection.
        
        Note: This matches Gensim's make_cum_table() exactly to ensure identical
        negative sampling behavior.
        """
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            self._cum_table = np.array([], dtype=np.uint32)
            return self._cum_table
        
        # Match Gensim's make_cum_table exactly:
        # 1. Compute sum of all count^ns_exponent (Z in paper)
        # 2. Build cumulative values with round() for each entry
        domain = 2**31 - 1
        self._cum_table = np.zeros(vocab_size, dtype=np.uint32)
        
        # Compute Z = sum of all count^exponent
        train_words_pow = 0.0
        for word in self.index2word:
            count = self.word_counts[word]
            train_words_pow += count ** self.ns_exponent
        
        # Build cumulative table entry by entry using round()
        # Using round() is critical - truncation via astype(uint32) causes different
        # negative samples to be drawn, leading to divergent training results
        cumulative = 0.0
        for word_index, word in enumerate(self.index2word):
            count = self.word_counts[word]
            cumulative += count ** self.ns_exponent
            self._cum_table[word_index] = round(cumulative / train_words_pow * domain)
        
        # Verify final value equals domain (Gensim does this assertion)
        if vocab_size > 0:
            assert self._cum_table[-1] == domain, f"Final cum_table value {self._cum_table[-1]} != {domain}"
        
        return self._cum_table
    
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
        
        for word, idx in self.vocab.items():
            word_freq = self.word_counts[word] / total_words
            # Google/Gensim formula: keep_prob = sqrt(t/f) + t/f
            keep_prob = np.sqrt(self.sample / word_freq) + (self.sample / word_freq)
            keep_prob = min(keep_prob, 1.0)
            
            # Convert to uint32 threshold: word is kept if threshold >= random
            # So threshold = keep_prob * (2^32 - 1)
            sample_ints[idx] = min(np.uint32(keep_prob * 4294967295.0), np.uint32(4294967295))
        
        return sample_ints
    
    def _iter_batches(self, sentences: Iterable[list[str]], batch_words: int = 10240):
        """
        Yield batches of sentences, batched by total word count (like Gensim).
        
        This allows streaming through large corpora without loading everything
        into memory at once. Batching by word count (not sentence count) ensures
        consistent batch sizes for training regardless of sentence length.
        
        Args:
            sentences: Iterable of tokenized sentences.
            batch_words: Maximum number of words per batch (default 10240).
        
        Yields:
            List of sentences where total word count <= batch_words.
        """
        batch = []
        word_count = 0
        
        for sent in sentences:
            sent_len = len(sent)
            
            # Can we fit this sentence in the current batch?
            if word_count + sent_len <= batch_words:
                batch.append(sent)
                word_count += sent_len
            else:
                # Yield current batch if non-empty
                if batch:
                    yield batch
                # Start new batch with this sentence
                batch = [sent]
                word_count = sent_len
        
        # Yield any remaining sentences
        if batch:
            yield batch
    
    def _get_thread_working_mem(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Allocate private work buffers for a worker thread.
        
        Each worker thread needs its own work buffers to avoid race conditions.
        These buffers are used for gradient accumulation during training.
        
        Returns:
            Tuple of (work, neu1) numpy arrays, each of shape (vector_size,).
        """
        work = np.zeros(self.vector_size, dtype=self.dtype)
        neu1 = np.zeros(self.vector_size, dtype=self.dtype)
        return work, neu1
    
    def _train_batch_worker(
        self,
        batch: list[list[str]],
        sample_ints: np.ndarray,
        alpha: float,
        random_seed: int,
        calculate_loss: bool,
        work: np.ndarray,
        neu1: np.ndarray,
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
        
        Returns:
            Tuple of (batch_loss, batch_examples, batch_words).
        """
        batch_loss, batch_examples, batch_words, _ = word2vec_c.train_batch(
            self.W,
            self.W_prime,
            batch,
            self.vocab,
            sample_ints,
            self._cum_table,
            work,
            neu1,
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
        work, neu1 = self._get_thread_working_mem()
        
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break
            
            batch, alpha, seed = job
            
            batch_loss, batch_examples, batch_words = self._train_batch_worker(
                batch, sample_ints, alpha, seed, calculate_loss, work, neu1
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
        
        for batch in self._iter_batches(sentences, batch_words=self.batch_size):
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
        LOSS_HISTORY_SIZE = 100
        
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
                if len(recent_losses) > LOSS_HISTORY_SIZE:
                    recent_losses.pop(0)
            
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

    def train(self) -> float | None:
        """
        Train word2vec model on sentences provided at initialization.
        
        Processes sentences in batches using Cython-accelerated training.
        This approach is memory-efficient and works with both lists and iterables.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        
        Raises:
            ValueError: If no sentences were provided at initialization.
        """
        sentences = self._sentences
        if sentences is None:
            raise ValueError(
                "No sentences provided. Pass sentences to Word2Vec() at initialization."
            )
        
        # Handle missing alpha
        if self.alpha is None:
            logger.warning("No initial learning rate (alpha) provided. Using default value of 0.025 with no decay.")
            self.alpha = 0.025
            self.min_alpha = None

        # Determine if we should decay the learning rate based on min_alpha
        decay_alpha = self.min_alpha is not None
        
        # Build vocab if not already done
        # Note: For non-list iterables, this will consume the iterator once
        # The iterator must be restartable for training
        if not self.vocab: 
            self.build_vocab(sentences)
        if self.W is None or self.W_prime is None:
            self._initialize_vectors()
        # Ensure work buffers exist (needed if continuing training on loaded model)
        if not hasattr(self, '_work') or self._work is None:
            self._work = np.zeros(self.vector_size, dtype=self.dtype)
            self._neu1 = np.zeros(self.vector_size, dtype=self.dtype)
        if self.noise_distribution is None:
            self._prepare_noise_distribution()
        
        # Build cumulative table for negative sampling (thread-safe)
        self._build_cum_table()
        
        # Compute sample_ints for subsampling (needed for both training and example estimation)
        sample_ints = self._compute_sample_ints()
        
        # Read training configuration from instance attributes
        epochs = self.epochs
        batch_size = self.batch_size
        callbacks = self.callbacks
        calculate_loss = self.calculate_loss
        total_examples = self.total_examples_hint
        
        # Setup for loss calculation
        total_loss = 0.0
        examples_processed_total = 0
        total_example_count = 0
        recent_losses = []
        
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
        
        most_similar_words = []
        for idx in (-sim).argsort():
            if idx != word_idx and len(most_similar_words) < topn:
                most_similar_words.append((self.index2word[idx], float(sim[idx])))
        
        return most_similar_words

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
        np.save(path, model_data, allow_pickle=True)
    
    @classmethod
    def load(cls, path: str) -> 'Word2Vec':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from.
        
        Returns:
            Loaded Word2Vec model.
        """
        model_data = np.load(path, allow_pickle=True).item()
        
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


__all__ = [
    'Word2Vec',
    'LineSentenceFile',
]