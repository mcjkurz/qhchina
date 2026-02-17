"""
Sample-based Word2Vec implementation
-----------------------------------
This module implements the Word2Vec algorithm with both CBOW and Skip-gram models,
accelerated with Cython for training.

- Skip-gram (sg=1): Each training example is a tuple (input_idx, output_idx), where 
  input_idx is the index of the center word and output_idx is the index of a context word.
  Negative examples are generated from the noise distribution for each positive example.

- CBOW (sg=0): Each training example is a tuple (input_indices, output_idx), where
  input_indices are the indices of context words, and output_idx is the index of the center word.
  Negative examples are generated from the noise distribution for each positive example.

Features:
- CBOW and Skip-gram architectures
- Cython-accelerated training with BLAS operations
- Negative sampling for each training example
- Subsampling of frequent words
- Dynamic window sizing with shrink_windows parameter
- Properly managed learning rate decay
- Vocabulary size restriction with max_vocab_size parameter
- Streaming support for large corpora that don't fit in memory
"""

import logging
import numpy as np
from collections import Counter
from collections.abc import Callable, Iterable
from tqdm.auto import tqdm
import time
from .vectors import cosine_similarity
from ..config import get_rng, resolve_seed
logger = logging.getLogger("qhchina.analytics.word2vec")

try:
    from .cython_ext import word2vec as word2vec_c
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    word2vec_c = None
    logger.warning("Cython extensions not available; Word2Vec requires compiled extensions.")

__all__ = [
    'Word2Vec',
    'TempRefWord2Vec',
]


class Word2Vec:
    """
    Implementation of Word2Vec algorithm with Cython-accelerated training. It is inspired by the Gensim implementation.
    
    This class implements both Skip-gram and CBOW architectures:
    - Skip-gram (sg=1): Each training example is (input_idx, output_idx) where input is the center word
      and output is a context word.
    - CBOW (sg=0): Each training example is (input_indices, output_idx) where inputs are context words
      and output is the center word.
    
    Training is performed using optimized Cython routines with BLAS operations.
    
    If ``sentences`` is provided at initialization, training starts immediately. Otherwise, call
    ``train()`` later with the sentences to train on.
    
    Features:
    - CBOW and Skip-gram architectures
    - Cython-accelerated training with BLAS operations
    - Negative sampling for each training example
    - Subsampling of frequent words
    - Dynamic window sizing with shrink_windows parameter
    - Properly managed learning rate decay
    - Vocabulary size restriction with max_vocab_size parameter
    
    Args:
        sentences (iterable of list of str, optional): Tokenized sentences for training. 
            If provided, training starts immediately during initialization.
            Note: The iterable must be restartable (can be iterated multiple times).
        vector_size (int): Dimensionality of the word vectors (default: 100).
        window (int): Maximum distance between the current and predicted word (default: 5).
        min_word_count (int): Ignores all words with frequency lower than this (default: 5).
        negative (int): Number of negative samples for negative sampling (default: 5).
        ns_exponent (float): Exponent used to shape the negative sampling distribution (default: 0.75).
        cbow_mean (bool): If True, use mean of context word vectors, else use sum (default: True).
        sg (int): Training algorithm: 1 for skip-gram; 0 for CBOW (default: 0).
        seed (int, optional): Seed for random number generator. If None, uses global seed setting.
        alpha (float): Initial learning rate (default: 0.025).
        min_alpha (float, optional): Minimum learning rate. If None, learning rate remains constant at alpha.
        sample (float): Threshold for subsampling frequent words. Default is 1e-3, set to 0 to disable.
        shrink_windows (bool): If True, the effective window size is uniformly sampled from [1, window] 
            for each target word during training. If False, always use the full window (default: True).
        max_vocab_size (int, optional): Maximum vocabulary size to keep, keeping the most frequent words.
            None means no limit (keep all words above min_word_count).
        verbose (bool): If True, log detailed progress information during training (default: False).
        epochs (int): Number of training iterations over the corpus (default: 1).
        batch_size (int): Target number of words per training batch (default: 10240). 
            Note that the Cython training buffer is limited to 10240 words, regardless of the batch_size parameter; 
            words beyond this limit will be dropped from the batch, which mimics Gensim behavior.
        callbacks (list of callable, optional): Callback functions to call after each epoch.
        calculate_loss (bool): Whether to calculate and return the final loss (default: True).
        total_examples (int, optional): Total number of training examples per epoch. When provided 
            along with ``min_alpha``, uses this exact value instead of estimating for learning rate decay.
    
    Example:
        from qhchina.analytics.word2vec import Word2Vec
        
        # Prepare corpus as list of tokenized sentences
        sentences = [['我', '喜欢', '学习'], ['他', '喜欢', '运动']]
        
        # Option 1: Train immediately by providing sentences at init
        model = Word2Vec(sentences, vector_size=100, window=5, min_word_count=1, epochs=5)
        
        # Option 2: Initialize first, train later
        model = Word2Vec(vector_size=100, window=5, min_word_count=1, epochs=5)
        model.train(sentences)
        
        # Get word vector (can use model directly or model.wv for gensim compatibility)
        vector = model['喜欢']
        vector = model.wv['喜欢']  # Same as above
        
        # Find similar words
        similar = model.most_similar('喜欢', topn=5)
        similar = model.wv.most_similar('喜欢', topn=5)  # Same as above
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
        epochs: int = 1,
        batch_size: int = 10240,
        callbacks: list[Callable] | None = None,
        calculate_loss: bool = True,
        total_examples: int | None = None,
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
            self.callbacks = callbacks
            self.calculate_loss = calculate_loss
            self.total_examples_hint = total_examples
            # Initialize empty structures - will be populated by load()
            self.vocab = {}
            self.index2word = []
            self.word_counts = Counter()
            self.corpus_word_count = 0
            self.total_corpus_tokens = 0
            self.W = None
            self.W_prime = None
            self.noise_distribution = None
            self._rng = get_rng(resolve_seed(seed))
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
        self.callbacks = callbacks
        self.calculate_loss = calculate_loss
        self.total_examples_hint = total_examples
        
        # Auto-train if sentences are provided
        if sentences is not None:
            self.train(sentences)

    @property
    def wv(self) -> 'Word2Vec':
        """
        Property for gensim compatibility. Returns self since the model directly
        supports vector access via __getitem__, most_similar(), etc.
        
        This allows users familiar with gensim to use model.wv['word'] syntax.
        Note: This returns self, not a copy - no additional memory is used.
        
        Returns:
            The model itself (self).
        """
        return self

    def build_vocab(self, sentences: Iterable[list[str]]) -> None:
        """
        Build vocabulary from sentences.
        
        Args:
            sentences: Iterable of tokenized sentences (each sentence is a list of words).
        
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
        
        # Check if any words were found (catches all-empty sentences)
        if not self.word_counts:
            raise ValueError("sentences contains no words. Provide non-empty tokenized sentences.")
        
        # Filter words by min_word_count and create vocabulary
        retained_words = {word for word, count in self.word_counts.items() if count >= self.min_word_count}
        
        # If max_vocab_size is set, keep only the most frequent words
        if self.max_vocab_size is not None and len(retained_words) > self.max_vocab_size:
            # Sort words by frequency (highest first) and take the top max_vocab_size
            top_words = [word for word, _ in self.word_counts.most_common(self.max_vocab_size)]
            # Intersect with words that meet min_word_count criteria
            retained_words = {word for word in top_words if word in retained_words}
            
        # Create mappings
        self.index2word = []
        for word, _ in self.word_counts.most_common():
            if word in retained_words:
                word_id = len(self.index2word)
                self.vocab[word] = word_id # word2index
                self.index2word.append(word)
        
        # Compute token counts for example estimation
        self.total_corpus_tokens = sum(self.word_counts.values())  # All tokens (including OOV)
        self.corpus_word_count = sum(self.word_counts[word] for word in self.vocab)  # Vocab tokens only
        
        if self.verbose:
            logger.info(f"Vocabulary built: {len(self.vocab):,} words, {self.corpus_word_count:,} tokens in vocab, {self.total_corpus_tokens:,} total tokens")

    def _estimate_example_count(self, sample_ints: np.ndarray) -> int:
        """
        Estimate total training examples per epoch without iterating through data.
        
        Used for learning rate decay scheduling when min_alpha is provided.
        The estimate accounts for subsampling, vocabulary coverage, and window effects.
        
        For Skip-gram: each retained word generates ~2*avg_window context pairs,
            but context words must also be in vocabulary and survive subsampling.
        For CBOW: each retained word generates 1 example (with context as input).
        
        Args:
            sample_ints: Pre-computed subsampling thresholds (uint32 array from _compute_sample_ints).
        
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
        """
        Initialize input (W) and output (W_prime) word embedding matrices.
        
        Uses Xavier/Glorot initialization with uniform distribution in range 
        [-0.5/vector_size, 0.5/vector_size] for better convergence during training.
        Both matrices are initialized with random values (not zeros).
        """
        vocab_size = len(self.vocab)
        if self.verbose:
            logger.info(f"Initializing vectors: 2 matrices of shape ({vocab_size:,}, {self.vector_size})...")
        
        # Initialize input and output matrices with Xavier/Glorot initialization
        bound = 0.5 / self.vector_size
        self.W = self._rng.uniform(
            low=-bound, high=bound, size=(vocab_size, self.vector_size)
        ).astype(self.dtype)
        self.W_prime = self._rng.uniform(
            low=-bound, high=bound, size=(vocab_size, self.vector_size)
        ).astype(self.dtype)

    def _prepare_noise_distribution(self) -> None:
        """
        Prepare noise distribution for negative sampling.
        More frequent words have higher probability of being selected.
        Applies subsampling with the ns_exponent parameter to prevent 
        extremely common words from dominating.
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
        """
        if self.noise_distribution is None or len(self.noise_distribution) == 0:
            self._cum_table = np.array([], dtype=np.uint32)
            return self._cum_table
        
        # Build cumulative table using NumPy
        # Domain is 2^31 - 1 (max value for signed 32-bit int, used for binary search)
        domain = 2147483647.0
        cumsum = np.cumsum(self.noise_distribution)
        self._cum_table = np.minimum(cumsum * domain, domain).astype(np.uint32)
        return self._cum_table
    
    def _get_random_state(self) -> int:
        """Get current random state for training.
        
        Returns an integer seed for the Cython training code. If seed is None,
        generates a random seed using the RNG.
        """
        if not hasattr(self, '_random_state'):
            if self.seed is not None:
                self._random_state = self.seed
            else:
                # Generate random seed from RNG when no seed was provided
                self._random_state = self._rng.randint(0, 2**32 - 1)
        return self._random_state
    
    def _set_random_state(self, state: int) -> None:
        """Update random state after training."""
        self._random_state = state
    
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
        
    def _train_batch(
        self,
        batch: list[list[str]],
        sample_ints: np.ndarray,
        alpha: float,
        calculate_loss: bool
    ) -> tuple[float, int, int, int]:
        """
        Train on a single batch of sentences. Override in subclasses for different training logic.
        
        Args:
            batch: List of tokenized sentences (already batched by word count in Python).
            sample_ints: Subsampling thresholds as uint32 array.
            alpha: Learning rate for this batch.
            calculate_loss: Whether to compute and return loss.
        
        Returns:
            Tuple of (batch_loss, examples_count, words_count, new_random_state).
        """
        return word2vec_c.train_batch(
            self.W,
            self.W_prime,
            batch,
            self.vocab,
            sample_ints,
            self._cum_table,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            alpha,
            self.sg,
            self.negative,
            self.cbow_mean,
            self._get_random_state(),
            calculate_loss,
        )

    def train(self, sentences: Iterable[list[str]]) -> float | None:
        """
        Train word2vec model on given sentences.
        
        Processes sentences in batches using Cython-accelerated training.
        This approach is memory-efficient and works with both lists and iterables.
        
        Args:
            sentences: Tokenized sentences. Can be:
                - A list of sentences
                - A restartable iterable (e.g., file-backed iterator)
                
                Note: Single-use generators are not supported. The iterable must be
                restartable.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
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
        
        # Training loop for each epoch
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_examples = 0
            epoch_words = 0
            batch_count = 0
            
            # Progress tracking for this epoch
            LOSS_HISTORY_SIZE = 100
            
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
            
            # Iterate through batches (batched by word count, like Gensim)
            for batch in self._iter_batches(sentences, batch_words=batch_size):
                batch_count += 1
                
                # Compute learning rate for this batch based on global word progress
                if decay_alpha and total_words_all_epochs > 0:
                    progress = min(global_words_processed / total_words_all_epochs, 1.0)
                    batch_alpha = start_alpha + (min_alpha_val - start_alpha) * progress
                else:
                    batch_alpha = start_alpha
                
                # Train on this batch (hook method - can be overridden by subclasses)
                batch_loss, batch_examples_count, batch_words_count, new_random_state = self._train_batch(
                    batch, sample_ints, batch_alpha, calculate_loss
                )
                self._set_random_state(new_random_state)
                
                # Accumulate stats
                epoch_loss += batch_loss
                epoch_examples += batch_examples_count
                epoch_words += batch_words_count
                global_words_processed += batch_words_count
                
                # Track loss history
                if batch_examples_count > 0 and batch_loss > 0:
                    batch_avg_loss = batch_loss / batch_examples_count
                    recent_losses.append(batch_avg_loss)
                    if len(recent_losses) > LOSS_HISTORY_SIZE:
                        recent_losses.pop(0)
                
                # Update progress bar
                if progress_bar is not None:
                    recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                    if decay_alpha:
                        postfix_str = f"loss={recent_avg:.6f}, lr={batch_alpha:.6f}"
                    else:
                        postfix_str = f"loss={recent_avg:.6f}"
                    progress_bar.set_postfix_str(postfix_str)
                    progress_bar.update(batch_words_count)
            
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
                          f"{epoch_words:,} words, {batch_count} batches in {elapsed:.1f}s")
        
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
    
    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        """
        Find the topn most similar words to the given word.
        
        Args:
            word: Input word.
            topn: Number of similar words to return.
        
        Returns:
            List of (word, similarity) tuples.
        """
        if word not in self.vocab:
            return []
        
        word_idx = self.vocab[word]
        word_vec = self.W[word_idx].reshape(1, -1)
        
        # Compute cosine similarities using vectors module
        sim = cosine_similarity(word_vec, self.W).flatten()
        
        # Get top similar words, excluding the input word
        most_similar_words = []
        for idx in (-sim).argsort():
            if idx != word_idx and len(most_similar_words) < topn:
                most_similar_words.append((self.index2word[idx], float(sim[idx])))
        
        return most_similar_words

    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word.
            word2: Second word.
        
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
        word2_vec = self.W[self.vocab[word2]]
        
        # Use the vectors module for cosine similarity
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


class TempRefWord2Vec(Word2Vec):
    """
    Implementation of Word2Vec with Temporal Referencing (TR) for tracking semantic change.
    
    This class extends Word2Vec to implement temporal referencing, where target words
    are represented with time period indicators (e.g., "bread_1800" for period 1800s) when used
    as target words, but remain unchanged when used as context words.
    
    The class takes multiple corpora corresponding to different time periods and automatically
    creates temporal references for specified target words.
    
    Note:
        This implementation only supports Skip-gram (sg=1). CBOW is not supported.
    
    Args:
        corpora (dict[str, list[list[str]]]): Dictionary mapping time period labels to corpora.
            Each corpus is a list of sentences (each sentence is a list of tokens).
            Example: {"1800s": [["bread", "baker"], ["food", "eat"]], 
                      "1900s": [["bread", "supermarket"], ["food", "buy"]]}
        targets (list[str]): List of target words to trace semantic change.
        balance (bool): Whether to balance the corpora to equal sizes (default: True).
            When True, larger corpora are downsampled to match the smallest corpus.
        **kwargs: Arguments passed to Word2Vec parent class. Common options include:
            - vector_size (int): Dimensionality of word vectors (default: 100)
            - window (int): Context window size (default: 5)
            - min_word_count (int): Minimum word frequency threshold (default: 5)
            - negative (int): Number of negative samples (default: 5)
            - epochs (int): Number of training epochs (default: 1)
            - alpha (float): Initial learning rate (default: 0.025)
            - verbose (bool): Whether to log progress (default: False)
            
            Note: sg must be 1 (Skip-gram) - CBOW is not supported.
    
    Example:
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        # Corpora from different time periods as a dictionary
        corpora = {
            "1800s": [["bread", "baker", "oven"], ["food", "eat", "cook"]],
            "1900s": [["bread", "supermarket", "buy"], ["food", "restaurant", "order"]]
        }
        
        # Initialize and train model (training starts automatically)
        model = TempRefWord2Vec(
            corpora=corpora,
            targets=["bread", "food", "money"],
            vector_size=100,
            window=5,
            sg=1  # Skip-gram required
        )
        
        # Analyze semantic change (model is already trained)
        model.most_similar("bread_1800s")  # Words similar to "bread" in the 1800s
        model.most_similar("bread_1900s")  # Words similar to "bread" in the 1900s
    """
    
    @staticmethod
    def _sample_sentences_to_token_count(
        corpus: list[list[str]], 
        target_tokens: int,
        seed: int | None = None
    ) -> list[list[str]]:
        """
        Sample sentences from a corpus until the target token count is reached.
        
        This method randomly selects sentences from the corpus until the total number
        of tokens reaches or slightly exceeds the target count. This is useful for balancing
        corpus sizes when comparing different time periods or domains.
        
        Args:
            corpus: A list of sentences, where each sentence is a list of tokens.
            target_tokens: The target number of tokens to sample.
            seed: Random seed for reproducibility. If None, uses global seed.
        
        Returns:
            A list of sampled sentences with token count close to target_tokens.
        """
        rng = get_rng(resolve_seed(seed))
        sampled_sentences = []
        current_tokens = 0
        sentence_indices = list(range(len(corpus)))
        rng.shuffle(sentence_indices)
        
        for idx in sentence_indices:
            sentence = corpus[idx]
            if current_tokens + len(sentence) <= target_tokens:
                sampled_sentences.append(sentence)
                current_tokens += len(sentence)
            if current_tokens >= target_tokens:
                break
        return sampled_sentences

    @staticmethod
    def _add_corpus_tags(
        corpora: dict[str, list[list[str]]], 
        target_words: list[str]
    ) -> dict[str, list[list[str]]]:
        """
        Add corpus-specific tags to target words in all corpora at once.
        
        Args:
            corpora: Dictionary mapping labels to corpora (each corpus is list of tokenized sentences).
            target_words: List of words to tag.
        
        Returns:
            Dictionary of processed corpora where target words have been tagged with their corpus label.
        """
        processed_corpora = {}
        target_words_set = set(target_words)
        
        for label, corpus in corpora.items():
            processed_corpus = []
            for sentence in corpus:
                processed_sentence = []
                for token in sentence:
                    if token in target_words_set:
                        processed_sentence.append(f"{token}_{label}")
                    else:
                        processed_sentence.append(token)
                processed_corpus.append(processed_sentence)
            processed_corpora[label] = processed_corpus
        
        return processed_corpora

    def __init__(
        self,
        corpora: dict[str, list[list[str]]],  # Dictionary mapping labels to corpora
        targets: list[str],                   # Target words to trace semantic change
        balance: bool = True,                 # Whether to balance the corpora
        _skip_init: bool = False,             # Used by load() to skip normal initialization
        _labels: list[str] | None = None,     # Internal: used by load() to restore labels
        **kwargs                              # Parameters passed to Word2Vec parent class
    ):
        # _skip_init is used by load() to create an empty shell that will be populated
        # with saved state. This avoids unnecessary corpus processing and training.
        if _skip_init:
            self.labels = _labels if _labels is not None else []
            self.targets = targets
            self.combined_corpus = []
            self.period_vocab_counts = {}
            self.temporal_word_map = {}
            self.reverse_temporal_map = {}
            # Initialize parent with _skip_init to avoid training
            super().__init__(_skip_init=True, **kwargs)
            return
        
        # Validate corpora is a dictionary
        if not isinstance(corpora, dict):
            raise TypeError(f"corpora must be a dictionary mapping labels to corpora, got {type(corpora).__name__}")
    
        # check if sg = 1, else raise NotImplementedError
        if kwargs.get('sg') != 1:
            raise NotImplementedError("TempRefWord2Vec only supports Skip-gram model (sg=1)")
        
        # Extract labels from corpora dictionary keys
        labels = list(corpora.keys())
        
        # Store labels and targets as instance variables
        self.labels = labels
        self.targets = targets
        
        # Extract verbose from kwargs for logging before super().__init__
        verbose = kwargs.get('verbose', False)
        
        # print how many sentences in each corpus (each corpus a list of sentences)
        # and total size of each corpus (how many words; each sentence a list of words)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if verbose and not all(len(corpus) == 0 for corpus in corpora.values()):
            for label, corpus in corpora.items():
                logger.info(f"Corpus {label} has {len(corpus)} sentences and {sum(len(sentence) for sentence in corpus)} words")

        # Calculate token counts and determine minimum
        if balance:
            corpus_token_counts = {label: sum(len(sentence) for sentence in corpus) for label, corpus in corpora.items()}
            target_token_count = min(corpus_token_counts.values())
            if verbose:
                logger.info(f"Balancing corpora to minimum size: {target_token_count} tokens")
            
            # Balance corpus sizes
            balanced_corpora = {}
            for label, corpus in corpora.items():
                if corpus_token_counts[label] <= target_token_count:
                    balanced_corpora[label] = corpus
                else:
                    sampled_corpus = self._sample_sentences_to_token_count(corpus, target_token_count)
                    balanced_corpora[label] = sampled_corpus
        
            # Add corpus tags to the corpora
            tagged_corpora = self._add_corpus_tags(balanced_corpora, targets)
        else:
            tagged_corpora = self._add_corpus_tags(corpora, targets)

        # Initialize combined corpus before using it
        self.combined_corpus = []
        
        # Calculate vocab counts for each period before combining
        self.period_vocab_counts = {}
        
        for label, corpus in tagged_corpora.items():
            period_counter = Counter()
            for sentence in corpus:
                period_counter.update(sentence)
            self.period_vocab_counts[label] = period_counter
            # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
            if verbose and not all(len(c) == 0 for c in corpora.values()):
                logger.info(f"Period '{label}': {len(period_counter)} unique tokens, {sum(period_counter.values())} total tokens")
        
        # Combine all tagged corpora
        for corpus in tagged_corpora.values():
            self.combined_corpus.extend(corpus)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if verbose and not all(len(c) == 0 for c in corpora.values()):
            logger.info(f"Combined corpus: {len(self.combined_corpus)} sentences, {sum(len(s) for s in self.combined_corpus)} tokens")
        
        
        # Create temporal word map: maps base words to their temporal variants
        self.temporal_word_map = {}
        for target in targets:
            variants = [f"{target}_{label}" for label in labels]
            self.temporal_word_map[target] = variants
        
        # Create reverse mapping: temporal variant -> base word
        self.reverse_temporal_map = {}
        for base_word, variants in self.temporal_word_map.items():
            for variant in variants:
                self.reverse_temporal_map[variant] = base_word
        
        # Initialize parent Word2Vec class with kwargs (without sentences - we handle training ourselves)
        super().__init__(**kwargs)
        
        # Auto-train if combined_corpus is non-empty (skip when loading from saved model with dummy corpora)
        if self.combined_corpus:
            self.train()
    
    def build_vocab(self, sentences: list[list[str]]) -> None:
        """
        Extends the parent build_vocab method to handle temporal word variants.
        Explicitly adds base words to the vocabulary even if they don't appear in the corpus.
        
        Args:
            sentences: List of tokenized sentences.
        """
        
        # Call parent method to build the basic vocabulary
        super().build_vocab(sentences)
        
        # Verify all temporal variants are in the vocabulary
        # If any variant is missing, issue a warning
        missing_variants = []
        for base_word, variants in self.temporal_word_map.items():
            for variant in variants:
                if variant not in self.vocab:
                    missing_variants.append(variant)
        
        if missing_variants:
            logger.warning(f"{len(missing_variants)} temporal variants not found in corpus:")
            logger.warning(f"Sample: {missing_variants[:10]}")
            logger.warning("These variants will not be part of the temporal analysis.")
        
        # Add base words to vocabulary with counts derived from their temporal variants.
        # Since context words are converted from tagged form (e.g., 张爱玲_民国) to base form
        # (e.g., 张爱玲), the base word's effective frequency equals the sum of all its variants.
        # This ensures proper negative sampling probability for base words.
        # Reference: Dubossarsky et al. (2019) "Time-Out: Temporal Referencing for Robust 
        # Modeling of Lexical Semantic Change" - context words share space across time periods.
        added_base_words = 0
        total_base_count = 0
        skipped_base_words = []
        for base_word, variants in self.temporal_word_map.items():
            # Calculate base word count as sum of all temporal variant counts
            base_count = sum(
                self.word_counts.get(variant, 0) 
                for variant in variants
            )
            
            # Only add base word if at least one variant has counts
            # (otherwise no training examples would use this base word as context)
            if base_count == 0:
                skipped_base_words.append(base_word)
                continue
            
            if base_word not in self.vocab:
                # Add the base word to vocabulary
                word_id = len(self.index2word)
                self.vocab[base_word] = word_id
                self.index2word.append(base_word)
                self.word_counts[base_word] = base_count
                added_base_words += 1
                total_base_count += base_count
            else:
                # Base word already in vocab (shouldn't happen normally)
                # Update its count to be the sum of variants
                self.word_counts[base_word] = base_count
        
        if skipped_base_words:
            logger.warning(f"Skipped {len(skipped_base_words)} base words with no variant counts: {skipped_base_words[:5]}...")
        
        if added_base_words > 0:
            # Add the base word counts to corpus total (for proper frequency normalization)
            self.corpus_word_count += total_base_count
        
        # Build the temporal index map for Cython acceleration
        self._build_temporal_index_map()
    
    def _build_temporal_index_map(self) -> None:
        """
        Build a numpy array for fast temporal index mapping in Cython.
        
        The array maps word indices to their base form indices:
        - For temporal variants: temporal_index_map[variant_idx] = base_word_idx
        - For regular words: temporal_index_map[word_idx] = word_idx (identity)
        
        This allows O(1) lookup during example generation in Cython instead of
        Python dictionary lookups.
        """
        vocab_size = len(self.vocab)
        
        # Initialize with identity mapping (each index maps to itself)
        self.temporal_index_map = np.arange(vocab_size, dtype=np.int32)
        
        # Override mappings for temporal variants
        for variant_word, base_word in self.reverse_temporal_map.items():
            if variant_word in self.vocab and base_word in self.vocab:
                variant_idx = self.vocab[variant_word]
                base_idx = self.vocab[base_word]
                self.temporal_index_map[variant_idx] = base_idx
        
        logger.debug(f"Built temporal index map with {len(self.reverse_temporal_map)} variant mappings")
    
    def _train_batch(
        self,
        batch: list[list[str]],
        sample_ints: np.ndarray,
        alpha: float,
        calculate_loss: bool
    ) -> tuple[float, int, int, int]:
        """
        Train on a single batch using temporal-aware training.
        
        Overrides the parent method to use temporal mapping that converts
        context words to their base forms while keeping center words as temporal variants.
        
        Args:
            batch: List of tokenized sentences (already batched by word count in Python).
            sample_ints: Subsampling thresholds as uint32 array.
            alpha: Learning rate for this batch.
            calculate_loss: Whether to compute and return loss.
        
        Returns:
            Tuple of (batch_loss, examples_count, words_count, new_random_state).
        """
        # Ensure temporal index map is built
        if not hasattr(self, 'temporal_index_map') or self.temporal_index_map is None:
            self._build_temporal_index_map()
        
        return word2vec_c.train_batch_temporal(
            self.W,
            self.W_prime,
            batch,
            self.vocab,
            self.temporal_index_map,
            sample_ints,
            self._cum_table,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            alpha,
            self.negative,
            self._get_random_state(),
            calculate_loss,
        )
    
    def train(self, sentences: list[list[str]] | None = None) -> float | None:
        """
        Train the TempRefWord2Vec model using the preprocessed combined corpus.
        
        Unlike the parent Word2Vec class, TempRefWord2Vec always uses its internal combined_corpus
        that was created and preprocessed during initialization. This ensures the training
        data has the proper temporal references.
        
        All training configuration (epochs, batch_size, alpha, min_alpha, etc.) is read
        from instance attributes set during initialization via ``**kwargs``.
        
        Args:
            sentences: Ignored in TempRefWord2Vec, will use self.combined_corpus instead.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        if sentences is not None:
            logger.warning("TempRefWord2Vec always uses its internal preprocessed corpus for training.")
            logger.warning("The provided 'sentences' argument will be ignored (using self.combined_corpus instead).")
        
        # Call the parent's train method with our combined corpus
        return super().train(sentences=self.combined_corpus)

    def calculate_semantic_change(self, target_word: str, labels: list[str] | None = None) -> dict[str, list[tuple[str, float]]]:
        """
        Calculate semantic change by comparing cosine similarities across time periods.
        
        Args:
            target_word: Target word to analyze (must be one of the targets specified 
                during initialization).
            labels: Time period labels (optional, defaults to labels from model initialization).
        
        Returns:
            Dict mapping transition names to lists of (word, change) tuples, sorted by 
            change score (descending).
        
        Example:
            changes = model.calculate_semantic_change("人民")
            for transition, word_changes in changes.items():
                print(f"\\n{transition}:")
                print("Words moved towards:", word_changes[:5])  # Top 5 increases
                print("Words moved away:", word_changes[-5:])   # Top 5 decreases
        """
        # Use stored labels if not provided
        if labels is None:
            labels = self.labels
        
        # Validate that target_word is one of the tracked targets
        if target_word not in self.targets:
            raise ValueError(f"Target word '{target_word}' was not specified during model initialization. "
                           f"Available targets: {self.targets}")
        
        results = {}
        
        # Get all words in vocabulary (excluding temporal variants)
        all_words = [word for word in self.vocab.keys() 
                    if word not in self.reverse_temporal_map]
        
        # Get embeddings for all words
        all_word_vectors = np.array([self.get_vector(word) for word in all_words])

        # For each adjacent pair of time periods
        for i in range(len(labels) - 1):
            from_period = labels[i]
            to_period = labels[i+1]
            transition = f"{from_period}_to_{to_period}"
            
            # Get temporal variants for the target word
            from_variant = f"{target_word}_{from_period}"
            to_variant = f"{target_word}_{to_period}"
            
            # Check if temporal variants exist in vocabulary
            if from_variant not in self.vocab or to_variant not in self.vocab:
                logger.warning(f"{from_variant} or {to_variant} not found in vocabulary. Skipping transition {transition}.")
                continue
            
            # Get vectors for the target word in each period
            from_vector = self.get_vector(from_variant).reshape(1, -1)
            to_vector = self.get_vector(to_variant).reshape(1, -1)
            
            # Calculate cosine similarity for all words with the target word in each period
            from_sims = cosine_similarity(from_vector, all_word_vectors)[0]
            to_sims = cosine_similarity(to_vector, all_word_vectors)[0]
            
            # Calculate differences in similarity
            sim_diffs = to_sims - from_sims
            
            # Create word-change pairs and sort by change
            word_changes = [(all_words[j], float(sim_diffs[j])) for j in range(len(all_words))]
            word_changes.sort(key=lambda x: x[1], reverse=True)
            
            results[transition] = word_changes
        
        return results

    def get_available_targets(self) -> list[str]:
        """
        Get the list of target words available for semantic change analysis.
        
        Returns:
            List of target words that were specified during model initialization.
        """
        return self.targets.copy()

    def get_time_labels(self) -> list[str]:
        """
        Get the list of time period labels used in the model.
        
        Returns:
            List of time period labels that were specified during model initialization.
        """
        return self.labels.copy()
    
    def get_period_vocab_counts(self, period: str | None = None) -> dict[str, Counter] | Counter:
        """
        Get vocabulary counts for a specific period or all periods.
        
        Args:
            period: The period label to get vocab counts for. If None, returns all periods.
            
        Returns:
            If period is None: dictionary mapping period labels to Counter objects.
            If period is specified: Counter object for that specific period.
            
        Raises:
            ValueError: If the specified period is not found in the model.
        """
        if not hasattr(self, 'period_vocab_counts'):
            raise AttributeError("Vocabulary counts not available. Make sure the model has been initialized properly.")
            
        if period is None:
            return self.period_vocab_counts.copy()
        else:
            if period not in self.period_vocab_counts:
                available_periods = list(self.period_vocab_counts.keys())
                raise ValueError(f"Period '{period}' not found. Available periods: {available_periods}")
            return self.period_vocab_counts[period].copy()
    
    def save(self, path: str) -> None:
        """
        Save the TempRefWord2Vec model to a file, including vocab counts and temporal metadata.
        
        This overrides the parent save method to also save:
        - Period-specific vocabulary counts
        - Target words and labels  
        - Temporal word mappings
        - All other model parameters from the parent class
        
        Note: The combined corpus is NOT saved to reduce file size.
        
        Args:
            path (str): Path to save the model file.
        """
        # Get the base model data from parent class
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
        
        # Add TempRefWord2Vec-specific data
        tempref_data = {
            'labels': self.labels,
            'targets': self.targets,
            'temporal_word_map': self.temporal_word_map,
            'reverse_temporal_map': self.reverse_temporal_map,
            'period_vocab_counts': {label: dict(counter) for label, counter in self.period_vocab_counts.items()},
            'model_type': 'TempRefWord2Vec'
        }
        
        # Combine all data
        model_data.update(tempref_data)
        
        # Save to file
        np.save(path, model_data, allow_pickle=True)
        if self.verbose:
            logger.info(f"TempRefWord2Vec model saved to {path}")
            logger.info(f"Saved data includes:")
            logger.info(f"  - Vocabulary: {len(self.vocab)} words")
            logger.info(f"  - Time periods: {len(self.labels)} ({', '.join(self.labels)})")
            logger.info(f"  - Target words: {len(self.targets)} ({', '.join(self.targets)})")
            logger.info(f"  - Period vocab counts: {len(self.period_vocab_counts)} periods")
    
    @classmethod
    def load(cls, path: str) -> 'TempRefWord2Vec':
        """
        Load a TempRefWord2Vec model from a file.
        
        This overrides the parent load method to also restore:
        - Period-specific vocabulary counts
        - Target words and labels  
        - Temporal word mappings
        
        Args:
            path (str): Path to load the model from.
        
        Returns:
            TempRefWord2Vec: Loaded TempRefWord2Vec model with all temporal metadata 
                restored.
        
        Raises:
            ValueError: If the file doesn't contain TempRefWord2Vec data.
        """
        model_data = np.load(path, allow_pickle=True).item()
        
        # Check if this is a TempRefWord2Vec model
        if model_data.get('model_type') != 'TempRefWord2Vec':
            raise ValueError("The loaded file does not contain a TempRefWord2Vec model. "
                           "Use Word2Vec.load() for regular Word2Vec models.")
        
        # Extract TempRefWord2Vec-specific data
        labels = model_data['labels']
        targets = model_data['targets']
        
        # Get base model parameters with defaults
        shrink_windows = model_data.get('shrink_windows', False)
        sample = model_data.get('sample', 1e-3)
        max_vocab_size = model_data.get('max_vocab_size', None)
        
        # Create model instance with _skip_init to avoid unnecessary initialization
        # Empty dict is passed for corpora - won't be processed with _skip_init
        model = cls(
            corpora={},  # Empty dict - won't be used with _skip_init
            targets=targets,
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
            _labels=labels,
        )
        
        # Restore saved model state
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
        # Restore corpus statistics
        model.corpus_word_count = model_data.get('corpus_word_count', sum(model.word_counts.values()))
        model.total_corpus_tokens = model_data.get('total_corpus_tokens', model.corpus_word_count)
        
        # Restore TempRefWord2Vec-specific data
        model.temporal_word_map = model_data['temporal_word_map']
        model.reverse_temporal_map = model_data['reverse_temporal_map']
        model.period_vocab_counts = {
            label: Counter(counts_dict) 
            for label, counts_dict in model_data['period_vocab_counts'].items()
        }
        
        # Build temporal index map for Cython acceleration
        model._build_temporal_index_map()
        
        # Clear the dummy combined_corpus to save memory
        model.combined_corpus = []
        
        if model.verbose:
            logger.info(f"TempRefWord2Vec model loaded from {path}")
            logger.info(f"Restored data includes:")
            logger.info(f"  - Vocabulary: {len(model.vocab)} words")
            logger.info(f"  - Time periods: {len(model.labels)} ({', '.join(model.labels)})")
            logger.info(f"  - Target words: {len(model.targets)} ({', '.join(model.targets)})")
            logger.info(f"  - Period vocab counts: {len(model.period_vocab_counts)} periods")
        
        return model

