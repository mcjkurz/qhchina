"""
Sample-based Word2Vec implementation
-----------------------------------
This module implements the Word2Vec algorithm with both CBOW and Skip-gram models,
accelerated with Cython for fast training.

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
"""

import logging
import numpy as np
from collections import Counter
from collections.abc import Iterator, Iterable
from types import GeneratorType
from typing import List, Dict, Tuple, Optional, Union, Callable
from tqdm.auto import tqdm
import warnings
import time
from .vectors import cosine_similarity
from ..config import get_rng, resolve_seed
from .cython_ext import word2vec as word2vec_c

logger = logging.getLogger("qhchina.analytics.word2vec")


__all__ = [
    'Word2Vec',
    'TempRefWord2Vec',
    'sample_sentences_to_token_count',
    'add_corpus_tags',
]


class Word2Vec:
    """
    Implementation of Word2Vec algorithm with Cython-accelerated training.
    
    This class implements both Skip-gram and CBOW architectures:
    - Skip-gram (sg=1): Each training example is (input_idx, output_idx) where input is the center word
      and output is a context word.
    - CBOW (sg=0): Each training example is (input_indices, output_idx) where inputs are context words
      and output is the center word.
    
    Training is performed using optimized Cython routines with BLAS operations for maximum speed.
    
    If ``sentences`` is provided at initialization, training starts immediately. Otherwise, call
    :meth:`train` later with the sentences to train on.
    
    Features:
    - CBOW and Skip-gram architectures
    - Cython-accelerated training with BLAS operations
    - Negative sampling for each training example
    - Subsampling of frequent words
    - Dynamic window sizing with shrink_windows parameter
    - Properly managed learning rate decay
    - Vocabulary size restriction with max_vocab_size parameter
    
    Args:
        sentences (list of list of str, optional): Tokenized sentences for training. If provided,
            training starts immediately during initialization.
        vector_size (int): Dimensionality of the word vectors (default: 100).
        window (int): Maximum distance between the current and predicted word (default: 5).
        min_word_count (int): Ignores all words with frequency lower than this (default: 5).
        negative (int): Number of negative samples for negative sampling (default: 5).
        ns_exponent (float): Exponent used to shape the negative sampling distribution (default: 0.75).
        cbow_mean (bool): If True, use mean of context word vectors, else use sum (default: True).
        sg (int): Training algorithm: 1 for skip-gram; 0 for CBOW (default: 0).
        seed (int): Seed for random number generator (default: 1).
        alpha (float): Initial learning rate (default: 0.025).
        min_alpha (float): Minimum learning rate. If None, learning rate remains constant at alpha.
        sample (float): Threshold for subsampling frequent words. Default is 1e-3, set to 0 to disable.
        shrink_windows (bool): If True, the effective window size is uniformly sampled from [1, window] 
            for each target word during training. If False, always use the full window (default: True).
        max_vocab_size (int): Maximum vocabulary size to keep, keeping the most frequent words.
            None means no limit (keep all words above min_word_count).
        epochs (int): Number of training iterations over the corpus (default: 1).
        batch_size (int): Number of sentences to process per batch (default: 10000).
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
        sentences: Optional[List[List[str]]] = None,
        vector_size: int = 100,
        window: int = 5,
        min_word_count: int = 5,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_mean: bool = True,
        sg: int = 0,
        seed: int = 1,
        alpha: float = 0.025,
        min_alpha: Optional[float] = None,
        sample: float = 1e-3,
        shrink_windows: bool = True,
        max_vocab_size: Optional[int] = None,
        verbose: bool = False,
        epochs: int = 1,
        batch_size: int = 10000,
        callbacks: Optional[List[Callable]] = None,
        calculate_loss: bool = True,
        total_examples: Optional[int] = None,
    ):
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
        self.vocab_size = 0  # Number of words in vocabulary
        self.corpus_word_count = 0  # Token count for words IN vocabulary
        self.total_corpus_tokens = 0  # Total token count (including OOV words)
        self.discard_probs = None  # Numpy array for subsampling frequent words (indexed by word ID)
        
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
        
        # For tracking training progress
        self.epoch_losses = []
        self.total_examples = 0
        
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

    def build_vocab(self, sentences: Iterable[List[str]]) -> None:
        """
        Build vocabulary from sentences.
        
        Args:
            sentences: Iterable of tokenized sentences (each sentence is a list of words).
                Can be a list (for fast path) or any iterable (for streaming path).
                If an iterable/generator is provided, it will be consumed during vocab building.
        
        Raises:
            ValueError: If sentences is empty or contains no words.
        """
        # Count word occurrences
        # For iterables, we can't show total progress, but we can still count
        self.word_counts = Counter()
        sentence_count = 0
        
        # Check if we can determine length for progress bar
        if hasattr(sentences, '__len__'):
            iterator = tqdm(sentences, desc="Building vocabulary", unit="sent", leave=True)
        else:
            # For generators/iterators, show progress without total
            iterator = tqdm(sentences, desc="Building vocabulary", unit="sent", leave=True)
        
        for sentence in iterator:
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
        self.vocab_size = len(self.vocab)
        
        if self.verbose:
            logger.info(f"Vocabulary built: {self.vocab_size:,} words, {self.corpus_word_count:,} tokens in vocab, {self.total_corpus_tokens:,} total tokens")

    def _calculate_discard_probs(self) -> None:
        """
        Calculate the probability of discarding frequent words during subsampling.
        
        Formula from the original word2vec paper:
        
        $P(w_i) = 1 - \\sqrt{\\frac{t}{f(w_i)}}$
        
        where $t$ is the sample threshold and $f(w_i)$ is the word frequency 
        normalized by the total corpus word count.
        
        A word will be discarded with probability $P(w_i)$.
        
        Creates a numpy array indexed by word ID for fast lookup during example generation.
        """
        self.discard_probs = np.zeros(len(self.vocab), dtype=np.float32)
        total_words = self.corpus_word_count
        
        for word, idx in self.vocab.items():
            # Calculate normalized word frequency
            word_freq = self.word_counts[word] / total_words
            # Calculate probability of discarding the word
            discard_prob = 1.0 - np.sqrt(self.sample / word_freq)
            # Clamp the probability to [0, 1]
            self.discard_probs[idx] = max(0.0, min(1.0, discard_prob))

    def _estimate_example_count(self) -> int:
        """
        Estimate total training examples per epoch without iterating through data.
        
        Used for learning rate decay scheduling when min_alpha is provided.
        The estimate accounts for subsampling, vocabulary coverage, and window effects.
        
        For Skip-gram: each retained word generates ~2*avg_window context pairs,
            but context words must also be in vocabulary and survive subsampling.
        For CBOW: each retained word generates 1 example (with context as input).
        
        Returns:
            Estimated number of training examples per epoch.
        """
        # Vocabulary coverage: probability that a random word position is in vocabulary
        # This is computed exactly from word counts, replacing the old magic constants
        if self.total_corpus_tokens > 0:
            vocab_coverage = self.corpus_word_count / self.total_corpus_tokens
        else:
            vocab_coverage = 1.0
        
        # Edge factor: accounts for sentence boundary effects where context windows
        # are truncated. This is the only remaining heuristic (~5% loss estimate).
        edge_factor = 0.95
        
        # Calculate effective word count after subsampling
        if self.sample > 0 and self.discard_probs is not None:
            # Expected retained words = sum of (count * keep_probability)
            effective_words = sum(
                self.word_counts[word] * (1.0 - self.discard_probs[idx])
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
        Initialize word vectors
        """
        vocab_size = len(self.vocab)
        if self.verbose:
            logger.info(f"Initializing vectors: 2 matrices of shape ({vocab_size:,}, {self.vector_size})...")
        
        # Initialize input and output matrices
        # Using Xavier/Glorot initialization for better convergence
        # Range is [-0.5/dim, 0.5/dim]
        bound = 0.5 / self.vector_size
        self.W = self._rng.uniform(
            low=-bound, 
            high=bound, 
            size=(vocab_size, self.vector_size)
        ).astype(self.dtype)
        
        # Initialize W_prime with small random values like W instead of zeros
        # This helps improve convergence during training
        self.W_prime = self._rng.uniform(
            low=-bound, 
            high=bound, 
            size=(vocab_size, self.vector_size)
        ).astype(self.dtype)

    def _prepare_noise_distribution(self) -> None:
        """
        Prepare noise distribution for negative sampling.
        More frequent words have higher probability of being selected.
        Applies subsampling with the ns_exponent parameter to prevent 
        extremely common words from dominating.
        """
        
        # Get counts of each word in the vocabulary
        word_counts = np.array([self.word_counts[word] for word in self.vocab])
        
        # Apply the exponent to smooth the distribution
        noise_dist = word_counts ** self.ns_exponent
        
        # Normalize to get a probability distribution
        noise_dist_normalized = noise_dist / np.sum(noise_dist)
        
        # Explicitly cast to the correct dtype (float32 or float64)
        self.noise_distribution = noise_dist_normalized.astype(self.dtype)

    def _initialize_cython_globals(self) -> None:
        """Initialize Cython module with noise distribution and hyperparameters."""
        # Ensure noise distribution is contiguous
        if not isinstance(self.noise_distribution, np.ndarray):
            raise ValueError("noise_distribution must be a numpy array")
        if not self.noise_distribution.flags['C_CONTIGUOUS']:
            self.noise_distribution = np.ascontiguousarray(self.noise_distribution, dtype=self.dtype)
        
        # Initialize Cython globals
        # Sync Cython RNG with the current seed for reproducibility
        word2vec_c.set_seed(self.seed)
        
        word2vec_c.init_globals(
            noise_distribution=self.noise_distribution,
            vector_size=self.vector_size,
            negative=self.negative,
            cbow_mean=int(self.cbow_mean),
        )
    
    def _compute_sample_ints(self) -> np.ndarray:
        """
        Compute subsampling thresholds as uint32 integers for fast comparison in Cython.
        
        Returns numpy array where sample_ints[word_id] is the threshold value.
        A word is kept if random_uint32 >= sample_ints[word_id].
        """
        vocab_size = len(self.vocab)
        sample_ints = np.zeros(vocab_size, dtype=np.uint32)
        
        if self.sample <= 0:
            # No subsampling - set thresholds to 0 (always keep)
            return sample_ints
        
        total_words = self.corpus_word_count
        
        for word, idx in self.vocab.items():
            word_freq = self.word_counts[word] / total_words
            # Keep probability = sqrt(sample / freq) (capped at 1.0)
            if word_freq > self.sample:
                keep_prob = np.sqrt(self.sample / word_freq)
            else:
                keep_prob = 1.0
            
            # Convert to uint32 threshold: word is kept if random < threshold
            # So threshold = keep_prob * 2^32 (we use 2^32 - 1 as max)
            sample_ints[idx] = min(np.uint32(keep_prob * 4294967295.0), np.uint32(4294967295))
        
        return sample_ints
    
    def _iter_chunks(self, sentences: Iterable[List[str]], chunk_size: int = 50000):
        """
        Yield chunks of sentences as lists.
        
        This allows streaming through large corpora without loading everything
        into memory at once.
        
        Args:
            sentences: Iterable of tokenized sentences.
            chunk_size: Maximum number of sentences per chunk.
        
        Yields:
            List of sentences (each chunk is a list).
        """
        chunk = []
        for sent in sentences:
            chunk.append(sent)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        
    def train(self, sentences: Union[List[List[str]], Iterable[List[str]]]) -> Optional[float]:
        """
        Train word2vec model on given sentences.
        
        Supports two training modes:
        
        1. **Fast path** (list input): When ``sentences`` is a list, the entire corpus
           is passed to Cython in a single call per epoch, minimizing Python/Cython
           boundary crossings. This is ~35% faster than Gensim for single-threaded training.
        
        2. **Streaming path** (iterable input): When ``sentences`` is a non-list iterable
           (e.g., file-backed iterator), sentences are processed in chunks to minimize
           memory usage. The iterable must be restartable (can be iterated multiple times).
        
        All training configuration (epochs, batch_size, alpha, min_alpha, etc.) is read
        from instance attributes set during initialization.
        
        Args:
            sentences: Tokenized sentences. Can be:
                - A list of sentences (fast path, requires all data in memory)
                - A restartable iterable (streaming path, memory-efficient)
                
                Note: Single-use generators will only work for 1 epoch and will raise
                a warning. For multi-epoch training with generators, convert to a list
                or use a restartable iterable (e.g., file-backed).
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        # Detect input type and choose training path
        is_list = isinstance(sentences, list)
        is_generator = isinstance(sentences, GeneratorType)
        
        # Warn about generators with multiple epochs
        if is_generator and self.epochs > 1:
            warnings.warn(
                f"Generator detected with epochs={self.epochs}. Generators can only be "
                "iterated once, so only the first epoch will have training data. "
                "For multi-epoch training, use a list or restartable iterable.",
                UserWarning
            )
        
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
        if self.sample > 0 and self.discard_probs is None:
            self._calculate_discard_probs()
        
        self._initialize_cython_globals()
        
        # Dispatch to appropriate training path
        if is_list:
            # FAST PATH: Pass entire list to Cython (minimal boundary crossings)
            return self._train_fast(sentences, decay_alpha)
        else:
            # STREAMING PATH: Process in chunks (memory-efficient)
            return self._train_streaming(sentences, decay_alpha)
    
    def _train_fast(self, sentences: List[List[str]], decay_alpha: bool) -> Optional[float]:
        """
        Fast training path for list input.
        
        Passes the entire sentence list to Cython in a single call per epoch,
        minimizing Python/Cython boundary crossings for maximum performance.
        
        Args:
            sentences: List of tokenized sentences.
            decay_alpha: Whether to decay learning rate.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
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
                # User provided the count
                examples_per_epoch = total_examples
                total_example_count = total_examples * epochs
            else:
                # Estimate examples mathematically (fast, avoids full iteration)
                examples_per_epoch = self._estimate_example_count()
                total_example_count = examples_per_epoch * epochs
                
        total_examples_processed = 0
        current_alpha = start_alpha = self.alpha
        
        if self.verbose:
            logger.info(f"Starting training (fast path): {epochs} epoch(s), {len(sentences):,} sentences, alpha={self.alpha}")

        # Training loop for each epoch
        for epoch in range(epochs):
            examples_processed_in_epoch = 0
            batch_count = 0
            epoch_loss = 0.0
            
            # Use unified training loop
            epoch_loss, examples_processed_in_epoch, batch_count, total_examples_processed, current_alpha = \
                self._train_epoch(
                    sentences=sentences,
                    epoch=epoch,
                    epochs=epochs,
                    batch_size=batch_size,
                    decay_alpha=decay_alpha,
                    total_example_count=total_example_count,
                    examples_per_epoch=examples_per_epoch,
                    total_examples_processed=total_examples_processed,
                    current_alpha=current_alpha,
                    start_alpha=start_alpha,
                    calculate_loss=calculate_loss,
                    recent_losses=recent_losses,
                    verbose=None,
                )
                
            # Add epoch loss to total
            if calculate_loss:
                total_loss += epoch_loss
                examples_processed_total += examples_processed_in_epoch
            
            # Update instance alpha to reflect current learning rate (for callbacks)
            if decay_alpha:
                self.alpha = current_alpha
            
            # Call any registered callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)
        
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
    
    def _train_streaming(self, sentences: Iterable[List[str]], decay_alpha: bool) -> Optional[float]:
        """
        Streaming training path for iterable input.
        
        Processes sentences in chunks, allowing training on corpora that don't fit
        in memory. The iterable must be restartable (can iterate multiple times)
        for multi-epoch training.
        
        This path is slower than the fast path due to more Python/Cython boundary
        crossings, but uses significantly less memory for large corpora.
        
        Args:
            sentences: Restartable iterable of tokenized sentences.
            decay_alpha: Whether to decay learning rate.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        # Read training configuration from instance attributes
        epochs = self.epochs
        callbacks = self.callbacks
        calculate_loss = self.calculate_loss
        total_examples = self.total_examples_hint
        
        # Chunk size for streaming (number of sentences per chunk)
        # Larger chunks = fewer Cython calls but more memory
        chunk_size = 50000  # 50K sentences per chunk
        
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
                examples_per_epoch = self._estimate_example_count()
                total_example_count = examples_per_epoch * epochs
        
        # Track progress across all epochs for learning rate decay
        total_words_all_epochs = self.corpus_word_count * epochs
        global_words_processed = 0
        
        start_alpha = self.alpha
        min_alpha_val = self.min_alpha if self.min_alpha else start_alpha
        
        if self.verbose:
            logger.info(f"Starting training (streaming path): {epochs} epoch(s), chunk_size={chunk_size:,}, alpha={self.alpha}")
        
        # Compute sample_ints for subsampling (same as fast path)
        sample_ints = self._compute_sample_ints()
        
        # Training loop for each epoch
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_examples = 0
            epoch_words = 0
            chunk_count = 0
            
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
            
            # Iterate through chunks
            for chunk in self._iter_chunks(sentences, chunk_size=chunk_size):
                chunk_count += 1
                
                # Compute learning rate for this chunk based on global progress
                if decay_alpha and total_words_all_epochs > 0:
                    progress = global_words_processed / total_words_all_epochs
                    progress = min(progress, 1.0)
                    chunk_alpha_start = start_alpha + (min_alpha_val - start_alpha) * progress
                    
                    # Estimate words in this chunk for end alpha
                    # Use average words per sentence from corpus stats
                    avg_words_per_sent = self.corpus_word_count / max(1, sum(1 for _ in []))  # Can't easily compute
                    # Rough estimate: chunk has about chunk_size * avg_sentence_length words
                    estimated_chunk_words = len(chunk) * (self.corpus_word_count / max(1, len(chunk) * chunk_count))
                    progress_end = min((global_words_processed + estimated_chunk_words) / total_words_all_epochs, 1.0)
                    chunk_alpha_end = start_alpha + (min_alpha_val - start_alpha) * progress_end
                else:
                    chunk_alpha_start = start_alpha
                    chunk_alpha_end = start_alpha
                
                # Train on this chunk
                chunk_loss, chunk_examples_count, chunk_words_count = word2vec_c.train_epoch(
                    self.W,
                    self.W_prime,
                    chunk,  # Pass the chunk (a list) to Cython
                    self.vocab,
                    sample_ints,
                    self.sample > 0,
                    self.window,
                    self.shrink_windows,
                    chunk_alpha_start,
                    chunk_alpha_end,
                    len(chunk) * 10,  # Rough estimate for intra-chunk LR decay
                    None,  # No callback for streaming (we handle progress in Python)
                    chunk_size,  # callback_every_n_sentences (won't be called with None callback)
                    self.sg,
                    calculate_loss,
                )
                
                # Accumulate stats
                epoch_loss += chunk_loss
                epoch_examples += chunk_examples_count
                epoch_words += chunk_words_count
                global_words_processed += chunk_words_count
                
                # Track loss history
                if chunk_examples_count > 0 and chunk_loss > 0:
                    chunk_avg_loss = chunk_loss / chunk_examples_count
                    recent_losses.append(chunk_avg_loss)
                    if len(recent_losses) > LOSS_HISTORY_SIZE:
                        recent_losses.pop(0)
                
                # Update progress bar
                if progress_bar is not None:
                    recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                    if decay_alpha:
                        postfix_str = f"loss={recent_avg:.6f}, lr={chunk_alpha_start:.6f}"
                    else:
                        postfix_str = f"loss={recent_avg:.6f}"
                    progress_bar.set_postfix_str(postfix_str)
                    progress_bar.update(chunk_words_count)
            
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
                          f"{epoch_words:,} words, {chunk_count} chunks in {elapsed:.1f}s")
        
        # Update the instance alpha to reflect the final learning rate
        if decay_alpha:
            self.alpha = self.min_alpha
        
        # Calculate and return the final average loss if requested
        if calculate_loss and examples_processed_total > 0:
            final_avg_loss = total_loss / examples_processed_total
            if self.verbose:
                logger.info(f"Training completed (streaming). Final average loss: {final_avg_loss:.6f}")
            return final_avg_loss
        
        return None

    def _train_epoch(
        self,
        sentences: List[List[str]],
        epoch: int,
        epochs: int,
        batch_size: int,
        decay_alpha: bool,
        total_example_count: int,
        examples_per_epoch: Optional[int],
        total_examples_processed: int,
        current_alpha: float,
        start_alpha: float,
        calculate_loss: bool,
        recent_losses: List[float],
        verbose: Optional[int] = None,
    ) -> Tuple[float, int, int, int, float]:
        """
        Train for one epoch using Cython-accelerated training.
        
        Returns
        -------
        Tuple of (epoch_loss, examples_processed, batch_count, total_examples_processed, current_alpha)
        """
        epoch_start_time = time.time()
        
        return self._train_epoch_cython(
            sentences=sentences,
            epoch=epoch,
            epochs=epochs,
            batch_size=batch_size,
            decay_alpha=decay_alpha,
            total_example_count=total_example_count,
            examples_per_epoch=examples_per_epoch,
            total_examples_processed=total_examples_processed,
            current_alpha=current_alpha,
            start_alpha=start_alpha,
            calculate_loss=calculate_loss,
            recent_losses=recent_losses,
            epoch_start_time=epoch_start_time,
            verbose=verbose,
        )

    def _train_epoch_cython(
        self,
        sentences: List[List[str]],
        epoch: int,
        epochs: int,
        batch_size: int,
        decay_alpha: bool,
        total_example_count: int,
        examples_per_epoch: Optional[int],
        total_examples_processed: int,
        current_alpha: float,
        start_alpha: float,
        calculate_loss: bool,
        recent_losses: List[float],
        epoch_start_time: float,
        verbose: Optional[int] = None,
    ) -> Tuple[float, int, int, int, float]:
        """
        Train for one epoch using fully optimized Cython.
        
        The entire epoch is processed in a SINGLE Cython call with minimal
        Python/Cython boundary crossings. Vocabulary lookup, subsampling,
        window processing, and training all happen in Cython.
        
        Progress is reported via callback from Cython.
        """
        def current_time_str() -> str:
            return time.strftime("%H:%M:%S")
        
        num_sentences = len(sentences)
        
        # Compute sample_ints for subsampling (gensim-style thresholds)
        sample_ints = self._compute_sample_ints()
        
        # Progress state (captured by closure for callback)
        progress_state = {
            'last_words': 0,
            'last_examples': 0,
            'last_time': epoch_start_time,
            'recent_losses': recent_losses,
            'batch_count': 0,
        }
        
        # Progress bar setup - use words processed (reliable) instead of estimated examples (unreliable)
        # We know exactly how many vocabulary words are in the corpus (corpus_word_count)
        use_simple_logging = not decay_alpha and examples_per_epoch is None
        progress_bar = None
        
        if calculate_loss and not use_simple_logging:
            # Use words as progress metric since we know exact word count
            bar_format = '{l_bar}{bar}| {percentage:.1f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            
            progress_bar = tqdm(
                desc=f"Epoch {epoch+1}/{epochs}",
                total=self.corpus_word_count,  # Use known word count instead of estimated examples
                bar_format=bar_format,
                unit=" tokens",  # Space prefix for "133k tokens/s" formatting
                unit_scale=True,
                mininterval=0.5
            )
        
        LOSS_HISTORY_SIZE = 100
        
        def progress_callback(words_processed: int, examples_processed: int, 
                              running_loss: float, current_lr: float):
            """Called periodically from Cython to report progress."""
            nonlocal current_alpha
            
            progress_state['batch_count'] += 1
            batch_count = progress_state['batch_count']
            
            # Calculate batch statistics
            batch_words = words_processed - progress_state['last_words']
            batch_examples = examples_processed - progress_state['last_examples']
            if batch_examples > 0 and running_loss > 0:
                # Approximate batch loss from running loss
                batch_avg_loss = running_loss / examples_processed if examples_processed > 0 else 0.0
                progress_state['recent_losses'].append(batch_avg_loss)
                if len(progress_state['recent_losses']) > LOSS_HISTORY_SIZE:
                    progress_state['recent_losses'].pop(0)
            
            progress_state['last_words'] = words_processed
            progress_state['last_examples'] = examples_processed
            current_alpha = current_lr
            
            if not calculate_loss:
                return
            
            recent_avg = sum(progress_state['recent_losses']) / len(progress_state['recent_losses']) if progress_state['recent_losses'] else 0.0
            
            if use_simple_logging:
                if verbose is not None and batch_count % max(1, verbose) == 0:
                    elapsed = time.time() - epoch_start_time
                    ex_per_sec = examples_processed / elapsed if elapsed > 0 else 0
                    if decay_alpha:
                        print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} | words={words_processed:,} | loss={recent_avg:.6f} | lr={current_lr:.6f} | {ex_per_sec:.0f} ex/s")
                    else:
                        print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} | words={words_processed:,} | loss={recent_avg:.6f} | {ex_per_sec:.0f} ex/s")
            elif progress_bar is not None:
                if decay_alpha:
                    postfix_str = f"loss={recent_avg:.6f}, lr={current_lr:.6f}"
                else:
                    postfix_str = f"loss={recent_avg:.6f}"
                progress_bar.set_postfix_str(postfix_str)
                progress_bar.update(batch_words)  # Update by words processed, not examples
        
        # Callback interval: report every N sentences (tune for responsiveness vs overhead)
        # With smaller buffer sizes (50K words), we want more frequent callbacks
        # Aim for ~50-100 callbacks per epoch for smooth progress
        callback_interval = max(100, min(5000, num_sentences // 50))
        
        # Compute per-epoch start and end alpha for smooth linear interpolation
        # Alpha decays linearly from start_alpha to min_alpha across all epochs
        min_alpha_val = self.min_alpha if self.min_alpha else start_alpha
        if decay_alpha and epochs > 0:
            # epoch_start_alpha: learning rate at the START of this epoch
            # epoch_end_alpha: learning rate at the END of this epoch
            epoch_start_alpha = start_alpha + (min_alpha_val - start_alpha) * (epoch / epochs)
            epoch_end_alpha = start_alpha + (min_alpha_val - start_alpha) * ((epoch + 1) / epochs)
        else:
            epoch_start_alpha = start_alpha
            epoch_end_alpha = start_alpha
        
        # Total words expected for this epoch (for intra-epoch LR decay)
        # Use corpus_word_count which is the exact count of in-vocab tokens
        total_words_for_decay = self.corpus_word_count if decay_alpha else 0
        
        # Call the unified Cython function for the entire epoch
        epoch_loss, examples_processed_in_epoch, words_processed = word2vec_c.train_epoch(
            self.W,
            self.W_prime,
            sentences,
            self.vocab,
            sample_ints,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            epoch_start_alpha,
            epoch_end_alpha,
            total_words_for_decay,
            progress_callback if calculate_loss else None,
            callback_interval,
            self.sg,  # sg parameter selects Skip-gram vs CBOW
            calculate_loss,  # compute_loss parameter
        )
        
        total_examples_processed += examples_processed_in_epoch
        batch_count = progress_state['batch_count']
        
        # Close progress bar
        if progress_bar is not None:
            # Update to final word count
            if progress_bar.total is not None:
                remaining = max(0, progress_bar.total - progress_bar.n)
                if remaining > 0:
                    progress_bar.update(remaining)
            progress_bar.close()
        
        # Log epoch summary for simple logging mode
        if use_simple_logging and calculate_loss:
            recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            elapsed = time.time() - epoch_start_time
            ex_per_sec = examples_processed_in_epoch / elapsed if elapsed > 0 else 0
            print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} completed | examples={examples_processed_in_epoch:,} | words={words_processed:,} | avg_loss={recent_avg:.6f} | {ex_per_sec:.0f} ex/s")
        
        # Set current_alpha to the end-of-epoch value (epoch_end_alpha was computed above)
        # This ensures correct alpha is returned even when callback wasn't used
        current_alpha = epoch_end_alpha
        
        return epoch_loss, examples_processed_in_epoch, batch_count, total_examples_processed, current_alpha

    def get_vector(self, word: str, normalize: bool = False) -> Optional[np.ndarray]:
        """
        Get the vector for a word.
        
        Parameters
        ----------
        Args:
            word: Input word.
            normalize: If True, return the normalized vector (unit length).
        
        Returns:
            Word vector.
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
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
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
        
        Args:
            path: Path to save the model.
        """
        model_data = {
            'vocab': self.vocab,
            'index2word': self.index2word,
            'word_counts': dict(self.word_counts),
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
        
        model = cls(
            vector_size=model_data['vector_size'],
            window=model_data['window'],
            min_word_count=model_data.get('min_word_count', model_data.get('min_count', 5)),  # Backward compatibility
            negative=model_data['negative'],
            ns_exponent=model_data['ns_exponent'],
            cbow_mean=model_data['cbow_mean'],
            sg=model_data['sg'],
            sample=sample,
            shrink_windows=shrink_windows,
            max_vocab_size=max_vocab_size,
        )
        
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        # Convert the word_counts back to a Counter
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
        return model


# Helper functions for TempRefWord2Vec
def sample_sentences_to_token_count(
    corpus: List[List[str]], 
    target_tokens: int,
    seed: Optional[int] = None
) -> List[List[str]]:
    """
    Samples sentences from a corpus until the target token count is reached.
    
    This function randomly selects sentences from the corpus until the total number
    of tokens reaches or slightly exceeds the target count. This is useful for balancing
    corpus sizes when comparing different time periods or domains.
    
    Args:
        corpus (List[List[str]]): A list of sentences, where each sentence is a list 
            of tokens.
        target_tokens (int): The target number of tokens to sample.
        seed (Optional[int]): Random seed for reproducibility. If None, uses global seed.
    
    Returns:
        List[List[str]]: A list of sampled sentences with token count close to 
            target_tokens.
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


def add_corpus_tags(
    corpora: List[List[List[str]]], 
    labels: List[str], 
    target_words: List[str]
) -> List[List[List[str]]]:
    """
    Add corpus-specific tags to target words in all corpora at once.
    
    Args:
        corpora: List of corpora (each corpus is list of tokenized sentences)
        labels: List of corpus labels
        target_words: List of words to tag
    
    Returns:
        List of processed corpora where target words have been tagged with their corpus label
    """
    processed_corpora = []
    target_words_set = set(target_words)
    
    for corpus, label in zip(corpora, labels):
        processed_corpus = []
        for sentence in corpus:
            processed_sentence = []
            for token in sentence:
                if token in target_words_set:
                    processed_sentence.append(f"{token}_{label}")
                else:
                    processed_sentence.append(token)
            processed_corpus.append(processed_sentence)
        processed_corpora.append(processed_corpus)
    
    return processed_corpora


class TempRefWord2Vec(Word2Vec):
    """
    Implementation of Word2Vec with Temporal Referencing (TR) for tracking semantic change.
    
    This class extends Word2Vec to implement temporal referencing, where target words
    are represented with time period indicators (e.g., "bread_1800" for period 1800s) when used
    as target words, but remain unchanged when used as context words.
    
    The class takes multiple corpora corresponding to different time periods and automatically
    creates temporal references for specified target words.
    
    Args:
        corpora (List[List[List[str]]]): List of corpora, each corpus is a list of sentences 
            for a time period.
        labels (List[str]): Labels for each corpus (e.g., time periods like "1800s", "1900s").
        targets (List[str]): List of target words to trace semantic change.
        balance (bool): Whether to balance the corpora to equal sizes (default: True).
        **kwargs: Arguments passed to Word2Vec parent class (vector_size, window, sg, etc.).
    
    Example:
        from qhchina.analytics.word2vec import TempRefWord2Vec
        
        # Corpora from different time periods
        corpus_1800s = [["bread", "baker", ...], ["food", "eat", ...], ...]
        corpus_1900s = [["bread", "supermarket", ...], ["food", "buy", ...], ...]
        
        # Initialize model (Note: only sg=1 is supported)
        model = TempRefWord2Vec(
        ...     corpora=[corpus_1800s, corpus_1900s],
        ...     labels=["1800s", "1900s"],
        ...     targets=["bread", "food", "money"],
        ...     vector_size=100,
        ...     window=5,
        ...     sg=1  # Skip-gram required
        ... )
        
        # Train (uses preprocessed internal corpus)
        model.train()
        
        # Analyze semantic change
        model.most_similar("bread_1800s")  # Words similar to "bread" in the 1800s
        model.most_similar("bread_1900s")  # Words similar to "bread" in the 1900s
    """
    
    def __init__(
        self,
        corpora: List[List[List[str]]],  # List of corpora, each corpus is a list of sentences
        labels: List[str],               # Labels for each corpus (e.g., time periods)
        targets: List[str],              # Target words to trace semantic change
        balance: bool = True,            # Whether to balance the corpora
        **kwargs                         # Parameters passed to Word2Vec parent class
    ):
        # Check that corpora and labels have the same length
        if len(corpora) != len(labels):
            raise ValueError(f"Number of corpora ({len(corpora)}) must match number of labels ({len(labels)})")
    
        # check if sg = 1, else raise NotImplementedError
        if kwargs.get('sg') != 1:
            raise NotImplementedError("TempRefWord2Vec only supports Skip-gram model (sg=1)")
        
        # Store labels and targets as instance variables
        self.labels = labels
        self.targets = targets
        
        # Extract verbose from kwargs for logging before super().__init__
        verbose = kwargs.get('verbose', False)
        
        # print how many sentences in each corpus (each corpus a list of sentences)
        # and total size of each corpus (how many words; each sentence a list of words)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if verbose and not all(len(corpus) == 0 for corpus in corpora):
            for i, corpus in enumerate(corpora):
                logger.info(f"Corpus {labels[i]} has {len(corpus)} sentences and {sum(len(sentence) for sentence in corpus)} words")

        # Calculate token counts and determine minimum
        if balance:
            corpus_token_counts = [sum(len(sentence) for sentence in corpus) for corpus in corpora]
            target_token_count = min(corpus_token_counts)
            if verbose:
                logger.info(f"Balancing corpora to minimum size: {target_token_count} tokens")
            
            # Balance corpus sizes
            balanced_corpora = []
            for i, corpus in enumerate(corpora):
                if corpus_token_counts[i] <= target_token_count:
                    balanced_corpora.append(corpus)
                else:
                    sampled_corpus = sample_sentences_to_token_count(corpus, target_token_count)
                    balanced_corpora.append(sampled_corpus)
        
            # Add corpus tags to the corpora
            tagged_corpora = add_corpus_tags(balanced_corpora, labels, targets)
        else:
            tagged_corpora = add_corpus_tags(corpora, labels, targets)

        # Initialize combined corpus before using it
        self.combined_corpus = []
        
        # Calculate vocab counts for each period before combining
        from collections import Counter
        self.period_vocab_counts = {}
        
        for i, (corpus, label) in enumerate(zip(tagged_corpora, labels)):
            period_counter = Counter()
            for sentence in corpus:
                period_counter.update(sentence)
            self.period_vocab_counts[label] = period_counter
            # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
            if verbose and not all(len(corpus) == 0 for corpus in corpora):
                logger.info(f"Period '{label}': {len(period_counter)} unique tokens, {sum(period_counter.values())} total tokens")
        
        # Combine all tagged corpora
        for corpus in tagged_corpora:
            self.combined_corpus.extend(corpus)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if verbose and not all(len(corpus) == 0 for corpus in corpora):
            logger.info(f"Combined corpus: {len(self.combined_corpus)} sentences, {sum(len(s) for s in self.combined_corpus)} tokens")
        
        # clear memory
        del tagged_corpora
        if balance:
            del balanced_corpora
            if 'sampled_corpus' in locals():
                del sampled_corpus
        
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
        
        # Initialize parent Word2Vec class with kwargs
        super().__init__(**kwargs)
    
    def build_vocab(self, sentences: List[List[str]]) -> None:
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
            # Update vocabulary size
            self.vocab_size = len(self.vocab)
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
    
    def _train_epoch_cython(
        self,
        sentences: List[List[str]],
        epoch: int,
        epochs: int,
        batch_size: int,
        decay_alpha: bool,
        total_example_count: int,
        examples_per_epoch: Optional[int],
        total_examples_processed: int,
        current_alpha: float,
        start_alpha: float,
        calculate_loss: bool,
        recent_losses: List[float],
        epoch_start_time: float,
        verbose: Optional[int] = None,
    ) -> Tuple[float, int, int, int, float]:
        """
        Train for one epoch using fully optimized Cython with temporal mapping.
        
        The entire epoch is processed in a SINGLE Cython call with temporal
        index mapping for context word to base form conversion.
        """
        def current_time_str() -> str:
            return time.strftime("%H:%M:%S")
        
        num_sentences = len(sentences)
        
        # Ensure temporal index map is built
        if not hasattr(self, 'temporal_index_map') or self.temporal_index_map is None:
            self._build_temporal_index_map()
        
        # Compute sample_ints for subsampling
        sample_ints = self._compute_sample_ints()
        
        # Progress state
        progress_state = {
            'last_words': 0,
            'last_examples': 0,
            'last_time': epoch_start_time,
            'recent_losses': recent_losses,
            'batch_count': 0,
        }
        
        # Progress bar setup - use words processed (reliable) instead of estimated examples
        use_simple_logging = not decay_alpha and examples_per_epoch is None
        progress_bar = None
        
        if calculate_loss and not use_simple_logging:
            bar_format = '{l_bar}{bar}| {percentage:.1f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            
            progress_bar = tqdm(
                desc=f"Epoch {epoch+1}/{epochs}",
                total=self.corpus_word_count,  # Use known word count
                bar_format=bar_format,
                unit=" tokens",  # Space prefix for "133k tokens/s" formatting
                unit_scale=True,
                mininterval=0.5
            )
        
        LOSS_HISTORY_SIZE = 100
        
        def progress_callback(words_processed: int, examples_processed: int, 
                              running_loss: float, current_lr: float):
            nonlocal current_alpha
            
            progress_state['batch_count'] += 1
            batch_count = progress_state['batch_count']
            
            batch_words = words_processed - progress_state['last_words']
            batch_examples = examples_processed - progress_state['last_examples']
            if batch_examples > 0 and running_loss > 0:
                batch_avg_loss = running_loss / examples_processed if examples_processed > 0 else 0.0
                progress_state['recent_losses'].append(batch_avg_loss)
                if len(progress_state['recent_losses']) > LOSS_HISTORY_SIZE:
                    progress_state['recent_losses'].pop(0)
            
            progress_state['last_words'] = words_processed
            progress_state['last_examples'] = examples_processed
            current_alpha = current_lr
            
            if not calculate_loss:
                return
            
            recent_avg = sum(progress_state['recent_losses']) / len(progress_state['recent_losses']) if progress_state['recent_losses'] else 0.0
            
            if use_simple_logging:
                if verbose is not None and batch_count % max(1, verbose) == 0:
                    elapsed = time.time() - epoch_start_time
                    ex_per_sec = examples_processed / elapsed if elapsed > 0 else 0
                    if decay_alpha:
                        print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} | words={words_processed:,} | loss={recent_avg:.6f} | lr={current_lr:.6f} | {ex_per_sec:.0f} ex/s")
                    else:
                        print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} | words={words_processed:,} | loss={recent_avg:.6f} | {ex_per_sec:.0f} ex/s")
            elif progress_bar is not None:
                if decay_alpha:
                    postfix_str = f"loss={recent_avg:.6f}, lr={current_lr:.6f}"
                else:
                    postfix_str = f"loss={recent_avg:.6f}"
                progress_bar.set_postfix_str(postfix_str)
                progress_bar.update(batch_words)  # Update by words processed
        
        callback_interval = max(100, min(5000, num_sentences // 50))
        
        # Compute per-epoch start and end alpha for smooth linear interpolation
        # Alpha decays linearly from start_alpha to min_alpha across all epochs
        min_alpha_val = self.min_alpha if self.min_alpha else start_alpha
        if decay_alpha and epochs > 0:
            # epoch_start_alpha: learning rate at the START of this epoch
            # epoch_end_alpha: learning rate at the END of this epoch
            epoch_start_alpha = start_alpha + (min_alpha_val - start_alpha) * (epoch / epochs)
            epoch_end_alpha = start_alpha + (min_alpha_val - start_alpha) * ((epoch + 1) / epochs)
        else:
            epoch_start_alpha = start_alpha
            epoch_end_alpha = start_alpha
        
        # Total words expected for this epoch (for intra-epoch LR decay)
        # Use corpus_word_count which is the exact count of in-vocab tokens
        total_words_for_decay = self.corpus_word_count if decay_alpha else 0
        
        # Call the unified Cython function for temporal training
        epoch_loss, examples_processed_in_epoch, words_processed = word2vec_c.train_epoch_temporal(
            self.W,
            self.W_prime,
            sentences,
            self.vocab,
            self.temporal_index_map,
            sample_ints,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            epoch_start_alpha,
            epoch_end_alpha,
            total_words_for_decay,
            progress_callback if calculate_loss else None,
            callback_interval,
            self.sg,  # sg parameter selects Skip-gram vs CBOW
            calculate_loss,  # compute_loss parameter
        )
        
        total_examples_processed += examples_processed_in_epoch
        batch_count = progress_state['batch_count']
        
        # Close progress bar
        if progress_bar is not None:
            # Update to final word count
            if progress_bar.total is not None:
                remaining = max(0, progress_bar.total - progress_bar.n)
                if remaining > 0:
                    progress_bar.update(remaining)
            progress_bar.close()
        
        # Log epoch summary
        if use_simple_logging and calculate_loss:
            recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            elapsed = time.time() - epoch_start_time
            ex_per_sec = examples_processed_in_epoch / elapsed if elapsed > 0 else 0
            print(f"[{current_time_str()}] Epoch {epoch+1}/{epochs} completed | examples={examples_processed_in_epoch:,} | words={words_processed:,} | avg_loss={recent_avg:.6f} | {ex_per_sec:.0f} ex/s")
        
        return epoch_loss, examples_processed_in_epoch, batch_count, total_examples_processed, current_alpha
    
    def train(self, sentences: Optional[List[str]] = None) -> Optional[float]:
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

    def calculate_semantic_change(self, target_word: str, labels: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
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

    def get_available_targets(self) -> List[str]:
        """
        Get the list of target words available for semantic change analysis.
        
        Returns
        -------
        List of target words that were specified during model initialization
        """
        return self.targets.copy()

    def get_time_labels(self) -> List[str]:
        """
        Get the list of time period labels used in the model.
        
        Returns
        -------
        List of time period labels that were specified during model initialization
        """
        return self.labels.copy()
    
    def get_period_vocab_counts(self, period: Optional[str] = None) -> Union[Dict[str, Counter], Counter]:
        """
        Get vocabulary counts for a specific period or all periods.
        
        Parameters
        ----------
        period : str, optional
            The period label to get vocab counts for. If None, returns all periods.
            
        Returns
        -------
        Union[Dict[str, Counter], Counter]
            If period is None: dictionary mapping period labels to Counter objects
            If period is specified: Counter object for that specific period
            
        Raises
        -------
        ValueError
            If the specified period is not found in the model
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
        
        # Create a dummy corpus for initialization (we don't save the actual corpus)
        # The model will be fully restored from saved vectors and vocab
        dummy_corpora = [[] for _ in labels]
        
        # Get base model parameters with defaults
        shrink_windows = model_data.get('shrink_windows', False)
        sample = model_data.get('sample', 1e-3)
        max_vocab_size = model_data.get('max_vocab_size', None)
        
        # Create model instance with dummy data (will be overwritten)
        model = cls(
            corpora=dummy_corpora,
            labels=labels,
            targets=targets,
            vector_size=model_data['vector_size'],
            window=model_data['window'],
            min_word_count=model_data.get('min_word_count', model_data.get('min_count', 5)),  # Backward compatibility
            negative=model_data['negative'],
            ns_exponent=model_data['ns_exponent'],
            cbow_mean=model_data['cbow_mean'],
            sg=model_data['sg'],
            sample=sample,
            shrink_windows=shrink_windows,
            max_vocab_size=max_vocab_size,
            balance=False  # Don't balance dummy corpora
        )
        
        # Restore saved model state
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
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

