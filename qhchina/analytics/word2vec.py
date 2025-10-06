"""
Sample-based Word2Vec implementation
-----------------------------------
This module implements the Word2Vec algorithm with both CBOW and Skip-gram models using
explicit samples represented as different types of training examples:

- Skip-gram (sg=1): Each training example is a tuple (input_idx, output_idx), where 
  input_idx is the index of the center word and output_idx is the index of a context word.
  Negative examples are generated from the noise distribution for each positive example.

- CBOW (sg=0): Each training example is a tuple (input_indices, output_idx), where
  input_indices are the indices of context words, and output_idx is the index of the center word.
  Negative examples are generated from the noise distribution for each positive example.

Features:
- CBOW and Skip-gram architectures with appropriate example generation
- Training with individual examples (one by one)
- Explicit negative sampling for each training example
- Subsampling of frequent words
- Dynamic window sizing with shrink_windows parameter
- Properly managed learning rate decay
- Sigmoid precomputation for faster training
- Vocabulary size restriction with max_vocab_size parameter
- Optional Cython acceleration for faster training
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Union, Iterator, Generator, Any, Callable
import random
import math
from tqdm.auto import tqdm
import warnings
import time
from .vectors import cosine_similarity


class Word2Vec:
    """
    Implementation of Word2Vec algorithm with sample-based training approach.
    
    This class implements both Skip-gram and CBOW architectures:
    - Skip-gram (sg=1): Each training example is (input_idx, output_idx) where input is the center word
      and output is a context word.
    - CBOW (sg=0): Each training example is (input_indices, output_idx) where inputs are context words
      and output is the center word.
    
    Training is performed one example at a time, with negative examples generated for each positive example.
    
    Features:
    - CBOW and Skip-gram architectures with appropriate example generation
    - Training with individual examples (one by one)
    - Explicit negative sampling for each training example
    - Subsampling of frequent words
    - Dynamic window sizing with shrink_windows parameter
    - Properly managed learning rate decay
    - Sigmoid precomputation for faster training
    - Vocabulary size restriction with max_vocab_size parameter
    - Optional double precision for numerical stability
    - Optional Cython acceleration for significantly faster training
    
    Performance options:
    - Use double precision (use_double_precision=True) for better numerical stability (slightly slower)
    - Use Cython acceleration (use_cython=True) for much faster training (requires Cython extension)
    """
    
    def __init__(
        self,
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
        exp_table_size: int = 1000,
        max_exp: float = 6.0,
        max_vocab_size: Optional[int] = None,
        use_double_precision: bool = False,
        use_cython: bool = True,
        gradient_clip: float = 1.0,
    ):
        """
        Initialize the Word2Vec model.
        
        Parameters:
        -----------
        vector_size: Dimensionality of the word vectors
        window: Maximum distance between the current and predicted word
        min_word_count: Ignores all words with frequency lower than this
        negative: Number of negative samples for negative sampling
        ns_exponent: Exponent used to shape the negative sampling distribution
        cbow_mean: If True, use mean of context word vectors, else use sum
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        seed: Seed for random number generator
        alpha: Initial learning rate
        min_alpha: Minimum learning rate. If None, learning rate remains constant at alpha.
        sample: Threshold for subsampling frequent words. Default is 1e-3, set to 0 to disable.
        shrink_windows: If True, the effective window size is uniformly sampled from [1, window] 
                        for each target word during training. If False, always use the full window.
        exp_table_size: Size of sigmoid lookup table for precomputation
        max_exp: Range of values for sigmoid precomputation [-max_exp, max_exp]
        max_vocab_size: Maximum vocabulary size to keep, keeping the most frequent words.
                        None means no limit (keep all words above min_word_count).
        use_double_precision: Whether to use float64 precision for better stability. Default is False.
        use_cython: Whether to use Cython acceleration if available. Default is True.
        gradient_clip: Maximum absolute value for gradients when using Cython. Default is 1.0.
        """
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
        self.use_double_precision = use_double_precision  # Whether to use double precision
        self.gradient_clip = gradient_clip  # For gradient clipping in Cython
        
        # Initialize use_cython to False by default
        self.word2vec_c = None
        self.use_cython = use_cython  # Save use_cython parameter as an attribute

        # Try to import Cython extension if requested
        if use_cython:
            self._attempt_cython_import()
        
        # Set the dtype based on precision choice
        self.dtype = np.float64 if self.use_double_precision else np.float32
        
        # Parameters for sigmoid precomputation
        self.exp_table_size = exp_table_size
        self.max_exp = max_exp
        
        # Precompute the sigmoid table and log sigmoid table
        self._precompute_sigmoid()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize vocabulary structures
        self.vocab = {}  # word -> index (direct mapping)
        self.index2word = []  # index -> word
        self.word_counts = Counter()  # word -> count
        self.corpus_word_count = 0
        self.discard_probs = {}  # For subsampling frequent words
        
        # These will be initialized in _initialize_weights
        self.W = None  # Input word embeddings
        self.W_prime = None  # Output word embeddings (for negative sampling)
        self.noise_distribution = None  # For negative sampling
        
        # For tracking training progress
        self.epoch_losses = []
        self.total_examples = 0

    def _attempt_cython_import(self) -> bool:
        """
        Attempt to import the Cython-optimized module.
        
        Returns:
        --------
        bool: True if import was successful, False otherwise
        """
        try:
            # Attempt to import the Cython module
            from .cython_ext import word2vec
            self.word2vec_c = word2vec
            self.use_cython = True
            return True
        except ImportError as e:
            self.use_cython = False
            warnings.warn(
                f"Cython acceleration for Word2Vec was requested but the extension "
                f"is not available in the current environment. Falling back to Python implementation, "
                f"which will be significantly slower.\n"
                f"Error: {e}"
            )
            return False
        
    def _precompute_sigmoid(self) -> None:
        """
        Precompute sigmoid values for faster training.
        
        This method creates a lookup table for sigmoid(x) and log(sigmoid(x))
        for x values from -max_exp to +max_exp, discretized into exp_table_size bins.
        """
        self.sigmoid_table = np.zeros(self.exp_table_size, dtype=self.dtype)
        self.log_sigmoid_table = np.zeros(self.exp_table_size, dtype=self.dtype)

        for i in range(self.exp_table_size):
            # Calculate x value in range [-max_exp, max_exp]
            x = (i / self.exp_table_size * 2 - 1) * self.max_exp
            # Compute sigmoid(x) = 1 / (1 + exp(-x))
            self.sigmoid_table[i] = self.dtype(1.0 / (1.0 + np.exp(-x)))
            # Compute log(sigmoid(x))
            self.log_sigmoid_table[i] = np.log(self.sigmoid_table[i], dtype=self.dtype)
        
        self.sigmoid_scale = self.dtype(self.exp_table_size / (2 * self.max_exp))  # Scale factor
        self.sigmoid_offset = self.dtype(self.exp_table_size // 2)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Get sigmoid values using the precomputed table.
        
        Parameters:
        -----------
        x: Input values
        
        Returns:
        --------
        Sigmoid values for the inputs
        """
        # Fast conversion of inputs to indices
        # Formula: idx = (x * scale + offset) 
        # This maps [-max_exp, max_exp] -> [0, exp_table_size-1]
        idx = np.rint(x * self.sigmoid_scale + self.sigmoid_offset).astype(np.int32)
        
        # Fast correction of out-of-bounds indices without np.clip
        # We use np.maximum and np.minimum which are faster than clip for simple bounds
        # First ensure idx >= 0, then ensure idx < exp_table_size
        idx = np.maximum(0, np.minimum(self.exp_table_size - 1, idx))
        
        # Look up values in the table
        return self.sigmoid_table[idx]
    
    def _log_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Get log sigmoid values using the precomputed table.
        
        Parameters:
        -----------
        x: Input values
        
        Returns:
        --------
        Log sigmoid values for the inputs
        """
        # Reuse the exact same fast index calculation from _sigmoid
        idx = np.rint(x * self.sigmoid_scale + self.sigmoid_offset).astype(np.int32)
        idx = np.maximum(0, np.minimum(self.exp_table_size - 1, idx))
        
        # Look up values in the table
        return self.log_sigmoid_table[idx]

    def build_vocab(self, sentences: Union[List[List[str]], Iterator[List[str]]]) -> None:
        """
        Build vocabulary from a list or iterator of sentences.
        
        Parameters:
        -----------
        sentences: List or iterator of tokenized sentences (each sentence is a list of words)
        
        Returns:
        --------
        None
        """
        
        # Count word occurrences - works with both lists and iterators
        for sentence in sentences:
            self.word_counts.update(sentence)
        
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
        
        self.corpus_word_count = sum(self.word_counts[word] for word in self.vocab)
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size} words")

    def _calculate_discard_probs(self) -> None:
        """
        Calculate the probability of discarding frequent words during subsampling.
        
        Formula from the original word2vec paper:
        P(w_i) = 1 - sqrt(t/f(w_i)) where t is the sample threshold
        and f(w_i) is the word frequency normalized by the total corpus word count.
        
        A word will be discarded with probability P(w_i).
        """
        
        self.discard_probs = {}
        total_words = self.corpus_word_count
        
        for word in self.vocab:
            # Calculate normalized word frequency
            word_freq = self.word_counts[word] / total_words
            # Calculate probability of discarding the word
            # Using the formula from the original word2vec paper
            discard_prob = 1.0 - np.sqrt(self.sample / word_freq)
            # Clamp the probability to [0, 1]
            discard_prob = max(0, min(1, discard_prob))
            self.discard_probs[word] = discard_prob

    def _initialize_vectors(self) -> None:
        """
        Initialize word vectors
        """
        vocab_size = len(self.vocab)
        
        # Initialize input and output matrices
        # Using Xavier/Glorot initialization for better convergence
        # Range is [-0.5/dim, 0.5/dim]
        bound = 0.5 / self.vector_size
        self.W = np.random.uniform(
            low=-bound, 
            high=bound, 
            size=(vocab_size, self.vector_size)
        ).astype(self.dtype)
        
        # Initialize W_prime with small random values like W instead of zeros
        # This helps improve convergence during training
        self.W_prime = np.random.uniform(
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
        
        if not self.use_cython:
            return

        # Validate arrays exist and are contiguous
        for name, arr in [
            ('sigmoid_table', self.sigmoid_table),
            ('log_sigmoid_table', self.log_sigmoid_table),
            ('noise_distribution', self.noise_distribution)
        ]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} must be a numpy array")
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=self.dtype)
        
        # Initialize Cython globals
        try:
            self.word2vec_c.init_globals(
                sigmoid_table=self.sigmoid_table,
                log_sigmoid_table=self.log_sigmoid_table,
                max_exp=self.max_exp,
                noise_distribution=self.noise_distribution,
                vector_size=self.vector_size,
                gradient_clip=self.gradient_clip,
                negative=self.negative,
                learning_rate=self.alpha,
                cbow_mean=int(self.cbow_mean),
                use_double_precision=self.use_double_precision,
            )
        except Exception as e:
            self.use_cython = False
            raise RuntimeError(f"Cython initialization failed: {e}") from e
        
    def _update_learning_rate(self, alpha: float) -> None:
        """
        Update the learning rate in the Cython extension when using Cython acceleration.
        
        Parameters:
        -----------
        alpha: New learning rate value
        """
        if self.use_cython:
            self.word2vec_c.update_learning_rate(alpha)
        self.alpha = alpha

    def generate_skipgram_examples(self, 
                                 sentences: Union[List[List[str]], Iterator[List[str]]]) -> Generator[Tuple[int, int], None, None]:
        """
        Generate Skip-gram training examples from sentences.
        
        A Skip-gram example is a tuple (input_idx, output_idx) where:
        - input_idx is the index of the center word
        - output_idx is the index of a context word
        
        For each positive example, the caller should generate negative examples using the noise distribution.
        
        Parameters:
        -----------
        sentences: List or iterator of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_idx, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Pre-compute all random numbers needed for this sentence
            # 1. Pre-generate window sizes for each position
            if self.shrink_windows:
                window_sizes = np.random.randint(1, self.window + 1, size=sentence_len)
            else:
                window_sizes = np.full(sentence_len, self.window)
                
            # 2. Pre-generate subsampling random values for all words
            if self.sample > 0:
                random_values = np.random.random(sentence_len)
            
            # 3. Pre-compute all word indices (-1 for words not in vocabulary)
            word_indices = [self.vocab.get(word, -1) for word in sentence]
            
            # Process each position in the sentence
            for pos in range(sentence_len):
                center_idx = word_indices[pos]
                
                # Skip if word is not in vocabulary
                if center_idx == -1:
                    continue
                
                # Apply subsampling to center word
                if self.sample > 0 and random_values[pos] < self.discard_probs[sentence[pos]]:
                    continue
                    
                # Determine window boundaries on the original sentence
                dynamic_window = window_sizes[pos]
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # For each potential context position within the window
                for context_pos in range(start, end):
                    # Skip the center word itself
                    if context_pos == pos:
                        continue
                    
                    context_idx = word_indices[context_pos]
                    
                    # Skip if context word is not in vocabulary
                    if context_idx == -1:
                        continue
                        
                    # Apply subsampling to context word
                    if self.sample > 0 and random_values[context_pos] < self.discard_probs[sentence[context_pos]]:
                        continue
                    
                    # Both words passed filters, yield the example
                    yield (center_idx, context_idx)

    def generate_cbow_examples(self, 
                             sentences: Union[List[List[str]], Iterator[List[str]]]) -> Generator[Tuple[List[int], int], None, None]:
        """
        Generate CBOW training examples from sentences.
        
        A CBOW example is a tuple (input_indices, output_idx) where:
        - input_indices is a list of indices of context words
        - output_idx is the index of the center word
        
        For each positive example, the caller should generate negative examples using the noise distribution.
        
        Parameters:
        -----------
        sentences: List or iterator of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_indices, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Pre-compute all random numbers needed for this sentence
            # 1. Pre-generate window sizes for each position
            if self.shrink_windows:
                window_sizes = np.random.randint(1, self.window + 1, size=sentence_len)
            else:
                window_sizes = np.full(sentence_len, self.window)
                
            # 2. Pre-generate subsampling random values for all words
            if self.sample > 0:
                random_values = np.random.random(sentence_len)
            
            # 3. Pre-compute all word indices (-1 for words not in vocabulary)
            word_indices = [self.vocab.get(word, -1) for word in sentence]
            
            # Process each position in the sentence
            for pos in range(sentence_len):
                center_idx = word_indices[pos]
                
                # Skip if center word is not in vocabulary
                if center_idx == -1:
                    continue
                
                # Apply subsampling to center word
                if self.sample > 0 and random_values[pos] < self.discard_probs[sentence[pos]]:
                    continue
                    
                # Determine window boundaries on the original sentence
                dynamic_window = window_sizes[pos]
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # Collect context indices with vocabulary and subsampling filters
                context_indices = []
                for context_pos in range(start, end):
                    # Skip the center word itself
                    if context_pos == pos:
                        continue
                    
                    context_idx = word_indices[context_pos]
                    
                    # Skip if context word is not in vocabulary
                    if context_idx == -1:
                        continue
                        
                    # Apply subsampling to context word
                    if self.sample > 0 and random_values[context_pos] < self.discard_probs[sentence[context_pos]]:
                        continue
                    
                    # Word passed all filters, add to context
                    context_indices.append(context_idx)
                
                # Only yield examples if we have at least one context word
                if context_indices:
                    yield (context_indices, center_idx)

    def _train_skipgram_example_python(self, input_idx: int, output_idx: int, learning_rate: float) -> float:
        """
        Train the model on a single Skip-gram example using pure Python implementation.
        
        Parameters:
        -----------
        input_idx: Index of the input word (center word)
        output_idx: Index of the output word (context word)
        learning_rate: Current learning rate
        
        Returns:
        --------
        Loss for this training example
        """
        # Get input and output vectors
        input_vector = self.W[input_idx]  # center word vector
        output_vector = self.W_prime[output_idx]  # context word vector
        
        # Compute dot product
        score = np.dot(input_vector, output_vector)
        
        # Apply sigmoid using the precomputed table
        prediction = self._sigmoid(score)
        
        # Calculate gradient for positive example
        gradient = prediction - 1.0  # gradient: sigmoid(x) - 1 (target)
        
        # Apply gradients to input and output vectors
        input_gradient = gradient * output_vector
        output_gradient = gradient * input_vector
        
        # Update weights
        self.W[input_idx] -= learning_rate * input_gradient
        self.W_prime[output_idx] -= learning_rate * output_gradient
        
        # Loss for positive example: -log(sigmoid(score))
        # Use precomputed log sigmoid table
        loss_pos = -self._log_sigmoid(score)
        
        # Sample negative examples
        neg_indices = np.random.choice(
            self.vocab_size, 
            size=self.negative, 
            p=self.noise_distribution,
            replace=True
        )
        
        # Filter out the target word
        neg_indices = neg_indices[neg_indices != output_idx]

        # Vectorized training on negative examples
        # Get all negative output vectors at once
        neg_output_vectors = self.W_prime[neg_indices]  # shape: (n_negative, vector_size)
        
        # Compute all negative scores at once using dot product
        neg_scores = np.dot(neg_output_vectors, input_vector)  # shape: (n_negative,)
        
        # Apply sigmoid to all scores at once using precomputed table
        neg_predictions = self._sigmoid(neg_scores)  # shape: (n_negative,)
        
        # Calculate gradients for all negative examples at once
        # Target for negative examples is 0
        neg_gradients = neg_predictions  # shape: (n_negative,)
        
        # Prepare for broadcasting: reshape gradients to (n_negative, 1)
        neg_gradients_reshaped = neg_gradients.reshape(-1, 1)  # shape: (n_negative, 1)
        
        # Compute gradients for all negative examples at once
        # broadcast input_vector to (n_negative, vector_size)
        neg_output_gradients = neg_gradients_reshaped * input_vector  # shape: (n_negative, vector_size)
        
        # Fully vectorized update of all negative output vectors at once
        # For each negative index, update its corresponding row in W_prime
        np.add.at(self.W_prime, neg_indices, -learning_rate * neg_output_gradients)
        
        # Compute gradient for input vector: sum of gradients from all negative examples
        # This step can be fully vectorized
        neg_input_gradient = np.sum(neg_gradients_reshaped * neg_output_vectors, axis=0)
        
        # Update input vector once with accumulated gradient
        self.W[input_idx] -= learning_rate * neg_input_gradient
        
        # Calculate loss for all negative examples at once: -sum(log(1 - sigmoid(score)))
        # Use precomputed log sigmoid for more efficient calculation
        loss_neg = -np.sum(self._log_sigmoid(-neg_scores))
        
        # Total loss
        total_loss = loss_pos + loss_neg
        
        return total_loss

    def _train_cbow_example_python(self, input_indices: List[int], output_idx: int, learning_rate: float) -> float:
        """
        Train the model on a single CBOW example using pure Python implementation.
        
        Parameters:
        -----------
        input_indices: List of indices for input context words
        output_idx: Index of the output word (center word)
        learning_rate: Current learning rate
        
        Returns:
        --------
        Loss for this training example
        """
        # Get input vectors (context word vectors)
        input_vectors = self.W[input_indices]  # shape: (n_context, vector_size)
        
        # Combine context vectors: mean if cbow_mean=True, else sum
        if self.cbow_mean and len(input_indices) > 1:
            combined_input = np.mean(input_vectors, axis=0)  # average
        else:
            combined_input = np.sum(input_vectors, axis=0)  # sum
            
        # Get output vector (center word)
        output_vector = self.W_prime[output_idx]
        
        # Compute dot product
        score = np.dot(combined_input, output_vector)
        
        # Apply sigmoid using precomputed table
        prediction = self._sigmoid(score)
        
        # Calculate gradient for positive example
        gradient = prediction - 1.0  # gradient: sigmoid(x) - 1 (target)
        
        # Apply gradients
        output_gradient = gradient * combined_input
        
        # For inputs, distribute the gradient
        input_gradient = gradient * output_vector
        if self.cbow_mean and len(input_indices) > 1:
            input_gradient = input_gradient / len(input_indices)  # normalize by context size
        
        # Update weights
        self.W_prime[output_idx] -= learning_rate * output_gradient
        
        # Vectorized update of all input vectors at once
        np.add.at(self.W, input_indices, -learning_rate * input_gradient)
        
        # Loss for positive example: -log(sigmoid(score))
        # Use precomputed log sigmoid
        loss_pos = -self._log_sigmoid(score)
        
        # Sample negative examples
        neg_indices = np.random.choice(
            self.vocab_size, 
            size=self.negative, 
            p=self.noise_distribution,
            replace=True
        )
        
        # Filter out the target word
        neg_indices = neg_indices[neg_indices != output_idx]
        
        # Vectorized training on negative examples
        # Get all negative output vectors at once
        neg_output_vectors = self.W_prime[neg_indices]  # shape: (n_negative, vector_size)
        
        # Compute all negative scores at once using dot product
        neg_scores = np.dot(neg_output_vectors, combined_input)  # shape: (n_negative,)
        
        # Apply sigmoid to all scores at once using precomputed table
        neg_predictions = self._sigmoid(neg_scores)  # shape: (n_negative,)
        
        # Calculate gradients for all negative examples at once
        # Target for negative examples is 0
        neg_gradients = neg_predictions  # shape: (n_negative,)
        
        # Prepare for broadcasting: reshape gradients to (n_negative, 1)
        neg_gradients_reshaped = neg_gradients.reshape(-1, 1)  # shape: (n_negative, 1)
        
        # Compute gradients for all negative output vectors at once
        neg_output_gradients = neg_gradients_reshaped * combined_input  # shape: (n_negative, vector_size)
        
        # Fully vectorized update of all negative output vectors at once
        # For each negative index, update its corresponding row in W_prime
        np.add.at(self.W_prime, neg_indices, -learning_rate * neg_output_gradients)
        
        # Compute gradients for input vectors
        # This is the sum of gradients from all negative examples
        # This step can be fully vectorized
        neg_input_gradient = np.sum(neg_gradients_reshaped * neg_output_vectors, axis=0)
            
        # Normalize if using mean
        if self.cbow_mean and len(input_indices) > 1:
            neg_input_gradient = neg_input_gradient / len(input_indices)
            
        # Vectorized update of all input vectors at once with accumulated gradient
        np.add.at(self.W, input_indices, -learning_rate * neg_input_gradient)
        
        # Calculate loss for all negative examples at once: -sum(log(1 - sigmoid(score)))
        # Use precomputed log sigmoid for efficiency
        loss_neg = -np.sum(self._log_sigmoid(-neg_scores))
        
        # Total loss
        total_loss = loss_pos + loss_neg
        
        return total_loss

    def _train_batch_python(self, samples: List[Tuple], learning_rate: float) -> float:
        """
        Train the model on a batch of samples in a fully vectorized manner using pure Python.
        
        Parameters:
        -----------
        samples: List of training samples:
                - for Skip-gram: list of (input_idx, output_idx) tuples
                - for CBOW: list of (input_indices, output_idx) tuples
        learning_rate: Current learning rate
        
        Returns:
        --------
        Total loss for the batch
        """
        batch_size = len(samples)
        
        if self.sg:  # Skip-gram mode
            # Extract input (center) and output (context) indices from samples
            input_indices = np.array([sample[0] for sample in samples])
            output_indices = np.array([sample[1] for sample in samples])
            
            # === POSITIVE EXAMPLES ===
            
            # Get all input vectors at once: shape (batch_size, vector_size)
            input_vectors = self.W[input_indices]
            
            # Get all output vectors at once: shape (batch_size, vector_size)
            output_vectors = self.W_prime[output_indices]
            
            # Compute scores for all positive examples at once: shape (batch_size,)
            scores = np.sum(input_vectors * output_vectors, axis=1)
            
            # Apply sigmoid to get predictions using precomputed table: shape (batch_size,)
            predictions = self._sigmoid(scores)
            
            # Calculate gradients for all positive examples at once: shape (batch_size,)
            # Target for positive examples is 1
            gradients = predictions - 1.0
            
            # Reshape gradients for broadcasting: (batch_size, 1)
            gradients_reshaped = gradients.reshape(-1, 1)
            
            # Compute all input gradients at once: shape (batch_size, vector_size)
            input_gradients = gradients_reshaped * output_vectors
            
            # Compute all output gradients at once: shape (batch_size, vector_size)
            output_gradients = gradients_reshaped * input_vectors
            
            # Update all input and output vectors at once using np.add.at
            # This handles the case where the same index appears multiple times
            np.add.at(self.W, input_indices, -learning_rate * input_gradients)
            np.add.at(self.W_prime, output_indices, -learning_rate * output_gradients)
            
            # Compute loss for all positive examples using precomputed log sigmoid: shape (batch_size,)
            loss_pos = -self._log_sigmoid(scores)
            total_pos_loss = np.sum(loss_pos)
            
            # === NEGATIVE EXAMPLES - FULLY VECTORIZED ===
            
            # Generate negative samples for the entire batch at once
            # Shape: (batch_size, self.negative)
            neg_indices_buffer = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.negative),
                p=self.noise_distribution,
                replace=True
            )
            
            # Conditional Replacement: replace any negative indices that match their corresponding output index
            output_indices_reshaped = output_indices.reshape(-1, 1)
            mask = (neg_indices_buffer == output_indices_reshaped)
            if np.any(mask):
                # Generate random replacement indices
                replacements = np.random.randint(0, self.vocab_size, size=np.sum(mask))
                neg_indices_buffer[mask] = replacements
            
            # Get all negative vectors: shape (batch_size, self.negative, vector_size)
            neg_vectors = self.W_prime[neg_indices_buffer]
            
            # Reshape input vectors for broadcasting with negative vectors
            # From (batch_size, vector_size) to (batch_size, 1, vector_size)
            input_vectors_reshaped = input_vectors.reshape(batch_size, 1, self.vector_size)
            
            # Compute scores for all negative examples at once using batch matmul
            # Shape: (batch_size, self.negative)
            neg_scores = np.sum(neg_vectors * input_vectors_reshaped, axis=2)
            
            # Apply sigmoid using precomputed table: shape (batch_size, self.negative)
            neg_predictions = self._sigmoid(neg_scores)
            
            # Calculate gradients for all negative examples (target is 0)
            # Shape: (batch_size, self.negative)
            neg_gradients = neg_predictions
            
            # Reshape for broadcasting: (batch_size, self.negative, 1)
            neg_gradients_reshaped = neg_gradients.reshape(batch_size, self.negative, 1)
            
            # Compute gradients for all negative vectors at once
            # Shape: (batch_size, self.negative, vector_size)
            neg_output_gradients = neg_gradients_reshaped * input_vectors_reshaped
            
            # Flatten negative indices and gradients for efficient update
            flat_neg_indices = neg_indices_buffer.reshape(-1)
            flat_neg_gradients = neg_output_gradients.reshape(-1, self.vector_size)
            
            # Update all negative vectors at once
            np.add.at(self.W_prime, flat_neg_indices, -learning_rate * flat_neg_gradients)
            
            # Compute gradients for input vectors from all negative examples
            # Sum across all negative samples for each input vector
            # Shape: (batch_size, vector_size)
            neg_input_gradients = np.sum(neg_gradients_reshaped * neg_vectors, axis=1)
            
            # Update all input vectors at once
            np.add.at(self.W, input_indices, -learning_rate * neg_input_gradients)
            
            # Calculate loss for all negative examples at once using precomputed log sigmoid
            # Shape: (batch_size,)
            neg_losses = -np.sum(self._log_sigmoid(-neg_scores), axis=1)
            total_neg_loss = np.sum(neg_losses)
            
            # Total loss is the sum of positive and negative losses
            total_loss = total_pos_loss + total_neg_loss
            
        else:  # CBOW mode
            # Extract context indices and center word indices
            context_indices_list = [sample[0] for sample in samples]
            center_indices = np.array([sample[1] for sample in samples])
            
            # === POSITIVE EXAMPLES ===
            
            # Process all positive examples
            # This part needs a loop due to variable-length context windows
            combined_inputs = np.zeros((batch_size, self.vector_size))
            context_sizes = np.zeros(batch_size, dtype=np.int32)
            
            for batch_idx in range(batch_size):
                context_indices = context_indices_list[batch_idx]
                context_vectors = self.W[context_indices]
                context_sizes[batch_idx] = len(context_indices)
                
                # Combine context vectors based on cbow_mean parameter
                if self.cbow_mean and len(context_indices) > 1:
                    combined_input = np.mean(context_vectors, axis=0)
                else:
                    combined_input = np.sum(context_vectors, axis=0)
                
                combined_inputs[batch_idx] = combined_input
            
            # Get center word vectors: shape (batch_size, vector_size)
            center_vectors = self.W_prime[center_indices]
            
            # Compute all scores at once: shape (batch_size,)
            scores = np.sum(combined_inputs * center_vectors, axis=1)
            
            # Apply sigmoid using precomputed table: shape (batch_size,)
            predictions = self._sigmoid(scores)
            
            # Calculate gradients for all positive examples (target is 1)
            # Shape: (batch_size,)
            gradients = predictions - 1.0
            
            # Reshape for broadcasting: (batch_size, 1)
            gradients_reshaped = gradients.reshape(-1, 1)
            
            # Compute gradients for center vectors: shape (batch_size, vector_size)
            center_gradients = gradients_reshaped * combined_inputs
            
            # Update all center vectors at once
            np.add.at(self.W_prime, center_indices, -learning_rate * center_gradients)
            
            # Compute loss for all positive examples using precomputed log sigmoid: shape (batch_size,)
            loss_pos = -self._log_sigmoid(scores)
            positive_loss = np.sum(loss_pos)
            
            # === NEGATIVE EXAMPLES - FULLY VECTORIZED ===
            
            # Generate negative samples for all batch items at once
            # Shape: (batch_size, self.negative)
            neg_indices_buffer = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.negative),
                p=self.noise_distribution,
                replace=True
            )
            
            # Conditional Replacement: replace any negative indices that match their corresponding center index
            center_indices_reshaped = center_indices.reshape(-1, 1)
            mask = (neg_indices_buffer == center_indices_reshaped)
            if np.any(mask):
                # Generate random replacement indices
                replacements = np.random.randint(0, self.vocab_size, size=np.sum(mask))
                neg_indices_buffer[mask] = replacements
            
            # Get all negative vectors at once: shape (batch_size, self.negative, vector_size)
            neg_vectors = self.W_prime[neg_indices_buffer]
            
            # Reshape combined inputs for broadcasting with negative vectors
            # From (batch_size, vector_size) to (batch_size, 1, vector_size)
            combined_inputs_reshaped = combined_inputs.reshape(batch_size, 1, self.vector_size)
            
            # Compute scores for all negative examples at once: shape (batch_size, self.negative)
            neg_scores = np.sum(neg_vectors * combined_inputs_reshaped, axis=2)
            
            # Apply sigmoid using precomputed table: shape (batch_size, self.negative)
            neg_predictions = self._sigmoid(neg_scores)
            
            # Calculate gradients for all negative examples (target is 0)
            # Shape: (batch_size, self.negative)
            neg_gradients = neg_predictions
            
            # Reshape for broadcasting: (batch_size, self.negative, 1)
            neg_gradients_reshaped = neg_gradients.reshape(batch_size, self.negative, 1)
            
            # Compute gradients for all negative vectors at once
            # Shape: (batch_size, self.negative, vector_size)
            neg_output_gradients = neg_gradients_reshaped * combined_inputs_reshaped
            
            # Flatten negative indices and gradients for efficient update
            flat_neg_indices = neg_indices_buffer.reshape(-1)
            flat_neg_gradients = neg_output_gradients.reshape(-1, self.vector_size)
            
            # Update all negative vectors at once
            np.add.at(self.W_prime, flat_neg_indices, -learning_rate * flat_neg_gradients)
            
            # Compute gradients for combined inputs from all negative examples
            # Shape: (batch_size, vector_size)
            neg_input_gradients = np.sum(neg_gradients_reshaped * neg_vectors, axis=1)
            
            # Calculate loss for all negative examples at once using precomputed log sigmoid: shape (batch_size,)
            neg_losses = -np.sum(self._log_sigmoid(-neg_scores), axis=1)
            negative_loss = np.sum(neg_losses)
            
            # Now update context vectors (this requires a loop due to variable context sizes)
            for batch_idx in range(batch_size):
                context_indices = context_indices_list[batch_idx]
                
                # Get gradients for context vectors from positive and negative examples
                pos_input_gradient = gradients[batch_idx] * center_vectors[batch_idx]
                neg_input_gradient = neg_input_gradients[batch_idx]
                
                # Combine gradients
                total_input_gradient = pos_input_gradient + neg_input_gradient
                
                # Normalize if using mean
                if self.cbow_mean and context_sizes[batch_idx] > 1:
                    total_input_gradient = total_input_gradient / context_sizes[batch_idx]
                
                # Update context vectors
                np.add.at(self.W, context_indices, -learning_rate * total_input_gradient)
            
            # Total loss
            total_loss = positive_loss + negative_loss
        
        return total_loss

    def _train_skipgram_example(self, input_idx: int, output_idx: int, learning_rate: float) -> float:
        """
        Train the model on a single Skip-gram example, using either Cython or Python implementation.
        
        Parameters:
        -----------
        input_idx: Index of the input word (center word)
        output_idx: Index of the output word (context word)
        learning_rate: Current learning rate
        
        Returns:
        --------
        Loss for this training example
        """
        if self.use_cython:
            # Call the Cython single example implementation directly
            return self.word2vec_c.train_skipgram_single(
                self.W,
                self.W_prime,
                input_idx,
                output_idx
            )
        else:
            # Use the Python implementation
            return self._train_skipgram_example_python(input_idx, output_idx, learning_rate)
    
    def _train_cbow_example(self, input_indices: List[int], output_idx: int, learning_rate: float) -> float:
        """
        Train a single CBOW example with negative sampling.
        
        Parameters:
        -----------
        input_indices: List of context word indices
        output_idx: Index of center word
        learning_rate: Current learning rate
        
        Returns:
        --------
        Loss for this training example
        """
        if self.use_cython:     
            # Convert input_indices to a numpy array for Cython
            context_indices = np.array(input_indices, dtype=np.int32)
            
            # Call the Cython single example implementation
            return self.word2vec_c.train_cbow_single(
                self.W,
                self.W_prime,
                context_indices,
                output_idx
            )
        else:
            # Use the Python implementation
            return self._train_cbow_example_python(input_indices, output_idx, learning_rate)
    
    def _train_batch(self, samples: List[Tuple], learning_rate: float) -> float:
        """
        Train the model on a batch of samples, using either Cython or Python implementation.
        
        Parameters:
        -----------
        samples: List of training samples:
                - for Skip-gram: list of (input_idx, output_idx) tuples
                - for CBOW: list of (input_indices, output_idx) tuples
        learning_rate: Current learning rate
        
        Returns:
        --------
        Total loss for the batch
        """
        if self.use_cython:
            # Process the batch differently based on the model type
            if self.sg:  # Skip-gram model
                # Extract input and output indices
                input_indices = np.array([sample[0] for sample in samples], dtype=np.int32)
                output_indices = np.array([sample[1] for sample in samples], dtype=np.int32)
                
                # Call the Cython implementation
                return self.word2vec_c.train_skipgram_batch(
                    self.W,
                    self.W_prime,
                    input_indices,
                    output_indices
                )
            else:  # CBOW model
                # Extract context indices and center indices
                context_indices_list = [sample[0] for sample in samples]
                center_indices = np.array([sample[1] for sample in samples], dtype=np.int32)
                
                # Call the Cython implementation
                return self.word2vec_c.train_cbow_batch(
                    self.W,
                    self.W_prime,
                    context_indices_list,
                    center_indices
                )
        else:
            # Use the Python implementation
            return self._train_batch_python(samples, learning_rate)

    def train(self, sentences: Union[List[List[str]], Iterator[List[str]]], 
              epochs: int = 1, 
              alpha: Optional[float] = None,
              min_alpha: Optional[float] = None,
              batch_size: int = 10000, 
              callbacks: List[Callable] = None,
              calculate_loss: bool = True) -> Optional[float]:
        """
        Train word2vec model on given sentences.
        
        Parameters:
        -----------
        sentences: List or iterator of tokenized sentences (lists of words)
        epochs: Number of training iterations over the corpus
        alpha: Initial learning rate
        min_alpha: Minimum allowed learning rate
        batch_size: Batch size for training; if 0, no batching is used; default is 10000;
        callbacks: List of callback functions to call after each epoch
        calculate_loss: Whether to calculate and return the final loss
        
        Returns:
        --------
        Final loss value if calculate_loss is True, None otherwise
        """
        self.epochs = epochs
        self.batch_size = batch_size
        if alpha:
            self.alpha = alpha
        if min_alpha:
            self.min_alpha = min_alpha
        if not self.alpha:
            print("Warning: No initial learning rate (alpha) provided. Using default value of 0.025 with no decay.")
            self.alpha = 0.025
            self.min_alpha = None

        # Determine if we should decay the learning rate based on min_alpha
        decay_alpha = self.min_alpha is not None

        # Convert iterator to list if we need multiple epochs (for shuffling)
        # or if we need to count examples for learning rate decay
        if epochs > 1 or decay_alpha:
            if not isinstance(sentences, list):
                print("Converting iterator to list for multi-epoch training...")
                sentences = list(sentences)
        
        if not self.vocab: 
            self.build_vocab(sentences)
        if self.W is None or self.W_prime is None:
            self._initialize_vectors()
        if self.noise_distribution is None:
            self._prepare_noise_distribution()
        if self.sample > 0 and not self.discard_probs:
            self._calculate_discard_probs()
        
        if self.use_cython:
            self._initialize_cython_globals()
        
        # Setup for loss calculation if needed
        total_loss = 0.0
        total_examples = 0
        total_example_count = 0
        
        # For tracking moving average loss
        recent_losses = []
        
        # Only count total examples if needed for learning rate decay
        examples_per_epoch = None  # For progress bar
        if decay_alpha:
            print(f"Counting total examples for learning rate decay (starting alpha = {self.alpha}, min_alpha = {self.min_alpha})...")
            # Calculate total example count for ONE epoch
            if self.sg:
                for _ in self.generate_skipgram_examples(sentences):
                    total_example_count += 1
            else:
                for _ in self.generate_cbow_examples(sentences):
                    total_example_count += 1
            
            # Store examples per epoch for progress bar
            examples_per_epoch = total_example_count
            # Multiply by epochs to get total examples across all epochs for learning rate decay
            total_example_count *= epochs

        # Create counter for total examples processed across all epochs
        total_examples_processed = 0

        # Initialize learning rate
        current_alpha = start_alpha = self.alpha
        self._update_learning_rate(current_alpha)

        if self.use_cython:
            print("Training started. Using Cython implementation")
        else:
            print("Training started. Using Python implementation")
        # Training loop for each epoch
        for epoch in range(epochs):
            # shuffle sentences each epoch (only if it's a list)
            if isinstance(sentences, list):
                random.shuffle(sentences)
                    
            examples_processed_in_epoch = 0
            batch_count = 0
            
            # Reset epoch loss if we're calculating it
            epoch_loss = 0.0
            
            # Create a progress bar if calculating loss
            if calculate_loss:
                # Choose bar format based on whether we know the total
                if decay_alpha:
                    # When we know total, show percentage
                    bar_format = '{l_bar}{bar}| {percentage:.2f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                else:
                    # When we don't know total, don't show percentage or remaining time
                    bar_format = '{l_bar}{bar}| [{elapsed}, {rate_fmt}{postfix}]'
                
                # Show a progress bar for this epoch
                progress_bar = tqdm(
                    desc=f"Epoch {epoch+1}/{epochs}",
                    total=examples_per_epoch,
                    bar_format=bar_format,
                    unit="ex",
                    mininterval=0.5  # Update twice per second
                )
            
            # If we're using batching and it's enabled (batch_size > 0)
            if batch_size > 0:
                batch_samples = []
                
                # For skip-gram, each sample is (input_word_idx, output_word_idx)
                if self.sg:
                    # Generate Skip-gram examples and batch them
                    for input_idx, output_idx in self.generate_skipgram_examples(sentences):
                        # Add this example to the current batch
                        batch_samples.append((input_idx, output_idx))
                        examples_processed_in_epoch += 1
                        
                        # Once we have enough examples for a batch, train on it
                        if len(batch_samples) >= batch_size:
                            # Train on this batch with the current learning rate
                            batch_loss = self._train_batch(batch_samples, current_alpha)
                            batch_count += 1
                            
                            # Update total examples processed for learning rate decay
                            total_examples_processed += batch_size
                            
                            # Decrease learning rate after processing batch if decay is enabled
                            if decay_alpha:
                                decay_factor = 1 - (total_examples_processed / total_example_count)
                                current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                                self._update_learning_rate(current_alpha)
                            
                            # If we're calculating loss, add this batch's loss to the epoch total
                            if calculate_loss:
                                epoch_loss += batch_loss
                                
                                # Add to recent losses for moving average
                                batch_avg_loss = batch_loss / len(batch_samples)
                                recent_losses.append(batch_avg_loss)
                                if len(recent_losses) > 100:
                                    recent_losses.pop(0)
                                    
                                # Update progress bar
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, b={batch_count}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, b={batch_count}"
                                progress_bar.set_postfix_str(postfix_str)
                                progress_bar.update(batch_size)
                            
                            # Clear the batch for the next set of examples
                            batch_samples = []
                    
                    # Train on any remaining examples in the final batch
                    if batch_samples:
                        remaining_batch_size = len(batch_samples)
                        
                        # Train on the final batch with the current learning rate
                        batch_loss = self._train_batch(batch_samples, current_alpha)
                        
                        # Update total examples processed for learning rate decay
                        total_examples_processed += remaining_batch_size
                        
                        # Decrease learning rate after the final batch if decay is enabled
                        if decay_alpha:
                            decay_factor = 1 - (total_examples_processed / total_example_count)
                            current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                            self._update_learning_rate(current_alpha)
                        
                        # If we're calculating loss, add this batch's loss to the epoch total
                        if calculate_loss:
                            epoch_loss += batch_loss
                            
                            # Update progress bar with final batch
                            if recent_losses:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, b={batch_count+1}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, b={batch_count+1}"
                                progress_bar.set_postfix_str(postfix_str)
                            progress_bar.update(remaining_batch_size)
                
                # For CBOW, each sample is ([context_word_indices], target_word_idx)
                else:
                    # Generate CBOW examples and batch them
                    for input_indices, output_idx in self.generate_cbow_examples(sentences):
                        # Add this example to the current batch
                        batch_samples.append((input_indices, output_idx))
                        examples_processed_in_epoch += 1
                        
                        # Once we have enough examples for a batch, train on it
                        if len(batch_samples) >= batch_size:
                            # Train on this batch with the current learning rate
                            batch_loss = self._train_batch(batch_samples, current_alpha)
                            batch_count += 1
                            
                            # Update total examples processed for learning rate decay
                            total_examples_processed += batch_size
                            
                            # Decrease learning rate after processing batch if decay is enabled
                            if decay_alpha:
                                decay_factor = 1 - (total_examples_processed / total_example_count)
                                current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                                self._update_learning_rate(current_alpha)
                            
                            # If we're calculating loss, add this batch's loss to the epoch total
                            if calculate_loss:
                                epoch_loss += batch_loss
                                
                                # Add to recent losses for moving average
                                batch_avg_loss = batch_loss / len(batch_samples)
                                recent_losses.append(batch_avg_loss)
                                if len(recent_losses) > 100:
                                    recent_losses.pop(0)
                                    
                                # Update progress bar
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, b={batch_count}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, b={batch_count}"
                                progress_bar.set_postfix_str(postfix_str)
                                progress_bar.update(batch_size)
                            
                            # Clear the batch for the next set of examples
                            batch_samples = []
                    
                    # Train on any remaining examples in the final batch
                    if batch_samples:
                        remaining_batch_size = len(batch_samples)
                        
                        # Train on the final batch with the current learning rate
                        batch_loss = self._train_batch(batch_samples, current_alpha)
                        
                        # Update total examples processed for learning rate decay
                        total_examples_processed += remaining_batch_size
                        
                        # Decrease learning rate after the final batch if decay is enabled
                        if decay_alpha:
                            decay_factor = 1 - (total_examples_processed / total_example_count)
                            current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                            self._update_learning_rate(current_alpha)
                        
                        # If we're calculating loss, add this batch's loss to the epoch total
                        if calculate_loss:
                            epoch_loss += batch_loss
                            
                            # Update progress bar with final batch
                            if recent_losses:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, b={batch_count+1}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, b={batch_count+1}"
                                progress_bar.set_postfix_str(postfix_str)
                            progress_bar.update(remaining_batch_size)
            
            # If we're not using batching, train on individual examples
            else:
                # For skip-gram
                if self.sg:
                    # Train on each Skip-gram example separately
                    for input_idx, output_idx in self.generate_skipgram_examples(sentences):
                        # Train on this skipgram example with the current learning rate
                        loss = self._train_skipgram_example(input_idx, output_idx, current_alpha)
                        
                        # Update total examples processed and decrease learning rate
                        total_examples_processed += 1
                        # Only decay learning rate if min_alpha is not None
                        if decay_alpha:
                            decay_factor = 1 - (total_examples_processed / total_example_count)
                            current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                            # Update Cython learning rate
                            self._update_learning_rate(current_alpha)
                        
                        # If we're calculating loss, add this example's loss to the epoch total
                        if calculate_loss:
                            epoch_loss += loss
                            
                            # Add to recent losses for moving average
                            recent_losses.append(loss)
                            if len(recent_losses) > 1000:
                                recent_losses.pop(0)
                                
                            # Only update progress bar every 10 examples for efficiency
                            if examples_processed_in_epoch % 10 == 0:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, e={examples_processed_in_epoch}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, e={examples_processed_in_epoch}"
                                progress_bar.set_postfix_str(postfix_str)
                                progress_bar.update(10)
                        
                        examples_processed_in_epoch += 1
                
                # For CBOW
                else:
                    # Train on each CBOW example separately
                    for input_indices, output_idx in self.generate_cbow_examples(sentences):
                        # Train on this CBOW example with the current learning rate
                        loss = self._train_cbow_example(input_indices, output_idx, current_alpha)
                        
                        # Update total examples processed and decrease learning rate
                        total_examples_processed += 1
                        # Only decay learning rate if min_alpha is not None
                        if decay_alpha:
                            decay_factor = 1 - (total_examples_processed / total_example_count)
                            current_alpha = max(self.min_alpha, start_alpha * decay_factor)
                            # Update Cython learning rate
                            self._update_learning_rate(current_alpha)
                        
                        # If we're calculating loss, add this example's loss to the epoch total
                        if calculate_loss:
                            epoch_loss += loss
                            
                            # Add to recent losses for moving average
                            recent_losses.append(loss)
                            if len(recent_losses) > 1000:
                                recent_losses.pop(0)
                                
                            # Only update progress bar every 10 examples for efficiency
                            if examples_processed_in_epoch % 10 == 0:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                if decay_alpha: 
                                    postfix_str = f"loss={recent_avg:.6f}, lr={current_alpha:.6f}, e={examples_processed_in_epoch}"
                                else:
                                    postfix_str = f"loss={recent_avg:.6f}, e={examples_processed_in_epoch}"
                                progress_bar.set_postfix_str(postfix_str)
                                progress_bar.update(10)
                        
                        examples_processed_in_epoch += 1
            
            # Close the progress bar at the end of the epoch
            if calculate_loss:
                # Close the progress bar - only update to total if we know it
                if examples_per_epoch is not None and progress_bar.total is not None:
                    progress_bar.update(progress_bar.total - progress_bar.n)
                progress_bar.close()
                
            # Add epoch loss to total if we're calculating it
            if calculate_loss:
                total_loss += epoch_loss
                total_examples += examples_processed_in_epoch
            
            # Call any registered callbacks with the model and epoch number
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)
        
        # Calculate and return the final average loss if requested
        if calculate_loss and total_examples > 0:
            final_avg_loss = total_loss / total_examples
            print(f"Training completed. Final average loss: {final_avg_loss:.6f}")
            return final_avg_loss
        
        return None

    def get_vector(self, word: str, normalize: bool = False) -> Optional[np.ndarray]:
        """
        Get the vector for a word.
        
        Parameters:
        -----------
        word: Input word
        normalize: If True, return the normalized vector (unit length)
        
        Returns:
        --------
        Word vector
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
        
        Parameters:
        -----------
        word: Input word
        
        Returns:
        --------
        Word vector or raises KeyError if word is not in vocabulary
        """
        
        return self.get_vector(word)
    
    def __contains__(self, word: str) -> bool:
        """
        Check if a word is in the vocabulary using the 'in' operator.
        
        Parameters:
        -----------
        word: Word to check
        
        Returns:
        --------
        True if the word is in the vocabulary, False otherwise
        """
        return word in self.vocab
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find the topn most similar words to the given word.
        
        Parameters:
        -----------
        word: Input word
        topn: Number of similar words to return
        
        Returns:
        --------
        List of (word, similarity) tuples
        """
        if word not in self.vocab:
            return []
        
        word_idx = self.vocab[word]
        word_vec = self.W[word_idx]
        
        # Compute cosine similarities
        norm = np.linalg.norm(self.W, axis=1)
        normalized_vecs = self.W / norm[:, np.newaxis]
        sim = np.dot(normalized_vecs, word_vec / np.linalg.norm(word_vec))
        
        # Get top similar words, excluding the input word
        most_similar = []
        for idx in (-sim).argsort():
            if idx != word_idx and len(most_similar) < topn:
                most_similar.append((self.index2word[idx], float(sim[idx])))
        
        return most_similar

    def similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Parameters:
        -----------
        word1: First word
        word2: Second word
        
        Returns:
        --------
        Cosine similarity between the two words (float between -1 and 1)
        
        Raises:
        -------
        KeyError: If either word is not in the vocabulary
        """
        if word1 not in self.vocab:
            raise KeyError(f"Word '{word1}' not found in vocabulary")
        if word2 not in self.vocab:
            raise KeyError(f"Word '{word2}' not found in vocabulary")
        
        word1_idx = self.vocab[word1]
        word2_idx = self.vocab[word2]
        
        word1_vec = self.W[word1_idx]
        word2_vec = self.W[word2_idx]
        
        # Calculate cosine similarity
        dot_product = np.dot(word1_vec, word2_vec)
        norm1 = np.linalg.norm(word1_vec)
        norm2 = np.linalg.norm(word2_vec)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path: Path to save the model
        
        Returns:
        --------
        None
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
            'use_double_precision': self.use_double_precision,
            'use_cython': self.use_cython,
            'gradient_clip': self.gradient_clip,
            'W': self.W,
            'W_prime': self.W_prime
        }
        np.save(path, model_data, allow_pickle=True)
    
    @classmethod
    def load(cls, path: str) -> 'Word2Vec':
        """
        Load a model from a file.
        
        Parameters:
        -----------
        path: Path to load the model from
        
        Returns:
        --------
        Loaded Word2Vec model
        """
        model_data = np.load(path, allow_pickle=True).item()
        
        # Get values with defaults if not found
        shrink_windows = model_data.get('shrink_windows', False)
        sample = model_data.get('sample', 1e-3)
        max_vocab_size = model_data.get('max_vocab_size', None)
        use_double_precision = model_data.get('use_double_precision', False)
        use_cython = model_data.get('use_cython', False)
        gradient_clip = model_data.get('gradient_clip', 1.0)
        
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
            use_double_precision=use_double_precision,
            use_cython=use_cython,
            gradient_clip=gradient_clip
        )
        
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        # Convert the word_counts back to a Counter
        model.word_counts = Counter(model_data.get('word_counts', {}))
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
        # If Cython is enabled but not loaded, try to re-import it
        if model.use_cython and model.word2vec_c is None:
            try:
                from .cython_ext import word2vec
                model.word2vec_c = word2vec
            except ImportError as e:
                model.use_cython = False
                warnings.warn(
                    f"The loaded model was trained with Cython acceleration, but the Cython extension " 
                    f"is not available in the current environment. Falling back to Python implementation.\n"
                    f"Error: {e}"
                )
                
        return model


# Helper functions for TempRefWord2Vec
def sample_sentences_to_token_count(corpus, target_tokens):
    """
    Samples sentences from a corpus until the target token count is reached.
    
    This function randomly selects sentences from the corpus until the total number
    of tokens reaches or slightly exceeds the target count. This is useful for balancing
    corpus sizes when comparing different time periods or domains.
    
    Parameters:
    -----------
    corpus : List[List[str]]
        A list of sentences, where each sentence is a list of tokens
    target_tokens : int
        The target number of tokens to sample
        
    Returns:
    --------
    List[List[str]]
        A list of sampled sentences with token count close to target_tokens
    """
    sampled_sentences = []
    current_tokens = 0
    sentence_indices = list(range(len(corpus)))
    random.shuffle(sentence_indices)
    
    for idx in sentence_indices:
        sentence = corpus[idx]
        if current_tokens + len(sentence) <= target_tokens:
            sampled_sentences.append(sentence)
            current_tokens += len(sentence)
        if current_tokens >= target_tokens:
            break
    return sampled_sentences


def add_corpus_tags(corpora, labels, target_words):
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
    
    Usage:
    ------
    1. Initialize with corpora from different time periods, labels for the periods,
       and target words to track for semantic change
    2. The model will process, balance, and combine the corpora
    3. Call train() without arguments to train on the preprocessed data
    4. Access semantic change through most_similar() or by directly analyzing the word vectors
       of temporal variants (e.g., "bread_1800" vs "bread_1900")
    
    Example:
    --------
    ```python
    # Corpora from different time periods
    corpus_1800s = [["bread", "baker", ...], ["food", "eat", ...], ...]
    corpus_1900s = [["bread", "supermarket", ...], ["food", "buy", ...], ...]
    
    # Initialize model
    model = TempRefWord2Vec(
        corpora=[corpus_1800s, corpus_1900s],
        labels=["1800s", "1900s"],
        targets=["bread", "food", "money"],
        vector_size=100,
        window=5,
        sg=0  # Use CBOW
    )
    
    # Train (uses preprocessed internal corpus)
    model.train(epochs=5)
    
    # Analyze semantic change
    model.most_similar("bread_1800")  # Words similar to "bread" in the 1800s
    model.most_similar("bread_1900")  # Words similar to "bread" in the 1900s
    ```
    """
    
    def __init__(
        self,
        corpora: List[List[List[str]]],  # List of corpora, each corpus is a list of sentences
        labels: List[str],               # Labels for each corpus (e.g., time periods)
        targets: List[str],              # Target words to trace semantic change
        balance: bool = True,            # Whether to balance the corpora
        **kwargs                         # Parameters passed to Word2Vec parent class
    ):
        """
        Initialize TempRefWord2Vec with multiple corpora and target words to track.
        
        Parameters:
        -----------
        corpora: List of corpora, each corpus is a list of sentences for a time period
        labels: Labels for each corpus (e.g., time periods like "1800s", "1900s")
        targets: List of target words to trace semantic change
        **kwargs: Arguments passed to Word2Vec parent class (vector_size, window, etc.)
        """
        # Check that corpora and labels have the same length
        if len(corpora) != len(labels):
            raise ValueError(f"Number of corpora ({len(corpora)}) must match number of labels ({len(labels)})")
    
        # check if sg = 1, else raise NotImplementedError
        if kwargs.get('sg') != 1:
            raise NotImplementedError("TempRefWord2Vec only supports Skip-gram model (sg=1)")
        
        # Store labels and targets as instance variables
        self.labels = labels
        self.targets = targets
        
        # print how many sentences in each corpus (each corpus a list of sentences)
        # and total size of each corpus (how many words; each sentence a list of words)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if not all(len(corpus) == 0 for corpus in corpora):
            for i, corpus in enumerate(corpora):
                print(f"Corpus {labels[i]} has {len(corpus)} sentences and {sum(len(sentence) for sentence in corpus)} words")

        # Calculate token counts and determine minimum
        if balance:
            corpus_token_counts = [sum(len(sentence) for sentence in corpus) for corpus in corpora]
            target_token_count = min(corpus_token_counts)
            print(f"Balancing corpora to minimum size: {target_token_count} tokens")
            
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
            if not all(len(corpus) == 0 for corpus in corpora):
                print(f"Period '{label}': {len(period_counter)} unique tokens, {sum(period_counter.values())} total tokens")
        
        # Combine all tagged corpora
        for corpus in tagged_corpora:
            self.combined_corpus.extend(corpus)
        # Skip printing if all corpora are empty (likely loading from saved model with dummy corpora)
        if not all(len(corpus) == 0 for corpus in corpora):
            print(f"Combined corpus: {len(self.combined_corpus)} sentences, {sum(len(s) for s in self.combined_corpus)} tokens")
        
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
        
        Parameters:
        -----------
        sentences: List of tokenized sentences
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
            print(f"Warning: {len(missing_variants)} temporal variants not found in corpus:")
            print(f"Sample: {missing_variants[:10]}")
            print("These variants will not be part of the temporal analysis.")
        
        # Add base words to vocabulary if they're not already there (should not be)
        added_base_words = 0
        for base_word in self.temporal_word_map:
            if base_word not in self.vocab:
                # Add the base word to vocabulary with count 1
                # First, get the index for this new word
                word_id = len(self.index2word)
                # Add to vocab dictionary
                self.vocab[base_word] = word_id
                # Add to index2word list
                self.index2word.append(base_word)
                # Add to word_counts
                self.word_counts[base_word] = 1  # Count 1 just to avoid division by 0
                added_base_words += 1
        
        if added_base_words > 0:
            # Update vocabulary size
            self.vocab_size = len(self.vocab)
            self.corpus_word_count += added_base_words
    
    def generate_skipgram_examples(self, 
                                 sentences: List[List[str]]) -> Generator[Tuple[int, int], None, None]:
        """
        Override parent method to implement temporal referencing in Skip-gram model.
        
        For Skip-gram, temporal referencing means that target words (inputs) are replaced
        with their temporal variants, while context words (outputs) remain unchanged.
        
        This implementation calls the parent's implementation and then modifies the yielded
        examples by converting any temporal variant context words to their base form.
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_idx, output_idx) tuples for positive examples
        """
        # Call the parent implementation to get basic examples
        for input_idx, output_idx in super().generate_skipgram_examples(sentences):
            # For Skip-gram:
            # - Input (center word) is already handled by data preprocessing (with temporal variants)
            # - Output (context word) should always use the base word form
            
            # Get the actual word for the output index
            output_word = self.index2word[output_idx]
            
            # Check if the output word is a temporal variant and convert to base form if needed
            if output_word in self.reverse_temporal_map:
                base_word = self.reverse_temporal_map[output_word]
                # Make sure the base word is in vocabulary
                if base_word in self.vocab:
                    output_idx = self.vocab[base_word]
            
            # Yield the modified example
            yield (input_idx, output_idx)
    
    def generate_cbow_examples(self, 
                             sentences: List[List[str]]) -> Generator[Tuple[List[int], int], None, None]:
        """
        Override parent method to implement temporal referencing in CBOW model.
        
        For CBOW, temporal referencing means:
        - Context words (inputs) should be converted to their base forms
        - Target words (outputs) are already handled by data preprocessing (with temporal variants)
        
        This implementation calls the parent's implementation and then modifies the yielded
        examples by converting any temporal variant context words to their base form.
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_indices, output_idx) tuples for positive examples
        """
        # Call the parent implementation to get basic examples
        for input_indices, output_idx in super().generate_cbow_examples(sentences):
            # For CBOW:
            # - Context words (inputs) should be in base form
            # - Output (center word) is already handled by data preprocessing (with temporal variants)
            
            # Process context word indices to ensure they use base forms
            modified_input_indices = []
            for idx in input_indices:
                word = self.index2word[idx]
                
                # If the word is a temporal variant, use its base form
                if word in self.reverse_temporal_map:
                    base_word = self.reverse_temporal_map[word]
                    # Make sure the base word is in vocabulary
                    if base_word in self.vocab:
                        idx = self.vocab[base_word]
                
                modified_input_indices.append(idx)
            
            # Yield the modified example
            yield (modified_input_indices, output_idx)

    def train(self, sentences: Optional[List[str]] = None, **kwargs) -> Optional[float]:
        """
        Train the TempRefWord2Vec model using the preprocessed combined corpus.
        
        Unlike the parent Word2Vec class, TempRefWord2Vec always uses its internal combined_corpus
        that was created and preprocessed during initialization. This ensures the training
        data has the proper temporal references.
        
        Parameters:
        -----------
        sentences: Ignored in TempRefWord2Vec, will use self.combined_corpus instead
        **kwargs: All additional arguments are passed to the parent's train method
                 (epochs, batch_size, alpha, min_alpha, callbacks, calculate_loss, etc.)
        
        Returns:
        --------
        Final loss value if calculate_loss is True in kwargs, None otherwise
        """
        if sentences is not None:
            print("Warning: TempRefWord2Vec always uses its internal preprocessed corpus for training.")
            print("The provided 'sentences' argument will be ignored (using self.combined_corpus instead).")
        
        # Call the parent's train method with our combined corpus and forward all kwargs
        return super().train(sentences=self.combined_corpus, **kwargs)

    def calculate_semantic_change(self, target_word: str, labels: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Calculate semantic change by comparing cosine similarities across time periods.
        
        Parameters:
        -----------
        target_word: Target word to analyze (must be one of the targets specified during initialization)
        labels: Time period labels (optional, defaults to labels from model initialization)
        
        Returns:
        --------
        Dict mapping transition names to lists of (word, change) tuples, sorted by change score (descending)
        
        Example:
        --------
        >>> changes = model.calculate_semantic_change("人民")
        >>> for transition, word_changes in changes.items():
        >>>     print(f"\\n{transition}:")
        >>>     print("Words moved towards:", word_changes[:5])  # Top 5 increases
        >>>     print("Words moved away:", word_changes[-5:])   # Top 5 decreases
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
                print(f"Warning: {from_variant} or {to_variant} not found in vocabulary. Skipping transition {transition}.")
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
        
        Returns:
        --------
        List of target words that were specified during model initialization
        """
        return self.targets.copy()

    def get_time_labels(self) -> List[str]:
        """
        Get the list of time period labels used in the model.
        
        Returns:
        --------
        List of time period labels that were specified during model initialization
        """
        return self.labels.copy()
    
    def get_period_vocab_counts(self, period: Optional[str] = None) -> Union[Dict[str, Counter], Counter]:
        """
        Get vocabulary counts for a specific period or all periods.
        
        Parameters:
        -----------
        period : str, optional
            The period label to get vocab counts for. If None, returns all periods.
            
        Returns:
        --------
        Union[Dict[str, Counter], Counter]
            If period is None: dictionary mapping period labels to Counter objects
            If period is specified: Counter object for that specific period
            
        Raises:
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
        
        Parameters:
        -----------
        path : str
            Path to save the model file
            
        Returns:
        --------
        None
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
            'use_double_precision': self.use_double_precision,
            'use_cython': self.use_cython,
            'gradient_clip': self.gradient_clip,
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
        print(f"TempRefWord2Vec model saved to {path}")
        print(f"Saved data includes:")
        print(f"  - Vocabulary: {len(self.vocab)} words")
        print(f"  - Time periods: {len(self.labels)} ({', '.join(self.labels)})")
        print(f"  - Target words: {len(self.targets)} ({', '.join(self.targets)})")
        print(f"  - Period vocab counts: {len(self.period_vocab_counts)} periods")
    
    @classmethod
    def load(cls, path: str) -> 'TempRefWord2Vec':
        """
        Load a TempRefWord2Vec model from a file.
        
        This overrides the parent load method to also restore:
        - Period-specific vocabulary counts
        - Target words and labels  
        - Temporal word mappings
        
        Parameters:
        -----------
        path : str
            Path to load the model from
            
        Returns:
        --------
        TempRefWord2Vec
            Loaded TempRefWord2Vec model with all temporal metadata restored
            
        Raises:
        -------
        ValueError
            If the file doesn't contain TempRefWord2Vec data
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
        use_double_precision = model_data.get('use_double_precision', False)
        use_cython = model_data.get('use_cython', False)
        gradient_clip = model_data.get('gradient_clip', 1.0)
        
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
            use_double_precision=use_double_precision,
            use_cython=use_cython,
            gradient_clip=gradient_clip,
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
        
        # Clear the dummy combined_corpus to save memory
        model.combined_corpus = []
        
        # Handle Cython if needed
        if model.use_cython and model.word2vec_c is None:
            try:
                from .cython_ext import word2vec
                model.word2vec_c = word2vec
            except ImportError as e:
                model.use_cython = False
                import warnings
                warnings.warn(
                    f"The loaded model was trained with Cython acceleration, but the Cython extension " 
                    f"is not available in the current environment. Falling back to Python implementation.\n"
                    f"Error: {e}"
                )
        
        print(f"TempRefWord2Vec model loaded from {path}")
        print(f"Restored data includes:")
        print(f"  - Vocabulary: {len(model.vocab)} words")
        print(f"  - Time periods: {len(model.labels)} ({', '.join(model.labels)})")
        print(f"  - Target words: {len(model.targets)} ({', '.join(model.targets)})")
        print(f"  - Period vocab counts: {len(model.period_vocab_counts)} periods")
        
        return model

