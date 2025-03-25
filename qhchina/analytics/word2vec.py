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
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Union, Iterator, Generator, Any
import random
import math
from tqdm.auto import tqdm


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
    """
    
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_mean: bool = True,
        sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
        seed: int = 1,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        sample: float = 1e-3,  # Threshold for subsampling frequent words
        shrink_windows: bool = True,  # Whether to use dynamic window size
        exp_table_size: int = 1000,  # Size of sigmoid lookup table
        max_exp: float = 6.0,  # Range of sigmoid precomputation [-max_exp, max_exp]
        max_vocab_size: Optional[int] = None,  # Maximum vocabulary size, None for no limit
    ):
        """
        Initialize the Word2Vec model.
        
        Parameters:
        -----------
        vector_size: Dimensionality of the word vectors
        window: Maximum distance between the current and predicted word
        min_count: Ignores all words with frequency lower than this
        negative: Number of negative samples for negative sampling
        ns_exponent: Exponent used to shape the negative sampling distribution
        cbow_mean: If True, use mean of context word vectors, else use sum
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        seed: Seed for random number generator
        alpha: Initial learning rate
        min_alpha: Minimum learning rate
        sample: Threshold for subsampling frequent words. Default is 1e-3, set to 0 to disable.
        shrink_windows: If True, the effective window size is uniformly sampled from [1, window] 
                        for each target word during training. If False, always use the full window.
        exp_table_size: Size of sigmoid lookup table for precomputation
        max_exp: Range of values for sigmoid precomputation [-max_exp, max_exp]
        max_vocab_size: Maximum vocabulary size to keep, keeping the most frequent words.
                        None means no limit (keep all words above min_count).
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
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
        
        # Parameters for sigmoid precomputation
        self.exp_table_size = exp_table_size
        self.max_exp = max_exp
        
        # Precompute the sigmoid table and log sigmoid table
        self.sigmoid_table = np.zeros(exp_table_size, dtype=np.float32)
        self.log_sigmoid_table = np.zeros(exp_table_size, dtype=np.float32)
        self._precompute_sigmoid()
        
        # Precompute constants for faster sigmoid lookup
        # Scale and offset for direct mapping from x to table indices
        self.sigmoid_scale = self.exp_table_size / (2 * self.max_exp)  # Scale factor
        self.sigmoid_offset = self.exp_table_size // 2                # Half the table size (integer division)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # These will be initialized in build_vocab
        self.vocab = {}  # word -> (index, count)
        self.index2word = []
        self.word_counts = Counter()
        self.corpus_word_count = 0
        self.discard_probs = {}  # For subsampling frequent words
        
        # These will be initialized in _initialize_weights
        self.W = None  # Input word embeddings
        self.W_prime = None  # Output word embeddings (for negative sampling)
        self.noise_distribution = None  # For negative sampling
        
        # For tracking training progress
        self.epoch_losses = []
        self.total_examples = 0

    def _precompute_sigmoid(self) -> None:
        """
        Precompute sigmoid values for faster training.
        
        This method creates a lookup table for sigmoid(x) and log(sigmoid(x))
        for x values from -max_exp to +max_exp, discretized into exp_table_size bins.
        """
        for i in range(self.exp_table_size):
            # Calculate x value in range [-max_exp, max_exp]
            x = (i / self.exp_table_size * 2 - 1) * self.max_exp
            # Compute sigmoid(x) = 1 / (1 + exp(-x))
            self.sigmoid_table[i] = 1.0 / (1.0 + np.exp(-x))
            # Compute log(sigmoid(x))
            self.log_sigmoid_table[i] = np.log(self.sigmoid_table[i])
    
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

    def build_vocab(self, sentences: List[List[str]]) -> None:
        """
        Build vocabulary from a list of sentences.
        
        Parameters:
        -----------
        sentences: List of tokenized sentences (each sentence is a list of words)
        
        Returns:
        --------
        None
        """
        print("Building vocabulary...")
        
        # Count word occurrences
        for sentence in sentences:
            self.word_counts.update(sentence)
        
        # Filter words by min_count and create vocabulary
        retained_words = {word for word, count in self.word_counts.items() if count >= self.min_count}
        
        # If max_vocab_size is set, keep only the most frequent words
        if self.max_vocab_size is not None and len(retained_words) > self.max_vocab_size:
            # Sort words by frequency (highest first) and take the top max_vocab_size
            top_words = [word for word, _ in self.word_counts.most_common(self.max_vocab_size)]
            # Intersect with words that meet min_count criteria
            retained_words = {word for word in top_words if word in retained_words}
            print(f"Vocabulary limited to {len(retained_words)} most frequent words due to max_vocab_size={self.max_vocab_size}")
        
        # Create mappings
        self.index2word = []
        for word, count in self.word_counts.most_common():
            if word in retained_words:
                word_id = len(self.index2word)
                self.vocab[word] = (word_id, count)
                self.index2word.append(word)
        
        self.corpus_word_count = sum(self.vocab[word][1] for word in self.vocab)
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
        print("Calculating discard probabilities for subsampling...")
        
        self.discard_probs = {}
        total_words = self.corpus_word_count
        
        for word, (word_id, count) in self.vocab.items():
            # Calculate normalized word frequency
            word_freq = count / total_words
            # Calculate probability of discarding the word
            # Using the formula from the original word2vec paper
            discard_prob = 1.0 - np.sqrt(self.sample / word_freq)
            # Clamp the probability to [0, 1]
            discard_prob = max(0, min(1, discard_prob))
            self.discard_probs[word] = discard_prob

    def _initialize_weights(self) -> None:
        """
        Initialize word vectors and prepare noise distribution for negative sampling.
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
        ).astype(np.float32)
        
        # Initialize W_prime with small random values like W instead of zeros
        # This helps improve convergence during training
        self.W_prime = np.random.uniform(
            low=-bound, 
            high=bound, 
            size=(vocab_size, self.vector_size)
        ).astype(np.float32)
        
        # Prepare noise distribution for negative sampling
        self._prepare_noise_distribution()

    def _prepare_noise_distribution(self) -> None:
        """
        Prepare noise distribution for negative sampling.
        More frequent words have higher probability of being selected.
        Applies subsampling with the ns_exponent parameter to prevent 
        extremely common words from dominating.
        """
        print("Preparing noise distribution for negative sampling...")
        
        # Get counts of each word in the vocabulary
        word_counts = np.array([self.vocab[word][1] for word in self.index2word])
        
        # Apply the exponent to smooth the distribution
        noise_dist = word_counts ** self.ns_exponent
        
        # Normalize to get a probability distribution
        self.noise_distribution = noise_dist / np.sum(noise_dist)

    def generate_skipgram_examples(self, 
                                 sentences: List[List[str]]) -> Generator[Tuple[int, int], None, None]:
        """
        Generate Skip-gram training examples from sentences.
        
        A Skip-gram example is a tuple (input_idx, output_idx) where:
        - input_idx is the index of the center word
        - output_idx is the index of a context word
        
        For each positive example, the caller should generate negative examples using the noise distribution.
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_idx, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            # Filter words based on subsampling and vocabulary
            if self.sample > 0:
                # Filter words based on subsampling probability
                kept_words = [
                    word for word in sentence 
                    if word in self.vocab and np.random.random() >= self.discard_probs[word]
                ]
            else:
                # No subsampling, just filter out words not in vocabulary
                kept_words = [word for word in sentence if word in self.vocab]
            
            # Convert words to indices
            indices = [self.vocab[word][0] for word in kept_words]
            
            sentence_len = len(indices)
            if sentence_len == 0:
                continue
            
            # Process each word in the sentence
            for pos in range(sentence_len):
                # Determine window size for this target word
                if self.shrink_windows:
                    # Uniform sampling from 1 to self.window (inclusive)
                    dynamic_window = random.randint(1, self.window)
                else:
                    dynamic_window = self.window
                
                # Define context window bounds
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # Center word is the input
                center_idx = indices[pos]
                
                # Generate training examples (center, context)
                for context_pos in range(start, end):
                    # Skip the center word itself
                    if context_pos == pos:
                        continue
                    
                    # Yield example (input_idx, output_idx) 
                    # For Skip-gram: input is center, output is context
                    context_idx = indices[context_pos]
                    yield (center_idx, context_idx)

    def generate_cbow_examples(self, 
                             sentences: List[List[str]]) -> Generator[Tuple[List[int], int], None, None]:
        """
        Generate CBOW training examples from sentences.
        
        A CBOW example is a tuple (input_indices, output_idx) where:
        - input_indices is a list of indices of context words
        - output_idx is the index of the center word
        
        For each positive example, the caller should generate negative examples using the noise distribution.
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_indices, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            # Filter words based on subsampling and vocabulary
            if self.sample > 0:
                # Filter words based on subsampling probability
                kept_words = [
                    word for word in sentence 
                    if word in self.vocab and np.random.random() >= self.discard_probs[word]
                ]
            else:
                # No subsampling, just filter out words not in vocabulary
                kept_words = [word for word in sentence if word in self.vocab]
            
            # Convert words to indices
            indices = [self.vocab[word][0] for word in kept_words]
            
            sentence_len = len(indices)
            if sentence_len == 0:
                continue
            
            # Process each word in the sentence
            for pos in range(sentence_len):
                # Determine window size for this target word
                if self.shrink_windows:
                    # Uniform sampling from 1 to self.window (inclusive)
                    dynamic_window = random.randint(1, self.window)
                else:
                    dynamic_window = self.window
                
                # Define context window bounds
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # Get all context indices for this position
                context_indices = []
                for context_pos in range(start, end):
                    if context_pos != pos:  # Skip the center word
                        context_indices.append(indices[context_pos])
                
                if not context_indices:
                    continue
                
                # Center word is the output
                center_idx = indices[pos]
                
                # Yield example (input_indices, output_idx)
                # For CBOW: inputs are context words, output is center
                yield (context_indices, center_idx)

    def _train_skipgram_example(self, input_idx: int, output_idx: int, learning_rate: float) -> float:
        """
        Train the model on a single Skip-gram example.
        
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

    def _train_cbow_example(self, input_indices: List[int], output_idx: int, learning_rate: float) -> float:
        """
        Train the model on a single CBOW example.
        
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

    def _train_batch(self, samples: List[Tuple], learning_rate: float) -> float:
        """
        Train the model on a batch of samples in a fully vectorized manner.
        
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

    def train(self, sentences: List[str], 
              epochs: int = 1, 
              batch_size: int = 32,
              calculate_loss: bool = False) -> List[float]:
        """
        Train the Word2Vec model.
        
        Parameters:
        -----------
        sentences: List of tokenized sentences to train on. Required for training.
        epochs: Number of training epochs.
        batch_size: Batch size for training. If 1, use example-by-example training.
        calculate_loss: Whether to calculate and display loss during training.
        
        Returns:
        --------
        List of average losses per epoch
        """ 
        # Build vocabulary if it doesn't exist yet
        if not self.vocab:
            self.build_vocab(sentences)
            self._initialize_weights()
            if self.sample > 0:
                self._calculate_discard_probs()
        
        # Calculate linear decay of learning rate
        alpha_delta = (self.alpha - self.min_alpha) / max(1, epochs - 1)
        
        # Track losses
        epoch_losses = []

        total_words = sum(len(sentence) for sentence in sentences)
        print(f"Training on {len(sentences)} sentences with {total_words} words...")

        # Training loop
        for epoch in range(epochs):
            random.shuffle(sentences)
            current_alpha = self.alpha - alpha_delta * epoch
            print(f"Epoch {epoch+1}/{epochs} - Learning rate: {current_alpha}")
            total_loss = 0
            example_count = 0
            batch_count = 0

            # For tracking moving average loss
            recent_losses = []
            
            # Show progress bar only if calculating loss
            if calculate_loss:
                # Create a tqdm progress bar that doesn't show progress percentage
                progress_bar = tqdm(
                    desc=f"Epoch {epoch+1}/{epochs}",
                    bar_format='{desc}{postfix}',
                    position=0,
                    leave=True
                )
            
            # Single example training mode (batch_size = 1)
            if batch_size <= 1:
                # Skip-gram training
                if self.sg:
                    for input_idx, output_idx in self.generate_skipgram_examples(sentences):
                        # Train on this skipgram example
                        loss = self._train_skipgram_example(input_idx, output_idx, current_alpha)
                        
                        # Accumulate total loss and example count
                        total_loss += loss
                        example_count += 1
                        
                        # Add to recent losses for moving average
                        recent_losses.append(loss)
                        if len(recent_losses) > 1000:
                            recent_losses.pop(0)
                        
                        # Update progress bar with current loss if using
                        if calculate_loss and example_count % 100 == 0:  # Update every 100 examples to avoid too frequent updates
                            recent_avg = sum(recent_losses) / len(recent_losses)
                            progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, examples={example_count}")
                else:
                    # CBOW training
                    for input_indices, output_idx in self.generate_cbow_examples(sentences):
                        # Train on this CBOW example
                        loss = self._train_cbow_example(input_indices, output_idx, current_alpha)
                        
                        # Accumulate total loss and example count
                        total_loss += loss
                        example_count += 1
                        
                        # Add to recent losses for moving average
                        recent_losses.append(loss)
                        if len(recent_losses) > 1000:
                            recent_losses.pop(0)
                        
                        # Update progress bar with current loss if using
                        if calculate_loss and example_count % 100 == 0:  # Update every 100 examples
                            recent_avg = sum(recent_losses) / len(recent_losses)
                            progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, examples={example_count}")
            
            # Batch training mode (batch_size > 1)
            else:
                # Skip-gram batch training
                if self.sg:
                    batch_samples = []
                    for input_idx, output_idx in self.generate_skipgram_examples(sentences):
                        # Add this example to the current batch
                        batch_samples.append((input_idx, output_idx))
                        
                        # If we've collected enough examples, process the batch
                        if len(batch_samples) >= batch_size:
                            # Train on this batch
                            batch_loss = self._train_batch(batch_samples, current_alpha)
                            batch_count += 1
                            
                            # Accumulate total loss and example count
                            total_loss += batch_loss
                            example_count += len(batch_samples)
                            
                            # Calculate batch average loss
                            batch_avg_loss = batch_loss / len(batch_samples)
                            
                            # Add to recent losses for moving average
                            recent_losses.append(batch_avg_loss)
                            if len(recent_losses) > 1000:
                                recent_losses.pop(0)
                            
                            # Update progress bar with current loss if using
                            if calculate_loss:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, batches={batch_count}, examples={example_count}")
                            
                            # Reset batch
                            batch_samples = []
                    
                    # Process any remaining examples in the last batch
                    if batch_samples:
                        batch_loss = self._train_batch(batch_samples, current_alpha)
                        
                        # Accumulate total loss and example count
                        total_loss += batch_loss
                        example_count += len(batch_samples)
                        batch_count += 1
                        
                        # Update progress bar if using
                        if calculate_loss:
                            batch_avg_loss = batch_loss / len(batch_samples)
                            recent_losses.append(batch_avg_loss)
                            recent_avg = sum(recent_losses) / len(recent_losses)
                            progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, batches={batch_count}, examples={example_count}")
                
                # CBOW batch training
                else:
                    batch_samples = []
                    for input_indices, output_idx in self.generate_cbow_examples(sentences):
                        # Add this example to the current batch
                        batch_samples.append((input_indices, output_idx))
                        
                        # If we've collected enough examples, process the batch
                        if len(batch_samples) >= batch_size:
                            # Train on this batch
                            batch_loss = self._train_batch(batch_samples, current_alpha)
                            batch_count += 1
                            
                            # Accumulate total loss and example count
                            total_loss += batch_loss
                            example_count += len(batch_samples)
                            
                            # Calculate batch average loss
                            batch_avg_loss = batch_loss / len(batch_samples)
                            
                            # Add to recent losses for moving average
                            recent_losses.append(batch_avg_loss)
                            if len(recent_losses) > 1000:
                                recent_losses.pop(0)
                            
                            # Update progress bar with current loss if using
                            if calculate_loss:
                                recent_avg = sum(recent_losses) / len(recent_losses)
                                progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, batches={batch_count}, examples={example_count}")
                            
                            # Reset batch
                            batch_samples = []
                    
                    # Process any remaining examples in the last batch
                    if batch_samples:
                        batch_loss = self._train_batch(batch_samples, current_alpha)
                        
                        # Accumulate total loss and example count
                        total_loss += batch_loss
                        example_count += len(batch_samples)
                        batch_count += 1
                        
                        # Update progress bar if using
                        if calculate_loss:
                            batch_avg_loss = batch_loss / len(batch_samples)
                            recent_losses.append(batch_avg_loss)
                            recent_avg = sum(recent_losses) / len(recent_losses)
                            progress_bar.set_postfix_str(f"loss={recent_avg:.6f}, batches={batch_count}, examples={example_count}")
            
            # Calculate epoch average loss
            epoch_avg_loss = total_loss / max(1, example_count)
            epoch_losses.append(epoch_avg_loss)
            
            # Close the progress bar and print epoch summary
            if calculate_loss:
                progress_bar.close()

        self.total_examples += example_count
        self.epoch_losses.extend(epoch_losses)
        
        return epoch_losses

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the vector for a word.
        
        Parameters:
        -----------
        word: Input word
        
        Returns:
        --------
        Word vector or None if the word is not in vocabulary
        """
        if word in self.vocab:
            return self.W[self.vocab[word][0]]
        return None
    
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
        
        word_idx = self.vocab[word][0]
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
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
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
        print(f"Model saved to {path}")
    
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
        
        model = cls(
            vector_size=model_data['vector_size'],
            window=model_data['window'],
            min_count=model_data['min_count'],
            negative=model_data['negative'],
            ns_exponent=model_data['ns_exponent'],
            cbow_mean=model_data['cbow_mean'],
            sg=model_data['sg'],
            sample=sample,
            shrink_windows=shrink_windows,
            max_vocab_size=max_vocab_size
        )
        
        model.vocab = model_data['vocab']
        model.index2word = model_data['index2word']
        model.W = model_data['W']
        model.W_prime = model_data['W_prime']
        
        return model

def plot_training_loss(epoch_losses, title="Word2Vec Training Loss"):
    """
    Plot the training loss progression across epochs.
    
    Parameters:
    -----------
    epoch_losses: List of losses per epoch
    title: Title for the plot
    
    Returns:
    --------
    None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization, please install it.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    
    # Add loss values as text labels
    for i, loss in enumerate(epoch_losses):
        plt.annotate(f'{loss:.4f}', 
                     (i + 1, loss),
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center')
    
    plt.show()

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
        # Import the sampling function
        from qhchina.helpers.texts import sample_sentences_to_token_count, add_corpus_tags
        
        # Check that corpora and labels have the same length
        if len(corpora) != len(labels):
            raise ValueError(f"Number of corpora ({len(corpora)}) must match number of labels ({len(labels)})")
        
        # Calculate token counts and determine minimum
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

        # Initialize combined corpus before using it
        self.combined_corpus = []
        
        # Combine all tagged corpora
        for corpus in tagged_corpora:
            self.combined_corpus.extend(corpus)
        print(f"Combined corpus: {len(self.combined_corpus)} sentences, {sum(len(s) for s in self.combined_corpus)} tokens")
        
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
        
        # Build vocabulary using the combined corpus
        self.build_vocab(self.combined_corpus)
        
        # Initialize weights after building vocabulary
        self._initialize_weights()
        
        # Calculate discard probabilities for subsampling if needed
        if self.sample > 0:
            self._calculate_discard_probs()
    
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
        
        # Add base words to vocabulary if they're not already there
        added_base_words = 0
        for base_word in self.temporal_word_map:
            if base_word not in self.vocab:
                # Add the base word to vocabulary with count 0
                # First, get the index for this new word
                word_id = len(self.index2word)
                # Add to vocab dictionary
                self.vocab[base_word] = (word_id, 1)  # Count 1 just to avoid division by 0
                # Add to index2word list
                self.index2word.append(base_word)
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
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_idx, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            # Filter words based on subsampling and vocabulary
            if self.sample > 0:
                # Filter words based on subsampling probability
                kept_words = [
                    word for word in sentence 
                    if word in self.vocab and np.random.random() >= self.discard_probs[word]
                ]
            else:
                # No subsampling, just filter out words not in vocabulary
                kept_words = [word for word in sentence if word in self.vocab]
            
            # Convert words to indices, using temporal variants as appropriate
            sentence_len = len(kept_words)
            if sentence_len == 0:
                continue
            
            # Process each word in the sentence
            for pos in range(sentence_len):
                # Get the current word
                current_word = kept_words[pos]
                
                # Determine window size for this target word
                if self.shrink_windows:
                    # Uniform sampling from 1 to self.window (inclusive)
                    dynamic_window = random.randint(1, self.window)
                else:
                    dynamic_window = self.window
                
                # Define context window bounds
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # For Skip-gram:
                # - Input (center word) uses the temporal variant if it's in the temporal_word_map
                # - Output (context word) always uses the base word
                
                # Get the index for the center word (input)
                center_word = current_word
                center_idx = self.vocab[center_word][0]
                
                # Generate training examples (center, context)
                for context_pos in range(start, end):
                    # Skip the center word itself
                    if context_pos == pos:
                        continue
                    
                    # Get context word (always use base form for output)
                    context_word = kept_words[context_pos]
                    
                    # If context word is a temporal variant, use its base form
                    if context_word in self.reverse_temporal_map:
                        base_context_word = self.reverse_temporal_map[context_word]
                        context_idx = self.vocab[base_context_word][0]
                    else:
                        context_idx = self.vocab[context_word][0]
                    
                    # Yield example (input_idx, output_idx)
                    yield (center_idx, context_idx)
    
    def generate_cbow_examples(self, 
                             sentences: List[List[str]]) -> Generator[Tuple[List[int], int], None, None]:
        """
        Override parent method to implement temporal referencing in CBOW model.
        
        For CBOW, our temporal referencing implementation means:
        - Context words (inputs) remain as base words
        - Target words (outputs) are replaced with their temporal variants (already included in the data)
        
        Parameters:
        -----------
        sentences: List of sentences (lists of words)
        
        Returns:
        --------
        Generator yielding (input_indices, output_idx) tuples for positive examples
        """
        # Process sentences one by one
        for sentence in sentences:
            # Filter words based on subsampling and vocabulary
            if self.sample > 0:
                # Filter words based on subsampling probability
                kept_words = [
                    word for word in sentence 
                    if word in self.vocab and np.random.random() >= self.discard_probs[word]
                ]
            else:
                # No subsampling, just filter out words not in vocabulary
                kept_words = [word for word in sentence if word in self.vocab]
            
            # Convert words to indices
            sentence_len = len(kept_words)
            if sentence_len == 0:
                continue
            
            # Process each word in the sentence
            for pos in range(sentence_len):
                # Determine window size for this target word
                if self.shrink_windows:
                    # Uniform sampling from 1 to self.window (inclusive)
                    dynamic_window = random.randint(1, self.window)
                else:
                    dynamic_window = self.window
                
                # Define context window bounds
                start = max(0, pos - dynamic_window)
                end = min(sentence_len, pos + dynamic_window + 1)
                
                # Get all context indices for this position
                context_indices = []
                for context_pos in range(start, end):
                    if context_pos != pos:  # Skip the center word
                        # Get context word (input)
                        context_word = kept_words[context_pos]
                        
                        # For CBOW inputs (context words), always use base form
                        if context_word in self.reverse_temporal_map:
                            base_context_word = self.reverse_temporal_map[context_word]
                            # Make sure the base word is in vocabulary
                            if base_context_word in self.vocab:
                                context_idx = self.vocab[base_context_word][0]
                            else:
                                # If base word not in vocab, use the temporal variant
                                context_idx = self.vocab[context_word][0]
                        else:
                            context_idx = self.vocab[context_word][0]
                            
                        context_indices.append(context_idx)
                
                if not context_indices:
                    continue
                
                # Get center word (output)
                center_word = kept_words[pos]
                center_idx = self.vocab[center_word][0]
                
                # Yield example (input_indices, output_idx)
                yield (context_indices, center_idx)

    def train(self, sentences: Optional[List[str]] = None, 
              epochs: int = 1, 
              batch_size: int = 32,
              calculate_loss: bool = False) -> List[float]:
        """
        Train the TempRefWord2Vec model using the preprocessed combined corpus.
        
        Unlike the parent Word2Vec class, TempRefWord2Vec always uses its internal combined_corpus
        that was created and preprocessed during initialization. This ensures the training
        data has the proper temporal references.
        
        Parameters:
        -----------
        sentences: Ignored in TempRefWord2Vec, will use self.combined_corpus instead
        epochs: Number of training epochs
        batch_size: Batch size for training. If None, use self.batch_size. If 1, use example-by-example training.
        calculate_loss: Whether to calculate and display loss during training with a progress bar
        
        Returns:
        --------
        List of average losses per epoch
        """
        if sentences is not None:
            print("Warning: TempRefWord2Vec always uses its internal preprocessed corpus for training.")
            print("The provided 'sentences' argument will be ignored.")
        # Call the parent's train method with our combined corpus
        # Always pass the combined_corpus to avoid ValueErrors in the parent class
        return super().train(sentences=self.combined_corpus, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            calculate_loss=calculate_loss)