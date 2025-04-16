import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import random
import time
import warnings
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm.auto import trange
from scipy.special import psi, polygamma  # Add scipy.special imports for alpha estimation

class LDAGibbsSampler:
    """
    Latent Dirichlet Allocation with Gibbs sampling implementation. 
    Using Cython for speed.
    """
    
    def __init__(
        self,
        n_topics: int = 10,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        iterations: int = 1000,
        random_state: Optional[int] = None,
        log_interval: Optional[int] = None,
        min_count: int = 1,
        max_vocab_size: Optional[int] = None,
        min_length: int = 1,
        stopwords: Optional[set] = None,
        use_cython: bool = True,
        estimate_alpha: int = 1
    ):
        """
        Initialize the LDA model with Gibbs sampling.
        
        Args:
            n_topics: Number of topics
            alpha: Dirichlet prior for document-topic distributions (can be float or array of floats).
                  If None, uses the heuristic 50/n_topics from Griffiths and Steyvers (2004).
            beta: Dirichlet prior for topic-word distributions (can be float or array of floats)
            iterations: Number of Gibbs sampling iterations
            random_state: Random seed for reproducibility
            log_interval: Calculate perplexity and print results every log_interval iterations
            min_count: Minimum count of word to be included in vocabulary
            max_vocab_size: Maximum vocabulary size to keep
            min_length: Minimum length of word to be included in vocabulary
            stopwords: Set of words to exclude from vocabulary
            use_cython: Whether to use Cython acceleration if available (default: True)
            estimate_alpha: Frequency for estimating alpha (0 = no estimation; default 1 = after every iteration)
        """
        self.n_topics = n_topics
        # Use Griffiths and Steyvers (2004) heuristic if alpha is None
        if alpha is None:
            self.alpha = 50.0 / n_topics
        else:
            self.alpha = alpha
        
        if beta is None:
            self.beta = 1.0 / n_topics
        else:
            self.beta = beta
            
        # Ensure alpha is a numpy array for vectorized operations
        if np.isscalar(self.alpha):
            self.alpha = np.ones(n_topics) * self.alpha
        else:
            self.alpha = np.asarray(self.alpha)
            
        self.iterations = iterations
        self.random_state = random_state
        self.log_interval = log_interval
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.min_length = min_length
        self.stopwords = set() if stopwords is None else set(stopwords)
        self.estimate_alpha = estimate_alpha
        
        # Initialize Cython-related attributes
        self.use_cython = False  # Default to False until successful import
        self.lda_sampler = None  # Will hold the Cython module if imported
        
        # Try to import Cython extension if requested
        if use_cython:
            self._attempt_cython_import()
        
        # Set random seed if provided
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # The following attributes will be initialized during fitting
        self.vocabulary = None
        self.vocabulary_size = None
        self.word_to_id = None
        self.id_to_word = None
        
        # Counters for Gibbs sampling
        self.n_wt = None  # Word-topic count: n_wt[word_id, topic] = count
        self.n_dt = None  # Document-topic count: n_dt[doc_id, topic] = count
        self.n_t = None   # Topic count: n_t[topic] = count
        
        # Topic assignments
        self.z = None     # z[doc_id, position] = topic
        self.z_shape = None  # Store shape (doc_count, max_doc_length)
        self.doc_lengths = None  # Store length of each document
        
        # Input data
        self.docs_tokens = None
        self.doc_ids = None
        
        # Results
        self.theta = None  # Document-topic distributions
        self.phi = None    # Topic-word distributions
    
    def _attempt_cython_import(self) -> bool:
        """
        Attempt to import the Cython-optimized module.
        
        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            # Attempt to import the Cython module
            from .cython_ext import lda_sampler
            self.lda_sampler = lda_sampler
            self.use_cython = True
            return True
        except ImportError:
            self.use_cython = False
            warnings.warn(
                "Cython acceleration for LDA was requested but the extension "
                "is not available in the current environment. Falling back to Python implementation, "
                "which will be significantly slower."
            )
            return False
        
    def preprocess(self, documents: List[List[str]]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        """
        Convert token documents to word IDs and build vocabulary.
        Filter vocabulary based on min_count, min_length, stopwords, and max_vocab_size.
        
        Args:
            documents: List of tokenized documents (each document is a list of tokens)
            
        Returns:
            Tuple containing:
                - docs_as_ids: Documents with tokens converted to integer IDs
                - word_to_id: Mapping from words to integer IDs
                - id_to_word: Mapping from integer IDs to words
        """
        # Count word frequencies
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)
        
        # Filter by minimum count, length, and stopwords
        filtered_words = {
            word for word, count in word_counts.items() 
            if count >= self.min_count and len(word) >= self.min_length and word not in self.stopwords
        }
        
        # If max_vocab_size is specified, keep only the most frequent words
        if self.max_vocab_size and len(filtered_words) > self.max_vocab_size:
            top_words = sorted(filtered_words, key=lambda w: word_counts[w], reverse=True)[:self.max_vocab_size]
            filtered_words = set(top_words)
        
        # Create word-to-id mapping (sorted alphabetically for reproducibility)
        word_to_id = {word: idx for idx, word in enumerate(sorted(filtered_words))}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        # Convert documents to ID format, ignoring words not in vocabulary
        docs_as_ids = []
        for doc in documents:
            doc_ids = [word_to_id[word] for word in doc if word in word_to_id]
            if doc_ids:  # Only add document if it contains at least one valid word
                docs_as_ids.append(doc_ids)

        return docs_as_ids, word_to_id, id_to_word
    
    def initialize(self, docs_as_ids: List[List[int]]) -> None:
        """
        Initialize data structures for Gibbs sampling.
        
        Args:
            docs_as_ids: Documents with tokens as integer IDs
        """
        n_docs = len(docs_as_ids)
        vocab_size = len(self.word_to_id)
        
        # Initialize count matrices
        self.n_wt = np.zeros((vocab_size, self.n_topics), dtype=np.int32)
        self.n_dt = np.zeros((n_docs, self.n_topics), dtype=np.int32)
        self.n_t = np.zeros(self.n_topics, dtype=np.int32)
        
        # Store document lengths for later use
        self.doc_lengths = np.array([len(doc) for doc in docs_as_ids], dtype=np.int32)
        
        # Find the maximum document length to create padded arrays
        max_doc_length = max(self.doc_lengths) if n_docs > 0 else 0
        
        # Create NumPy array for topic assignments with padding
        # We'll use -1 as a padding value to indicate positions beyond doc length
        self.z = np.full((n_docs, max_doc_length), -1, dtype=np.int32)
        self.z_shape = (n_docs, max_doc_length)
        
        # Pre-generate all random topics for efficiency
        total_tokens = sum(self.doc_lengths)
        all_topics = np.random.randint(0, self.n_topics, size=total_tokens)
        
        token_idx = 0
        for d, doc in enumerate(docs_as_ids):
            doc_len = len(doc)
            
            # Get a slice of the pre-generated topics for this document
            doc_topics = all_topics[token_idx:token_idx+doc_len]
            token_idx += doc_len
            
            # Assign topics to the NumPy array
            self.z[d, :doc_len] = doc_topics
            
            # Update count matrices - using NumPy's broadcasting
            for i, (word_id, topic) in enumerate(zip(doc, doc_topics)):
                self.n_wt[word_id, topic] += 1
                self.n_dt[d, topic] += 1
                self.n_t[topic] += 1
    
    def _dirichlet_expectation(self, alpha):
        """
        For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
        
        Args:
            alpha: Dirichlet parameter
            
        Returns:
            Expected value of log(theta)
        """
        if len(alpha.shape) == 1:
            result = psi(alpha) - psi(np.sum(alpha))
        else:
            result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        return result.astype(alpha.dtype)  # keep the same precision as input
    
    def _update_alpha(self, gammat, learning_rate=1.0):
        """
        Update parameters for the Dirichlet prior on the per-document
        topic weights `alpha` using Newton's method.
        
        Args:
            gammat: Matrix of document-topic distributions (n_docs, n_topics)
            learning_rate: Factor to scale the update (default=1.0)
            
        Returns:
            Updated alpha vector
        """
        # Calculate the number of documents
        N = float(len(gammat))
        
        # Calculate the expected log probabilities
        logphat = np.zeros(self.n_topics)
        for gamma in gammat:
            logphat += self._dirichlet_expectation(gamma)
        logphat /= N
        
        # Copy current alpha for updates
        dalpha = np.copy(self.alpha)
        
        # Calculate gradient
        gradf = N * (psi(np.sum(self.alpha)) - psi(self.alpha) + logphat)
        
        # Calculate Hessian coefficients
        c = N * polygamma(1, np.sum(self.alpha))
        q = -N * polygamma(1, self.alpha)
        
        # Calculate the b coefficient
        b = np.sum(gradf / q) / (1.0 / c + np.sum(1.0 / q))
        
        # Calculate the step size
        dalpha = -(gradf - b) / q
        
        # Apply the step with learning rate if it keeps alpha positive
        if np.all(learning_rate * dalpha + self.alpha > 0):
            self.alpha += learning_rate * dalpha
        
        return self.alpha
        
    def run_gibbs_sampling(self) -> None:
        """
        Run Gibbs sampling for the specified number of iterations. 
        
        Uses Cython if available and enabled.
        """
        n_docs = len(self.docs_tokens)
        
        if self.use_cython:
            print(f"Running Gibbs sampling for {self.iterations} iterations (Cython implementation).")
        else:
            print(f"Running Gibbs sampling for {self.iterations} iterations (Python implementation).")
        
        for it in range(self.iterations):
            start_time = time.time()
            
            if self.use_cython:
                # Use the optimized Cython implementation
                # Make sure to cast beta to float to avoid NoneType errors
                self.z = self.lda_sampler.run_iteration(
                    self.n_wt, self.n_dt, self.n_t, self.z, 
                    self.docs_tokens, self.alpha, self.beta,
                    self.n_topics, self.vocabulary_size
                )
            else:
                # For each document
                for d in range(n_docs):
                    doc = self.docs_tokens[d]
                    # For each word in the document
                    for i, w in enumerate(doc):
                        # Sample a new topic
                        self.z[d, i] = self._sample_topic(d, i, w)
            
            # Estimate alpha if needed and we're past the first 1/3 of iterations
            if self.estimate_alpha > 0 and it > 0 and it % self.estimate_alpha == 0:
                # Calculate diminishing learning rate (from 1.0 to 0.1)
                # Map the iteration from alpha_start_iter..iterations to 1.0..0.1
                progress = it / self.iterations
                learning_rate = 1.0 - 0.9 * progress
                
                # Update document-topic distributions for alpha estimation
                self._update_distributions()
                
                # Convert theta to the gamma parameter expected by _update_alpha
                # gamma = count + alpha, which is n_dt + alpha
                gamma = self.n_dt + self.alpha
                
                # Update alpha with learning rate
                self._update_alpha(gamma, learning_rate)
                
            # Calculate perplexity every eval iterations
            if it > 0 and self.log_interval and it % self.log_interval == 0:
                elapsed = time.time() - start_time
                self._update_distributions()
                perplexity = self.perplexity()
                tokens_per_sec = sum(len(doc) for doc in self.docs_tokens) / elapsed
                print(f"Iteration {it}: Perplexity = {perplexity:.2f}, Tokens/sec = {tokens_per_sec:.1f}")
    
    def _sample_topic(self, d: int, i: int, w: int) -> int:
        """
        Sample a new topic for word w in document d at position i.
        
        Args:
            d: Document ID
            i: Position in document
            w: Word ID
            
        Returns:
            Sampled topic ID
        """

        # Decrease counts for current topic assignment
        old_topic = self.z[d, i]
        self.n_wt[w, old_topic] -= 1
        self.n_dt[d, old_topic] -= 1
        self.n_t[old_topic] -= 1
        
        # Calculate probability for each topic - vectorized
        topic_word_probs = (self.n_wt[w, :] + self.beta) / (self.n_t + self.vocabulary_size * self.beta)
        doc_topic_probs = self.n_dt[d, :] + self.alpha
        p = topic_word_probs * doc_topic_probs
        
        # Normalize probabilities
        p = p / np.sum(p)
        
        # Sample new topic
        new_topic = np.random.choice(self.n_topics, p=p)
        
        # Update counts for new topic assignment
        self.n_wt[w, new_topic] += 1
        self.n_dt[d, new_topic] += 1
        self.n_t[new_topic] += 1
        
        return new_topic
    
    def _update_distributions(self) -> None:
        """Update document-topic and topic-word distributions based on count matrices."""
        # Document-topic distribution (theta) - vectorized
        doc_lengths = np.array([len(doc) for doc in self.docs_tokens])[:, np.newaxis]
        
        # Ensure proper handling of alpha as an array
        alpha_sum = np.sum(self.alpha)
        self.theta = (self.n_dt + self.alpha) / (doc_lengths + alpha_sum)
        
        # Topic-word distribution (phi) - vectorized
        # phi should have shape (n_topics, vocabulary_size)
        
        # First calculate phi as (vocab_size, n_topics)
        # Note: n_wt has shape (vocab_size, n_topics) and n_t has shape (n_topics,)
        # We need to broadcast n_t correctly
        phi = np.zeros((self.vocabulary_size, self.n_topics), dtype=np.float64)
        
        for k in range(self.n_topics):
            # For each topic, calculate P(word|topic)
            # Here, n_wt[:,k] is the count of each word in topic k
            if self.n_t[k] > 0:  # Avoid division by zero for empty topics
                denominator = self.n_t[k] + self.vocabulary_size * self.beta
                phi[:, k] = (self.n_wt[:, k] + self.beta) / denominator
            else:
                # If topic is empty, use uniform distribution
                phi[:, k] = 1.0 / self.vocabulary_size
                
        # Transpose to get shape (n_topics, vocabulary_size)
        self.phi = phi.T
        
    def fit(self, documents: List[List[str]]) -> 'LDAGibbsSampler':
        """
        Fit the LDA model to the given documents.
        
        Args:
            documents: List of tokenized documents (each document is a list of tokens)
            
        Returns:
            The fitted model instance (self)
        """        
        # Preprocess documents
        self.docs_tokens, self.word_to_id, self.id_to_word = self.preprocess(documents)
        self.vocabulary = list(self.word_to_id.keys())
        self.vocabulary_size = len(self.vocabulary)
        
        print(f"Vocabulary size: {self.vocabulary_size}")
        print(f"Number of documents: {len(self.docs_tokens)}")
        
        # Initialize data structures
        self.initialize(self.docs_tokens)
        
        # Run Gibbs sampling, the core function
        self.run_gibbs_sampling()
        
        # Update distributions for final state
        self._update_distributions()
        
        return self
    
    def perplexity(self) -> float:
        """
        Calculate perplexity of the model on the training data.
        
        Returns:
            Perplexity value (lower is better)
        """
        # If Cython is available and enabled, use the optimized implementation
        if self.use_cython:
            return self.lda_sampler.calculate_perplexity(
                self.phi, self.theta, self.docs_tokens
            )
        
        # Otherwise, fall back to the Python implementation
        log_likelihood = 0
        token_count = 0
        
        # Disable indexing with documents and calculate per word
        for d, doc in enumerate(self.docs_tokens):
            doc_topics = self.theta[d, :]  # P(topic|doc)
            
            if len(doc) == 0:
                continue

            for word_id in doc:
                # P(word|topic) * P(topic|doc) for each topic
                word_topic_probs = self.phi[:, word_id] 
                word_prob = np.sum(word_topic_probs * doc_topics)
                
                # Prevent log(0) errors
                if word_prob > 0:
                    log_likelihood += np.log(word_prob)
                else:
                    # Use a small value instead of zero
                    log_likelihood += np.log(1e-10)
                
            token_count += len(doc)
        
        if token_count == 0:
            return float('inf')  # Return infinity if no tokens processed
            
        return np.exp(-log_likelihood / token_count)
    
    def get_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get the top words for each topic along with their probabilities.
        
        Args:
            n_words: Number of top words to return for each topic
            
        Returns:
            List of topics, each containing a list of (word, probability) tuples
        """
        result = []
        # Vectorized top-n selection for each topic
        top_indices = np.argsort(-self.phi, axis=1)[:, :n_words]
        
        for k in range(self.n_topics):
            topic_indices = top_indices[k]
            topic_words = [(self.id_to_word[i], self.phi[k, i]) for i in topic_indices]
            result.append(topic_words)
        
        return result
    
    def get_document_topics(self, doc_id: int) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a specific document.
        
        Args:
            doc_id: ID of the document
            
        Returns:
            List of (topic_id, probability) tuples
        """
        return [(k, self.theta[doc_id, k]) for k in range(self.n_topics)]
    
    def get_topic_distribution(self) -> np.ndarray:
        """
        Get overall topic distribution across the corpus.
        
        Returns:
            Array of topic probabilities
        """
        return np.mean(self.theta, axis=0)
    
    def inference(self, new_doc: List[str], 
                 inference_iterations: int = 100) -> np.ndarray:
        """
        Infer topic distribution for a new document.
        
        Args:
            new_doc: Tokenized document (list of tokens)
            inference_iterations: Number of sampling iterations for inference
            
        Returns:
            Topic distribution for the document
        """
        # Filter tokens not in vocabulary
        filtered_doc = [self.word_to_id[w] for w in new_doc if w in self.word_to_id]
        
        if not filtered_doc:
            return np.ones(self.n_topics) / self.n_topics  # Uniform if no known words
        
        # Initialize topic assignments randomly
        z_doc = np.random.randint(0, self.n_topics, size=len(filtered_doc))
        
        # Initialize document-topic counts
        n_dt_doc = np.zeros(self.n_topics, dtype=np.int32)
        np.add.at(n_dt_doc, z_doc, 1)  # Vectorized count update
        
        # Note: Currently there's no Cython implementation for inference,
        # so we always use the Python implementation regardless of use_cython flag
        if self.use_cython and hasattr(self.lda_sampler, 'run_inference'):
            # If a future Cython implementation becomes available
            # Replace this comment with call to self.lda_sampler.run_inference
            pass
        
        # Run Gibbs sampling
        for _ in range(inference_iterations):
            for i, w in enumerate(filtered_doc):
                # Remove current topic assignment
                old_topic = z_doc[i]
                n_dt_doc[old_topic] -= 1
                
                # Calculate probabilities for new topic - vectorized
                topic_word_probs = (self.n_wt[w, :] + self.beta) / (self.n_t + self.vocabulary_size * self.beta)
                doc_topic_probs = n_dt_doc + self.alpha
                p = topic_word_probs * doc_topic_probs
                
                # Normalize and sample
                p = p / np.sum(p)
                new_topic = np.random.choice(self.n_topics, p=p)
                
                # Update assignment
                z_doc[i] = new_topic
                n_dt_doc[new_topic] += 1
        
        # Calculate document-topic distribution with proper alpha sum
        alpha_sum = np.sum(self.alpha)
        theta_doc = (n_dt_doc + self.alpha) / (len(filtered_doc) + alpha_sum)
        return theta_doc
    
    def plot_topic_words(self, n_words: int = 10, figsize: Tuple[int, int] = (12, 8), 
                        fontsize: int = 10, filename: Optional[str] = None,
                        separate_files: bool = False, dpi: int = 72) -> None:
        """
        Plot the top words for each topic as a vertical bar chart.
        
        Args:
            n_words: Number of top words to display per topic
            figsize: Figure size as (width, height)
            fontsize: Font size for the plot
            filename: If provided, save the plot to this file (or use as base name for separate files)
            separate_files: If True, save each topic as a separate file
            dpi: Resolution of the output image in dots per inch
        """
        # Get top words for each topic
        topics = self.get_topics(n_words)
        
        if separate_files:
            # Create separate plots for each topic
            for k, topic in enumerate(topics):
                words, probs = zip(*topic)
                x_pos = np.arange(len(words))
                
                fig, ax = plt.subplots(figsize=figsize)
                ax.bar(x_pos, probs, align='center')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(words, fontsize=fontsize)
                ax.set_ylabel('Probability', fontsize=fontsize)
                ax.set_title(f'Topic {k}', fontsize=fontsize + 2)
                plt.tight_layout(pad=2.0)
                
                if filename:
                    # Create filename for each topic
                    base_name = filename.rsplit('.', 1)[0]
                    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
                    topic_filename = f"{base_name}_topic_{k}.{ext}"
                    plt.savefig(topic_filename, dpi=dpi, bbox_inches='tight')
                plt.close()
        else:
            # Create a single figure with subplots for all topics
            fig, axes = plt.subplots(self.n_topics, 1, figsize=(figsize[0], figsize[1] * self.n_topics / 2), 
                                    constrained_layout=True)
            if self.n_topics == 1:
                axes = [axes]
            
            for k, (ax, topic) in enumerate(zip(axes, topics)):
                words, probs = zip(*topic)
                x_pos = np.arange(len(words))
                
                ax.bar(x_pos, probs, align='center')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(words, fontsize=fontsize)
                ax.set_ylabel('Probability', fontsize=fontsize)
                ax.set_title(f'Topic {k}', fontsize=fontsize + 2)
            
            plt.tight_layout(pad=3.0)
            if filename:
                plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.show()
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'n_topics': self.n_topics,
            'alpha': self.alpha,
            'beta': self.beta,
            'min_length': self.min_length,
            'stopwords': list(self.stopwords) if self.stopwords else None,
            'vocabulary': self.vocabulary,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'n_wt': self.n_wt.tolist() if self.n_wt is not None else None,
            'n_dt': self.n_dt.tolist() if self.n_dt is not None else None,
            'n_t': self.n_t.tolist() if self.n_t is not None else None,
            'theta': self.theta.tolist() if self.theta is not None else None,
            'phi': self.phi.tolist() if self.phi is not None else None,
            'z': self.z.tolist() if self.z is not None else None,
            'z_shape': self.z_shape,
            'doc_lengths': self.doc_lengths.tolist() if self.doc_lengths is not None else None,
            'use_cython': self.use_cython,  # Save whether Cython was used
            'estimate_alpha': self.estimate_alpha  # Save alpha estimation settings
        }
        
        np.save(filepath, model_data)
    
    @classmethod
    def load(cls, filepath: str) -> 'LDAGibbsSampler':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded LDA model
        """
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Get use_cython value from the saved model, default to True if not present
        use_cython = model_data.get('use_cython', True)
        
        # Get estimate_alpha value from the saved model, default to 0 if not present
        estimate_alpha = model_data.get('estimate_alpha', 0)
        
        model = cls(
            n_topics=model_data['n_topics'],
            alpha=model_data['alpha'],
            beta=model_data['beta'],
            min_length=model_data.get('min_length', 1),  # Default to 1 for backward compatibility
            stopwords=set(model_data.get('stopwords', [])) if model_data.get('stopwords') else None,
            use_cython=use_cython,  # Pass the use_cython parameter from saved model
            estimate_alpha=estimate_alpha  # Pass the estimate_alpha parameter from saved model
        )
        
        model.vocabulary = model_data['vocabulary']
        model.vocabulary_size = len(model.vocabulary)
        model.word_to_id = model_data['word_to_id']
        model.id_to_word = model_data['id_to_word']
        
        if model_data['n_wt'] is not None:
            model.n_wt = np.array(model_data['n_wt'])
        if model_data['n_dt'] is not None:
            model.n_dt = np.array(model_data['n_dt'])
        if model_data['n_t'] is not None:
            model.n_t = np.array(model_data['n_t'])
        if model_data['theta'] is not None:
            model.theta = np.array(model_data['theta'])
        if model_data['phi'] is not None:
            model.phi = np.array(model_data['phi'])
        if model_data['z'] is not None:
            model.z = np.array(model_data['z'])
        model.z_shape = model_data.get('z_shape')
        if model_data.get('doc_lengths') is not None:
            model.doc_lengths = np.array(model_data['doc_lengths'])
        
        # If Cython is enabled in the loaded model but not available, try to re-import
        if model.use_cython and model.lda_sampler is None:
            try:
                from .cython_ext import lda_sampler
                model.lda_sampler = lda_sampler
            except ImportError:
                model.use_cython = False
                warnings.warn(
                    "The loaded model was trained with Cython acceleration, but the Cython extension " 
                    "is not available in the current environment. Falling back to Python implementation."
                )
        
        return model
    
    def get_top_documents(self, topic_id: int, n_docs: int = 10) -> List[Tuple[int, float]]:
        """
        Get the top n documents for a specific topic.
        
        Args:
            topic_id: ID of the topic
            n_docs: Number of top documents to return
            
        Returns:
            List of (document_id, probability) tuples, sorted by probability in descending order
        """
        # Check if topic_id is valid
        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}. Must be between 0 and {self.n_topics-1}")
            
        # Get the probability of the topic for each document
        topic_probs = self.theta[:, topic_id]
        
        # Get the indices of the top n documents
        top_doc_indices = np.argsort(-topic_probs)[:n_docs]
        
        # Return the document IDs and probabilities
        return [(int(doc_id), float(topic_probs[doc_id])) for doc_id in top_doc_indices]
    
    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top n words for a specific topic.
        
        Args:
            topic_id: ID of the topic
            n_words: Number of top words to return
            
        Returns:
            List of (word, probability) tuples, sorted by probability in descending order
        """
        # Check if topic_id is valid
        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}. Must be between 0 and {self.n_topics-1}")
        
        # Get the word probabilities for the given topic
        topic_word_probs = self.phi[topic_id]
        
        # Get the indices of the top n words
        top_word_indices = np.argsort(-topic_word_probs)[:n_words]
        
        # Return the words and probabilities
        return [(self.id_to_word[i], float(topic_word_probs[i])) for i in top_word_indices]