import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import random
import time
import warnings
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm.auto import trange
from scipy.special import psi, polygamma

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
        iterations: int = 100,
        burnin: int = 0,
        random_state: Optional[int] = None,
        log_interval: Optional[int] = None,
        min_word_count: int = 1,
        max_vocab_size: Optional[int] = None,
        min_word_length: int = 1,
        stopwords: Optional[set] = None,
        use_cython: bool = True,
        estimate_alpha: int = 1
    ):
        """
        Initialize the LDA model with Gibbs sampling.
        
        Args:
            n_topics: Number of topics
            alpha: Dirichlet prior for document-topic distributions (can be float or array of floats, where each float is the alpha for a different topic).
                  If None, uses the heuristic 50/n_topics from Griffiths and Steyvers (2004).
            beta: Dirichlet prior for topic-word distributions (float).
                  If None, uses the heuristic 1/n_topics from Griffiths and Steyvers (2004).
            iterations: Number of Gibbs sampling iterations, excluding burnin
            burnin: Number of initial iterations to run before hyperparameters estimation (default 0)
            random_state: Random seed for reproducibility
            log_interval: Calculate perplexity and print results every log_interval iterations
            min_word_count: Minimum count of word to be included in vocabulary
            max_vocab_size: Maximum vocabulary size to keep
            min_word_length: Minimum length of word to be included in vocabulary
            stopwords: Set of words to exclude from vocabulary
            use_cython: Whether to use Cython acceleration if available (default: True)
            estimate_alpha: Frequency for estimating alpha (0 = no estimation; default 1 = after every iteration, 2 = after every 2 iterations, etc.)
        """
        # Validate parameters
        if not isinstance(n_topics, int) or n_topics <= 0:
            raise ValueError(f"n_topics must be a positive integer, got {n_topics}")
        if alpha is not None and not (np.isscalar(alpha) or isinstance(alpha, (list, tuple, np.ndarray))):
            raise ValueError(f"alpha must be a scalar or array-like, got {type(alpha)}")
        if alpha is not None:
            alpha_array = np.atleast_1d(alpha)
            if np.any(alpha_array <= 0):
                raise ValueError(f"alpha must be positive, got values <= 0")
        if beta is not None and (not np.isscalar(beta) or beta <= 0):
            raise ValueError(f"beta must be a positive scalar, got {beta}")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {iterations}")
        if not isinstance(burnin, int) or burnin < 0:
            raise ValueError(f"burnin must be a non-negative integer, got {burnin}")
        if not isinstance(min_word_count, int) or min_word_count < 1:
            raise ValueError(f"min_word_count must be a positive integer, got {min_word_count}")
        if not isinstance(min_word_length, int) or min_word_length < 1:
            raise ValueError(f"min_word_length must be a positive integer, got {min_word_length}")
        if max_vocab_size is not None and (not isinstance(max_vocab_size, int) or max_vocab_size <= 0):
            raise ValueError(f"max_vocab_size must be a positive integer or None, got {max_vocab_size}")
        if not isinstance(estimate_alpha, int) or estimate_alpha < 0:
            raise ValueError(f"estimate_alpha must be a non-negative integer, got {estimate_alpha}")
        
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
            
        if np.isscalar(self.alpha):
            self.alpha = np.ones(n_topics, dtype=np.float64) * self.alpha
        else:
            self.alpha = np.ascontiguousarray(self.alpha, dtype=np.float64)
        
        self.alpha_sum = np.sum(self.alpha)
            
        self.iterations = iterations
        self.burnin = burnin
        self.random_state = random_state
        self.log_interval = log_interval
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.min_word_length = min_word_length
        self.stopwords = set() if stopwords is None else set(stopwords)
        self.estimate_alpha = estimate_alpha
        
        self.use_cython = False  # Default to False until successful import
        self.lda_sampler = None
        
        if use_cython:
            self._attempt_cython_import()
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
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
        
        self.docs_tokens = None
        self.doc_ids = None
        self.total_tokens = None
        
        # Results
        self.theta = None  # Document-topic distributions
        self.phi = None    # Topic-word distributions
        
        # Private thresholds for internal processing
        self._min_doc_length = 50  # Minimum document length threshold for warnings
    
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
        except ImportError as e:
            self.use_cython = False
            warnings.warn(
                f"Cython acceleration for LDA was requested but the extension "
                f"is not available in the current environment. Falling back to Python implementation, "
                f"which will be significantly slower.\n"
                f"Error: {e}"
            )
            return False
        
    def preprocess(self, documents: List[List[str]]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        """
        Convert token documents to word IDs and build vocabulary.
        Filter vocabulary based on min_word_count, min_word_length, stopwords, and max_vocab_size.
        
        Args:
            documents: List of tokenized documents (each document is a list of tokens)
            
        Returns:
            Tuple containing:
                - docs_as_ids: Documents with tokens converted to integer IDs
                - word_to_id: Mapping from words to integer IDs
                - id_to_word: Mapping from integer IDs to words
        """
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)
        
        filtered_words = {
            word for word, count in word_counts.items() 
            if count >= self.min_word_count and len(word) >= self.min_word_length and word not in self.stopwords
        }
        
        if self.max_vocab_size and len(filtered_words) > self.max_vocab_size:
            top_words = sorted(filtered_words, key=lambda w: word_counts[w], reverse=True)[:self.max_vocab_size]
            filtered_words = set(top_words)
        
        word_to_id = {word: idx for idx, word in enumerate(sorted(filtered_words))}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        docs_as_ids = []
        short_doc_count = 0
        for i, doc in enumerate(documents):
            doc_ids = [word_to_id[word] for word in doc if word in word_to_id]
            if doc_ids:
                docs_as_ids.append(doc_ids)
                # Warn about short documents after filtering
                if len(doc_ids) < self._min_doc_length:
                    short_doc_count += 1
        
        # Issue a single warning for all short documents
        if short_doc_count > 0:
            warnings.warn(
                f"{short_doc_count} document(s) have fewer than {self._min_doc_length} tokens after filtering. "
                f"This may affect topic model quality, but training will continue.",
                UserWarning
            )

        return docs_as_ids, word_to_id, id_to_word
    
    def initialize(self, docs_as_ids: List[List[int]]) -> None:
        """
        Initialize data structures for Gibbs sampling.
        
        Args:
            docs_as_ids: Documents with tokens as integer IDs
        """
        n_docs = len(docs_as_ids)
        vocab_size = len(self.word_to_id)
        
        self.n_wt = np.zeros((vocab_size, self.n_topics), dtype=np.int32)
        self.n_dt = np.zeros((n_docs, self.n_topics), dtype=np.int32)
        self.n_t = np.zeros(self.n_topics, dtype=np.int32)
        
        self.doc_lengths = np.array([len(doc) for doc in docs_as_ids], dtype=np.int32)
        self.total_tokens = sum(self.doc_lengths)
        
        max_doc_length = max(self.doc_lengths) if n_docs > 0 else 0
        
        # Create 2D NumPy array for documents and topic assignments with padding (-1)
        self.docs_tokens = np.full((n_docs, max_doc_length), -1, dtype=np.int32)
        self.z = np.full((n_docs, max_doc_length), -1, dtype=np.int32)
        self.z_shape = (n_docs, max_doc_length)
        
        total_tokens = sum(self.doc_lengths)
        all_topics = np.random.randint(0, self.n_topics, size=total_tokens)
        
        token_idx = 0
        for d, doc in enumerate(docs_as_ids):
            doc_len = len(doc)
            # Store document tokens in 2D array
            self.docs_tokens[d, :doc_len] = doc
            doc_topics = all_topics[token_idx:token_idx+doc_len]
            token_idx += doc_len
            self.z[d, :doc_len] = doc_topics
            
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
        N = float(len(gammat))
        
        logphat = np.zeros(self.n_topics)
        for gamma in gammat:
            logphat += self._dirichlet_expectation(gamma)
        logphat /= N
        
        dalpha = np.copy(self.alpha)
        
        # Newton's method: compute gradient and Hessian
        gradf = N * (psi(np.sum(self.alpha)) - psi(self.alpha) + logphat)
        c = N * polygamma(1, np.sum(self.alpha))
        q = -N * polygamma(1, self.alpha)
        b = np.sum(gradf / q) / (1.0 / c + np.sum(1.0 / q))
        dalpha = -(gradf - b) / q
        
        if np.all(learning_rate * dalpha + self.alpha > 0):
            self.alpha += learning_rate * dalpha
            self.alpha_sum = np.sum(self.alpha)
        
        return self.alpha
        
    def run_gibbs_sampling(self) -> None:
        """
        Run Gibbs sampling for the specified number of iterations. 
        
        Uses Cython if available and enabled.
        """
        n_docs = len(self.docs_tokens)
        total_iterations = self.iterations + self.burnin
        
        if self.use_cython:
            if hasattr(self.lda_sampler, 'seed_rng'):
                self.lda_sampler.seed_rng(self.random_state)
        
        impl_type = "Cython" if self.use_cython else "Python"
        n_iter = total_iterations if self.burnin > 0 else self.iterations
        print(f"Running Gibbs sampling for {n_iter} iterations ({impl_type} implementation).")
        if self.burnin > 0:
            print(f"First {self.burnin} iterations are burn-in (discarded), then {self.iterations} iterations for inference.")
        
        for it in range(total_iterations):
            start_time = time.time()
            
            if self.use_cython:
                self.z = self.lda_sampler.run_iteration(
                    self.n_wt, self.n_dt, self.n_t, self.z, 
                    self.docs_tokens, self.doc_lengths, self.alpha, self.beta,
                    self.n_topics, self.vocabulary_size
                )
            else:
                for d in range(n_docs):
                    doc_len = self.doc_lengths[d]
                    for i in range(doc_len):
                        w = self.docs_tokens[d, i]
                        self.z[d, i] = self._sample_topic(d, i, w)
            
            is_burnin = it < self.burnin
            actual_it = it - self.burnin
            is_last_iteration = actual_it == self.iterations - 1
            is_hyperparam_estimation = self.estimate_alpha > 0 and actual_it % self.estimate_alpha == 0
            is_perplexity_estimation = self.log_interval and (actual_it % self.log_interval == 0 or is_last_iteration)

            if not is_burnin:
                if is_hyperparam_estimation or is_perplexity_estimation:
                    self._update_distributions()

                if is_hyperparam_estimation:
                    learning_rate = 1.0 - 0.9 * (actual_it / self.iterations)
                    gamma = self.n_dt + self.alpha
                    self._update_alpha(gamma, learning_rate)

                if is_perplexity_estimation:
                    elapsed = time.time() - start_time
                    perplexity = self.perplexity()
                    tokens_per_sec = self.total_tokens / elapsed
                    print(f"Iteration {actual_it}: Perplexity = {perplexity:.2f}, Tokens/sec = {tokens_per_sec:.1f}")
            
    def _compute_topic_probabilities(self, w: int, doc_topic_counts: np.ndarray, 
                                    topic_normalizers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute normalized topic probabilities for sampling.
        
        Args:
            w: Word ID
            doc_topic_counts: Document-topic counts (can be for a single document)
            topic_normalizers: Pre-computed normalizers (1 / (n_t + vocab_size * beta)).
                              If None, computes on-the-fly.
            
        Returns:
            Normalized probability distribution over topics
        """
        if topic_normalizers is None:
            topic_word_probs = (self.n_wt[w, :] + self.beta) / (self.n_t + self.vocabulary_size * self.beta)
        else:
            topic_word_probs = (self.n_wt[w, :] + self.beta) * topic_normalizers
        
        doc_topic_probs = doc_topic_counts + self.alpha
        p = topic_word_probs * doc_topic_probs
        
        return p / np.sum(p)
    
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
        old_topic = self.z[d, i]
        self.n_wt[w, old_topic] -= 1
        self.n_dt[d, old_topic] -= 1
        self.n_t[old_topic] -= 1
        
        p = self._compute_topic_probabilities(w, self.n_dt[d, :])
        new_topic = np.random.choice(self.n_topics, p=p)
        
        self.n_wt[w, new_topic] += 1
        self.n_dt[d, new_topic] += 1
        self.n_t[new_topic] += 1
        
        return new_topic
    
    def _update_distributions(self) -> None:
        """Update document-topic and topic-word distributions based on count matrices."""
        doc_lengths = self.doc_lengths[:, np.newaxis]
        self.theta = (self.n_dt + self.alpha) / (doc_lengths + self.alpha_sum)
        
        # Calculate phi with shape (vocab_size, n_topics), then transpose
        phi = np.zeros((self.vocabulary_size, self.n_topics), dtype=np.float64)
        
        for k in range(self.n_topics):
            if self.n_t[k] > 0:
                denominator = self.n_t[k] + self.vocabulary_size * self.beta
                phi[:, k] = (self.n_wt[:, k] + self.beta) / denominator
            else:
                phi[:, k] = 1.0 / self.vocabulary_size
                
        # Ensure phi is C-contiguous after transpose
        self.phi = np.ascontiguousarray(phi.T)
        
    def fit(self, documents: List[List[str]]) -> 'LDAGibbsSampler':
        """
        Fit the LDA model to the given documents.
        
        Args:
            documents: List of tokenized documents (each document is a list of tokens)
            
        Returns:
            The fitted model instance (self)
        """
        # Validate input documents
        if not isinstance(documents, list):
            raise TypeError(f"documents must be a list, got {type(documents)}")
        if len(documents) == 0:
            raise ValueError("documents cannot be empty")
        
        # Check that all documents are lists and not empty
        for i, doc in enumerate(documents):
            if not isinstance(doc, list):
                raise TypeError(f"Document {i} must be a list, got {type(doc)}")
            if len(doc) == 0:
                raise ValueError(f"Document {i} is empty. All documents must contain at least one token.")
        
        self.docs_tokens, self.word_to_id, self.id_to_word = self.preprocess(documents)
        self.vocabulary = list(self.word_to_id.keys())
        self.vocabulary_size = len(self.vocabulary)
        
        # Check that preprocessing left us with valid data
        if self.vocabulary_size == 0:
            raise ValueError("Vocabulary is empty after preprocessing. Check your min_word_count, "
                           "min_word_length, and stopwords settings.")
        if len(self.docs_tokens) == 0:
            raise ValueError("All documents were filtered out during preprocessing. "
                           "Check your vocabulary filtering settings.")
        
        print(f"Vocabulary size: {self.vocabulary_size}")
        print(f"Number of documents: {len(self.docs_tokens)}")
        
        self.initialize(self.docs_tokens)
        self.run_gibbs_sampling()
        self._update_distributions()
        
        return self
    
    def perplexity(self) -> float:
        """
        Calculate perplexity of the model on the training data.
        
        Returns:
            Perplexity value (lower is better)
        """
        if self.use_cython:
            # Ensure arrays are C-contiguous for Cython
            phi_contig = np.ascontiguousarray(self.phi)
            theta_contig = np.ascontiguousarray(self.theta)
            return self.lda_sampler.calculate_perplexity(
                phi_contig, theta_contig, self.docs_tokens, self.doc_lengths
            )
        
        log_likelihood = 0
        token_count = 0
        
        for d in range(len(self.doc_lengths)):
            doc_len = self.doc_lengths[d]
            doc_topics = self.theta[d, :]
            
            if doc_len == 0:
                continue

            for i in range(doc_len):
                word_id = self.docs_tokens[d, i]
                word_topic_probs = self.phi[:, word_id] 
                word_prob = np.sum(word_topic_probs * doc_topics)
                
                if word_prob > 0:
                    log_likelihood += np.log(word_prob)
                else:
                    log_likelihood += np.log(1e-10)
                
            token_count += doc_len
        
        if token_count == 0:
            return float('inf')
            
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
        top_indices = np.argsort(-self.phi, axis=1)[:, :n_words]
        
        for k in range(self.n_topics):
            topic_indices = top_indices[k]
            topic_words = [(self.id_to_word[i], self.phi[k, i]) for i in topic_indices]
            result.append(topic_words)
        
        return result
    
    def get_document_topics(self, doc_id: int, sort_by_prob: bool = False) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a specific document.
        
        Args:
            doc_id: ID of the document
            sort_by_prob: If True, sort topics by probability in descending order (default: False)
            
        Returns:
            List of (topic_id, probability) tuples
        """
        topics = [(k, self.theta[doc_id, k]) for k in range(self.n_topics)]
        if sort_by_prob:
            topics.sort(key=lambda x: x[1], reverse=True)
        return topics
    
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
        # Validate inputs
        if not isinstance(new_doc, list):
            raise TypeError(f"new_doc must be a list, got {type(new_doc)}")
        if len(new_doc) == 0:
            raise ValueError("new_doc cannot be empty")
        if not isinstance(inference_iterations, int) or inference_iterations <= 0:
            raise ValueError(f"inference_iterations must be a positive integer, got {inference_iterations}")
        
        filtered_doc = [self.word_to_id[w] for w in new_doc if w in self.word_to_id]
        
        if not filtered_doc:
            return np.ones(self.n_topics) / self.n_topics
        
        if self.use_cython and hasattr(self.lda_sampler, 'run_inference'):
            if hasattr(self.lda_sampler, 'seed_rng'):
                self.lda_sampler.seed_rng(self.random_state)
            
            # Convert to numpy array for Cython
            filtered_doc_array = np.array(filtered_doc, dtype=np.int32)
            
            return self.lda_sampler.run_inference(
                self.n_wt, self.n_t, filtered_doc_array,
                self.alpha, self.beta,
                self.n_topics, self.vocabulary_size,
                inference_iterations
            )
        
        z_doc = np.random.randint(0, self.n_topics, size=len(filtered_doc))
        n_dt_doc = np.zeros(self.n_topics, dtype=np.int32)
        np.add.at(n_dt_doc, z_doc, 1)
        
        vocab_size_beta = self.vocabulary_size * self.beta
        topic_normalizers = 1.0 / (self.n_t + vocab_size_beta)
        
        for _ in range(inference_iterations):
            for i, w in enumerate(filtered_doc):
                old_topic = z_doc[i]
                n_dt_doc[old_topic] -= 1
                
                p = self._compute_topic_probabilities(w, n_dt_doc, topic_normalizers)
                new_topic = np.random.choice(self.n_topics, p=p)
                
                z_doc[i] = new_topic
                n_dt_doc[new_topic] += 1
        
        alpha_sum = np.sum(self.alpha)
        theta_doc = (n_dt_doc + self.alpha) / (len(filtered_doc) + alpha_sum)
        return theta_doc
    
    def plot_topic_words(self, n_words: int = 10, figsize: Tuple[int, int] = (12, 8), 
                        fontsize: int = 10, filename: Optional[str] = None,
                        separate_files: bool = False, dpi: int = 72, 
                        orientation: str = "horizontal") -> None:
        """
        Plot the top words for each topic as a bar chart.
        
        Args:
            n_words: Number of top words to display per topic
            figsize: Figure size as (width, height)
            fontsize: Font size for the plot
            filename: If provided, save the plot to this file (or use as base name for separate files)
            separate_files: If True, save each topic as a separate file
            dpi: Resolution of the output image in dots per inch
            orientation: "horizontal" (words on x-axis, probabilities on y-axis) or 
                        "vertical" (probabilities on x-axis, words on y-axis with highest at top)
        """
        # Get top words for each topic
        topics = self.get_topics(n_words)
        
        if separate_files:
            # Create separate plots for each topic
            for k, topic in enumerate(topics):
                words, probs = zip(*topic)
                
                fig, ax = plt.subplots(figsize=figsize)
                
                if orientation == "vertical":
                    # Reverse order so highest probability is at top
                    words = words[::-1]
                    probs = probs[::-1]
                    y_pos = np.arange(len(words))
                    ax.barh(y_pos, probs, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=fontsize)
                    ax.set_xlabel('Probability', fontsize=fontsize)
                    ax.set_title(f'Topic {k}', fontsize=fontsize + 2)
                else:  # horizontal
                    x_pos = np.arange(len(words))
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
            fig, axes = plt.subplots(self.n_topics, 1, figsize=(figsize[0], figsize[1] * self.n_topics / 2))
            if self.n_topics == 1:
                axes = [axes]
            
            for k, (ax, topic) in enumerate(zip(axes, topics)):
                words, probs = zip(*topic)
                
                if orientation == "vertical":
                    # Reverse order so highest probability is at top
                    words = words[::-1]
                    probs = probs[::-1]
                    y_pos = np.arange(len(words))
                    ax.barh(y_pos, probs, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=fontsize)
                    ax.set_xlabel('Probability', fontsize=fontsize)
                    ax.set_title(f'Topic {k}', fontsize=fontsize + 2)
                else:  # horizontal
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
            'min_word_length': self.min_word_length,
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
            'use_cython': self.use_cython,
            'estimate_alpha': self.estimate_alpha,
            'burnin': self.burnin,
            'random_state': self.random_state
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
        
        use_cython = model_data.get('use_cython', True)
        estimate_alpha = model_data.get('estimate_alpha', 0)
        burnin = model_data.get('burnin', 0)
        random_state = model_data.get('random_state', None)
        
        model = cls(
            n_topics=model_data['n_topics'],
            alpha=model_data['alpha'],
            beta=model_data['beta'],
            min_word_length=model_data.get('min_word_length', model_data.get('min_length', 1)),
            stopwords=set(model_data.get('stopwords', [])) if model_data.get('stopwords') else None,
            use_cython=use_cython,
            estimate_alpha=estimate_alpha,
            burnin=burnin,
            random_state=random_state
        )
        
        model.vocabulary = model_data['vocabulary']
        model.vocabulary_size = len(model.vocabulary)
        model.word_to_id = model_data['word_to_id']
        model.id_to_word = model_data['id_to_word']
        model.alpha_sum = np.sum(model.alpha)
        
        if model_data['n_wt'] is not None:
            model.n_wt = np.array(model_data['n_wt'], dtype=np.int32)
        if model_data['n_dt'] is not None:
            model.n_dt = np.array(model_data['n_dt'], dtype=np.int32)
        if model_data['n_t'] is not None:
            model.n_t = np.array(model_data['n_t'], dtype=np.int32)
        if model_data['theta'] is not None:
            model.theta = np.array(model_data['theta'])
        if model_data['phi'] is not None:
            model.phi = np.array(model_data['phi'])
        if model_data['z'] is not None:
            model.z = np.array(model_data['z'], dtype=np.int32)
        model.z_shape = model_data.get('z_shape')
        if model_data.get('doc_lengths') is not None:
            model.doc_lengths = np.array(model_data['doc_lengths'], dtype=np.int32)
        
        if model.use_cython and model.lda_sampler is None:
            try:
                from .cython_ext import lda_sampler
                model.lda_sampler = lda_sampler
            except ImportError as e:
                model.use_cython = False
                warnings.warn(
                    f"The loaded model was trained with Cython acceleration, but the Cython extension " 
                    f"is not available in the current environment. Falling back to Python implementation.\n"
                    f"Error: {e}"
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
        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}. Must be between 0 and {self.n_topics-1}")
            
        topic_probs = self.theta[:, topic_id]
        top_doc_indices = np.argsort(-topic_probs)[:n_docs]
        
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
        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}. Must be between 0 and {self.n_topics-1}")
        
        topic_word_probs = self.phi[topic_id]
        top_word_indices = np.argsort(-topic_word_probs)[:n_words]
        
        return [(self.id_to_word[i], float(topic_word_probs[i])) for i in top_word_indices]
    
    def topic_similarity(self, topic_i: int, topic_j: int, metric: str = 'jsd') -> float:
        """
        Calculate similarity between two topics.
        
        Args:
            topic_i: First topic ID
            topic_j: Second topic ID
            metric: Similarity metric to use. Options:
                    - 'jsd': Jensen-Shannon divergence (default, lower is more similar)
                    - 'hellinger': Hellinger distance (lower is more similar)
                    - 'cosine': Cosine similarity (higher is more similar)
                    - 'kl': KL divergence (lower is more similar, asymmetric)
            
        Returns:
            Similarity/distance value based on chosen metric
        """
        if topic_i < 0 or topic_i >= self.n_topics or topic_j < 0 or topic_j >= self.n_topics:
            raise ValueError(f"Invalid topic IDs. Must be between 0 and {self.n_topics-1}")
        
        p = self.phi[topic_i]
        q = self.phi[topic_j]
        
        if metric == 'jsd':
            m = 0.5 * (p + q)
            eps = 1e-10
            kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0))
            kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0))
            return 0.5 * (kl_pm + kl_qm)
        
        elif metric == 'hellinger':
            return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
        
        elif metric == 'cosine':
            return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
        
        elif metric == 'kl':
            eps = 1e-10
            return np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0))
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jsd', 'hellinger', 'cosine', or 'kl'")
    
    def topic_correlation_matrix(self, metric: str = 'jsd') -> np.ndarray:
        """
        Calculate pairwise similarity/distance between all topics.
        
        Args:
            metric: Similarity metric to use (see topic_similarity for options)
            
        Returns:
            Square matrix of shape (n_topics, n_topics) with pairwise similarities/distances
        """
        corr_matrix = np.zeros((self.n_topics, self.n_topics))
        
        for i in range(self.n_topics):
            for j in range(i, self.n_topics):
                if i == j:
                    if metric == 'cosine':
                        corr_matrix[i, j] = 1.0
                    else:
                        corr_matrix[i, j] = 0.0
                else:
                    sim = self.topic_similarity(i, j, metric)
                    corr_matrix[i, j] = sim
                    corr_matrix[j, i] = sim
        
        return corr_matrix
    
    def document_similarity(self, doc_i: int, doc_j: int, metric: str = 'jsd') -> float:
        """
        Calculate similarity between two documents based on their topic distributions.
        
        Args:
            doc_i: First document ID
            doc_j: Second document ID
            metric: Similarity metric to use. Options:
                    - 'jsd': Jensen-Shannon divergence (default, lower is more similar)
                    - 'hellinger': Hellinger distance (lower is more similar)
                    - 'cosine': Cosine similarity (higher is more similar)
                    - 'kl': KL divergence (lower is more similar, asymmetric)
            
        Returns:
            Similarity/distance value based on chosen metric
        """
        n_docs = self.theta.shape[0]
        if doc_i < 0 or doc_i >= n_docs or doc_j < 0 or doc_j >= n_docs:
            raise ValueError(f"Invalid document IDs. Must be between 0 and {n_docs-1}")
        
        p = self.theta[doc_i]
        q = self.theta[doc_j]
        
        if metric == 'jsd':
            m = 0.5 * (p + q)
            eps = 1e-10
            kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0))
            kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0))
            return 0.5 * (kl_pm + kl_qm)
        
        elif metric == 'hellinger':
            return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
        
        elif metric == 'cosine':
            return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
        
        elif metric == 'kl':
            eps = 1e-10
            return np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0))
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jsd', 'hellinger', 'cosine', or 'kl'")
    
    def document_similarity_matrix(self, doc_ids: Optional[List[int]] = None, 
                                   metric: str = 'jsd') -> np.ndarray:
        """
        Calculate pairwise similarity/distance between documents.
        
        Args:
            doc_ids: List of document IDs to compare. If None, compares all documents.
            metric: Similarity metric to use (see document_similarity for options)
            
        Returns:
            Square matrix with pairwise similarities/distances
        """
        if doc_ids is None:
            doc_ids = list(range(self.theta.shape[0]))
        
        n = len(doc_ids)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    if metric == 'cosine':
                        sim_matrix[i, j] = 1.0
                    else:
                        sim_matrix[i, j] = 0.0
                else:
                    sim = self.document_similarity(doc_ids[i], doc_ids[j], metric)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        return sim_matrix