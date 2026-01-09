import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import random
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from scipy.special import psi, polygamma

from ..config import get_rng, get_python_rng, resolve_seed

logger = logging.getLogger("qhchina.analytics.topicmodels")

# Template directory path
_TEMPLATE_DIR = Path(__file__).parent.parent / "data" / "templates"

class LDAGibbsSampler:
    """
    Latent Dirichlet Allocation with Gibbs sampling implementation. 
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
        estimate_alpha: int = 1,
        min_doc_length: int = 24
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
            min_doc_length: Minimum document length (tokens) to trigger a warning during preprocessing (default: 24)
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
            # Validate alpha array length matches n_topics
            if not np.isscalar(alpha) and len(alpha_array) != n_topics:
                raise ValueError(f"alpha array length ({len(alpha_array)}) must match n_topics ({n_topics})")
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
        if not isinstance(min_doc_length, int) or min_doc_length < 1:
            raise ValueError(f"min_doc_length must be a positive integer, got {min_doc_length}")
        
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
        
        # Use isolated RNG instances to avoid affecting global state
        effective_seed = resolve_seed(random_state)
        self._rng = get_rng(effective_seed)
        self._python_rng = get_python_rng(effective_seed)
        
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
        
        # Minimum document length threshold for warnings
        self.min_doc_length = min_doc_length
    
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
                if len(doc_ids) < self.min_doc_length:
                    short_doc_count += 1
        
        # Issue a single warning for all short documents
        if short_doc_count > 0:
            warnings.warn(
                f"{short_doc_count} document(s) have fewer than {self.min_doc_length} tokens after filtering. "
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
        all_topics = self._rng.randint(0, self.n_topics, size=total_tokens)
        
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
        logger.info(f"Running Gibbs sampling for {n_iter} iterations ({impl_type} implementation).")
        if self.burnin > 0:
            logger.info(f"First {self.burnin} iterations are burn-in (discarded), then {self.iterations} iterations for inference.")
        
        # Use tqdm progress bar for iterations
        progress_bar = tqdm(
            range(total_iterations), 
            desc="Gibbs sampling",
            unit="iter"
        )
        
        for it in progress_bar:
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
                    progress_bar.set_postfix_str(f"perp={perplexity:.2f}, tok/s={tokens_per_sec:.0f}")
            
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
        new_topic = self._rng.choice(self.n_topics, p=p)
        
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
        
    def fit(self, documents: List[List[str]]) -> None:
        """
        Fit the LDA model to the given documents.
        
        Args:
            documents: List of tokenized documents (each document is a list of tokens)
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
        
        logger.info(f"Vocabulary size: {self.vocabulary_size}")
        logger.info(f"Number of documents: {len(self.docs_tokens)}")
        
        self.initialize(self.docs_tokens)
        self.run_gibbs_sampling()
        self._update_distributions()
    
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
        
        log_likelihood = 0.0
        token_count = 0
        
        # Small epsilon to prevent log(0)
        epsilon = 1e-10
        
        for d in range(len(self.doc_lengths)):
            doc_len = self.doc_lengths[d]
            doc_topics = self.theta[d, :]
            
            if doc_len == 0:
                continue

            for i in range(doc_len):
                word_id = self.docs_tokens[d, i]
                word_topic_probs = self.phi[:, word_id] 
                word_prob = np.sum(word_topic_probs * doc_topics)
                
                # Clamp word_prob to prevent log(0) or log(negative)
                word_prob = max(word_prob, epsilon)
                log_likelihood += np.log(word_prob)
                
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
        
        z_doc = self._rng.randint(0, self.n_topics, size=len(filtered_doc))
        n_dt_doc = np.zeros(self.n_topics, dtype=np.int32)
        np.add.at(n_dt_doc, z_doc, 1)
        
        vocab_size_beta = self.vocabulary_size * self.beta
        topic_normalizers = 1.0 / (self.n_t + vocab_size_beta)
        
        for _ in range(inference_iterations):
            for i, w in enumerate(filtered_doc):
                old_topic = z_doc[i]
                n_dt_doc[old_topic] -= 1
                
                p = self._compute_topic_probabilities(w, n_dt_doc, topic_normalizers)
                new_topic = self._rng.choice(self.n_topics, p=p)
                
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
        # Convert id_to_word keys to integers explicitly to preserve type after serialization
        id_to_word_int_keys = {int(k): v for k, v in self.id_to_word.items()} if self.id_to_word else None
        
        model_data = {
            'n_topics': self.n_topics,
            'alpha': self.alpha.tolist() if isinstance(self.alpha, np.ndarray) else self.alpha,
            'beta': self.beta,
            'min_word_count': self.min_word_count,
            'min_word_length': self.min_word_length,
            'max_vocab_size': self.max_vocab_size,
            'stopwords': list(self.stopwords) if self.stopwords else None,
            'vocabulary': self.vocabulary,
            'vocabulary_size': self.vocabulary_size,
            'word_to_id': self.word_to_id,
            'id_to_word': id_to_word_int_keys,
            'n_wt': self.n_wt.tolist() if self.n_wt is not None else None,
            'n_dt': self.n_dt.tolist() if self.n_dt is not None else None,
            'n_t': self.n_t.tolist() if self.n_t is not None else None,
            'theta': self.theta.tolist() if self.theta is not None else None,
            'phi': self.phi.tolist() if self.phi is not None else None,
            'z': self.z.tolist() if self.z is not None else None,
            'z_shape': self.z_shape,
            'doc_lengths': self.doc_lengths.tolist() if self.doc_lengths is not None else None,
            'docs_tokens': self.docs_tokens.tolist() if self.docs_tokens is not None else None,
            'total_tokens': self.total_tokens,
            'use_cython': self.use_cython,
            'estimate_alpha': self.estimate_alpha,
            'burnin': self.burnin,
            'random_state': self.random_state,
            'iterations': self.iterations,
            'min_doc_length': self.min_doc_length
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
        iterations = model_data.get('iterations', 100)
        
        model = cls(
            n_topics=model_data['n_topics'],
            alpha=model_data['alpha'],
            beta=model_data['beta'],
            iterations=iterations,
            min_word_count=model_data.get('min_word_count', 1),
            min_word_length=model_data.get('min_word_length', 1),
            max_vocab_size=model_data.get('max_vocab_size', None),
            stopwords=set(model_data.get('stopwords', [])) if model_data.get('stopwords') else None,
            use_cython=use_cython,
            estimate_alpha=estimate_alpha,
            burnin=burnin,
            random_state=random_state,
            min_doc_length=model_data.get('min_doc_length', 24)
        )
        
        model.vocabulary = model_data['vocabulary']
        model.vocabulary_size = model_data.get('vocabulary_size', len(model.vocabulary))
        model.word_to_id = model_data['word_to_id']
        
        # Restore id_to_word with integer keys (they may have been converted to strings during serialization)
        raw_id_to_word = model_data['id_to_word']
        if raw_id_to_word is not None:
            model.id_to_word = {int(k): v for k, v in raw_id_to_word.items()}
        else:
            model.id_to_word = None
        
        model.alpha_sum = np.sum(model.alpha)
        
        if model_data['n_wt'] is not None:
            model.n_wt = np.array(model_data['n_wt'], dtype=np.int32)
        if model_data['n_dt'] is not None:
            model.n_dt = np.array(model_data['n_dt'], dtype=np.int32)
        if model_data['n_t'] is not None:
            model.n_t = np.array(model_data['n_t'], dtype=np.int32)
        if model_data['theta'] is not None:
            model.theta = np.array(model_data['theta'], dtype=np.float64)
        if model_data['phi'] is not None:
            model.phi = np.array(model_data['phi'], dtype=np.float64)
        if model_data['z'] is not None:
            model.z = np.array(model_data['z'], dtype=np.int32)
        model.z_shape = model_data.get('z_shape')
        if model_data.get('doc_lengths') is not None:
            model.doc_lengths = np.array(model_data['doc_lengths'], dtype=np.int32)
        
        # Restore docs_tokens (required for perplexity and coherence calculations)
        if model_data.get('docs_tokens') is not None:
            model.docs_tokens = np.array(model_data['docs_tokens'], dtype=np.int32)
        
        # Restore total_tokens
        model.total_tokens = model_data.get('total_tokens', None)
        if model.total_tokens is None and model.doc_lengths is not None:
            model.total_tokens = int(np.sum(model.doc_lengths))
        
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
    
    @staticmethod
    def _compute_distribution_similarity(p: np.ndarray, q: np.ndarray, metric: str, eps: float = 1e-10) -> float:
        """
        Compute similarity/distance between two probability distributions.
        
        This is a shared implementation used by both topic_similarity and document_similarity.
        
        Args:
            p: First probability distribution (1D numpy array)
            q: Second probability distribution (1D numpy array)
            metric: Similarity metric to use. Options:
                    - 'jsd': Jensen-Shannon divergence (lower is more similar)
                    - 'hellinger': Hellinger distance (lower is more similar)
                    - 'cosine': Cosine similarity (higher is more similar)
                    - 'kl': KL divergence (lower is more similar, asymmetric)
            eps: Small constant to avoid numerical issues (default: 1e-10)
            
        Returns:
            Similarity/distance value based on chosen metric
            
        Raises:
            ValueError: If an unknown metric is specified
        """
        if metric == 'jsd':
            # Jensen-Shannon Divergence: symmetric measure based on KL divergence
            m = 0.5 * (p + q)
            kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0))
            kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0))
            return 0.5 * (kl_pm + kl_qm)
        
        elif metric == 'hellinger':
            # Hellinger distance: bounded in [0, 1]
            return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
        
        elif metric == 'cosine':
            # Cosine similarity: bounded in [-1, 1], typically [0, 1] for probability distributions
            norm_p = np.linalg.norm(p)
            norm_q = np.linalg.norm(q)
            if norm_p == 0 or norm_q == 0:
                return 0.0
            return np.dot(p, q) / (norm_p * norm_q)
        
        elif metric == 'kl':
            # KL divergence: asymmetric measure (p || q)
            return np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0))
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'jsd', 'hellinger', 'cosine', or 'kl'")
    
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
        
        return self._compute_distribution_similarity(self.phi[topic_i], self.phi[topic_j], metric)
    
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
        
        return self._compute_distribution_similarity(self.theta[doc_i], self.theta[doc_j], metric)
    
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
    
    def _compute_word_cooccurrence(self, window_size: int = 10) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
        """
        Compute word occurrence and co-occurrence counts from the corpus.
        
        Args:
            window_size: Size of the sliding window for co-occurrence (default: 10).
                        If window_size <= 0, uses document-level co-occurrence.
        
        Returns:
            Tuple of:
                - word_doc_count: Dict mapping word_id to number of documents containing it
                - word_pair_doc_count: Dict mapping (word_id_1, word_id_2) to co-occurrence count
        """
        word_doc_count = defaultdict(int)
        word_pair_doc_count = defaultdict(int)
        
        n_docs = len(self.doc_lengths)
        
        for d in tqdm(range(n_docs), desc="Computing co-occurrence", leave=False):
            doc_len = self.doc_lengths[d]
            if doc_len == 0:
                continue
            
            doc_tokens = self.docs_tokens[d, :doc_len]
            
            if window_size <= 0:
                # Document-level co-occurrence
                unique_words = set(doc_tokens)
                for w in unique_words:
                    word_doc_count[w] += 1
                
                # Count pairs (only once per document)
                unique_list = list(unique_words)
                for i, w1 in enumerate(unique_list):
                    for w2 in unique_list[i+1:]:
                        pair = (min(w1, w2), max(w1, w2))
                        word_pair_doc_count[pair] += 1
            else:
                # Sliding window co-occurrence
                seen_words = set()
                seen_pairs = set()
                
                for i in range(doc_len):
                    w1 = doc_tokens[i]
                    seen_words.add(w1)
                    
                    # Look at words within the window
                    window_end = min(i + window_size, doc_len)
                    for j in range(i + 1, window_end):
                        w2 = doc_tokens[j]
                        pair = (min(w1, w2), max(w1, w2))
                        seen_pairs.add(pair)
                
                for w in seen_words:
                    word_doc_count[w] += 1
                for pair in seen_pairs:
                    word_pair_doc_count[pair] += 1
        
        return dict(word_doc_count), dict(word_pair_doc_count)
    
    def coherence_umass(self, n_words: int = 10, eps: float = 1e-12) -> Tuple[float, List[float]]:
        """
        Calculate UMass topic coherence (Mimno et al., 2011).
        
        UMass coherence uses document co-occurrence and is defined as:
        C_UMass = (2 / (N*(N-1))) * sum_{i<j} log((D(w_i, w_j) + eps) / D(w_j))
        
        where D(w) is the document frequency of word w, and D(w_i, w_j) is the 
        number of documents containing both words.
        
        Args:
            n_words: Number of top words per topic to use for coherence calculation
            eps: Small constant to avoid log(0)
        
        Returns:
            Tuple of:
                - Average coherence across all topics
                - List of coherence values for each topic
        """
        if self.phi is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get document-level co-occurrence
        word_doc_count, word_pair_doc_count = self._compute_word_cooccurrence(window_size=0)
        
        topic_coherences = []
        
        for k in range(self.n_topics):
            # Get top n_words for this topic
            top_word_indices = np.argsort(-self.phi[k])[:n_words]
            
            coherence = 0.0
            n_pairs = 0
            
            for i, w_i in enumerate(top_word_indices):
                for j in range(i):
                    w_j = top_word_indices[j]
                    
                    # D(w_j) - document frequency of the more common word
                    d_wj = word_doc_count.get(w_j, 0)
                    if d_wj == 0:
                        continue
                    
                    # D(w_i, w_j) - co-occurrence count
                    pair = (min(w_i, w_j), max(w_i, w_j))
                    d_wi_wj = word_pair_doc_count.get(pair, 0)
                    
                    coherence += np.log((d_wi_wj + eps) / d_wj)
                    n_pairs += 1
            
            if n_pairs > 0:
                coherence /= n_pairs
            
            topic_coherences.append(coherence)
        
        avg_coherence = np.mean(topic_coherences)
        return avg_coherence, topic_coherences
    
    def coherence_npmi(self, n_words: int = 10, window_size: int = 10, 
                       eps: float = 1e-12) -> Tuple[float, List[float]]:
        """
        Calculate NPMI (Normalized Pointwise Mutual Information) topic coherence.
        
        NPMI coherence uses sliding window co-occurrence and is defined as:
        NPMI(w_i, w_j) = (log(P(w_i, w_j) / (P(w_i) * P(w_j)))) / (-log(P(w_i, w_j)))
        
        Values range from -1 (never co-occur) to +1 (always co-occur).
        
        Args:
            n_words: Number of top words per topic to use
            window_size: Size of the sliding window for co-occurrence
            eps: Small constant to avoid division by zero
        
        Returns:
            Tuple of:
                - Average coherence across all topics
                - List of coherence values for each topic
        """
        if self.phi is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get sliding window co-occurrence
        word_doc_count, word_pair_doc_count = self._compute_word_cooccurrence(window_size=window_size)
        
        n_docs = len(self.doc_lengths)
        topic_coherences = []
        
        for k in range(self.n_topics):
            top_word_indices = np.argsort(-self.phi[k])[:n_words]
            
            coherence = 0.0
            n_pairs = 0
            
            for i, w_i in enumerate(top_word_indices):
                for j in range(i):
                    w_j = top_word_indices[j]
                    
                    # P(w_i), P(w_j), P(w_i, w_j)
                    p_wi = word_doc_count.get(w_i, 0) / n_docs
                    p_wj = word_doc_count.get(w_j, 0) / n_docs
                    
                    pair = (min(w_i, w_j), max(w_i, w_j))
                    p_wi_wj = word_pair_doc_count.get(pair, 0) / n_docs
                    
                    if p_wi > 0 and p_wj > 0 and p_wi_wj > eps:
                        # NPMI formula
                        pmi = np.log((p_wi_wj + eps) / (p_wi * p_wj + eps))
                        npmi = pmi / (-np.log(p_wi_wj + eps))
                        coherence += npmi
                        n_pairs += 1
            
            if n_pairs > 0:
                coherence /= n_pairs
            
            topic_coherences.append(coherence)
        
        avg_coherence = np.mean(topic_coherences)
        return avg_coherence, topic_coherences
    
    def coherence(self, method: str = 'umass', n_words: int = 10, 
                  window_size: Optional[int] = None, **kwargs) -> Tuple[float, List[float]]:
        """
        Calculate topic coherence using the specified method.
        
        Coherence measures how semantically similar the top words in each topic are.
        Higher coherence generally indicates more interpretable topics.
        
        Args:
            method: Coherence measure to use. Options:
                   - 'umass': UMass coherence (Mimno et al., 2011). Uses document co-occurrence.
                              Range: typically negative, higher (less negative) is better.
                   - 'npmi': NPMI coherence. Uses sliding window co-occurrence.
                            Range: -1 to 1, higher is better.
            n_words: Number of top words per topic to use (default: 10)
            window_size: Size of sliding window for 'npmi' method (default: 10).
            **kwargs: Additional arguments passed to the specific coherence method
        
        Returns:
            Tuple of:
                - Average coherence across all topics
                - List of coherence values for each topic
        
        Example:
            >>> model.fit(documents)
            >>> avg_coherence, topic_coherences = model.coherence('npmi')
            >>> print(f"Average NPMI coherence: {avg_coherence:.4f}")
        """
        method = method.lower()
        
        if method == 'umass':
            return self.coherence_umass(n_words=n_words, **kwargs)
        elif method == 'npmi':
            ws = window_size if window_size is not None else 10
            return self.coherence_npmi(n_words=n_words, window_size=ws, **kwargs)
        else:
            raise ValueError(f"Unknown coherence method: {method}. Use 'umass' or 'npmi'")
    
    def evaluate(self, n_words: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the topic model.
        
        Calculates multiple quality metrics including perplexity, coherence measures,
        and topic diversity.
        
        Args:
            n_words: Number of top words per topic for coherence calculation
            verbose: Whether to print results
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        if self.phi is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        results = {}
        
        # Perplexity
        results['perplexity'] = self.perplexity()
        
        # Coherence measures
        results['coherence_umass'], results['coherence_umass_per_topic'] = self.coherence_umass(n_words=n_words)
        results['coherence_npmi'], results['coherence_npmi_per_topic'] = self.coherence_npmi(n_words=n_words)
        
        # Topic diversity (proportion of unique words in top-N across all topics)
        all_top_words = set()
        total_words = 0
        for k in range(self.n_topics):
            top_word_indices = np.argsort(-self.phi[k])[:n_words]
            all_top_words.update(top_word_indices)
            total_words += n_words
        results['topic_diversity'] = len(all_top_words) / total_words
        
        # Average topic size (entropy of topic distribution across documents)
        topic_dist = self.get_topic_distribution()
        results['topic_entropy'] = -np.sum(topic_dist * np.log(topic_dist + 1e-10))
        
        if verbose:
            logger.info("=" * 50)
            logger.info("Topic Model Evaluation")
            logger.info("=" * 50)
            logger.info(f"Perplexity:         {results['perplexity']:.2f}")
            logger.info(f"UMass Coherence:    {results['coherence_umass']:.4f}")
            logger.info(f"NPMI Coherence:     {results['coherence_npmi']:.4f}")
            logger.info(f"Topic Diversity:    {results['topic_diversity']:.4f}")
            logger.info(f"Topic Entropy:      {results['topic_entropy']:.4f}")
            logger.info("=" * 50)
        
        return results
    
    @classmethod
    def train_multiple(
        cls,
        documents: List[List[str]],
        n_runs: int = 5,
        n_topics: int = 10,
        random_seeds: Optional[List[int]] = None,
        return_all_models: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train multiple LDA models with different random seeds and analyze robustness.
        
        This method trains several models and computes stability metrics to assess
        how consistent the discovered topics are across different random initializations.
        
        Args:
            documents: List of tokenized documents
            n_runs: Number of models to train (default: 5)
            n_topics: Number of topics
            random_seeds: Optional list of random seeds (if None, auto-generated)
            return_all_models: Whether to return all trained models (default: False)
            verbose: Whether to print progress and results
            **kwargs: Additional arguments passed to LDAGibbsSampler
        
        Returns:
            Dictionary containing:
                - 'best_model': The model with highest coherence
                - 'coherence_scores': List of coherence scores for each run
                - 'perplexity_scores': List of perplexity scores for each run
                - 'stability_score': Average pairwise topic similarity across runs
                - 'topic_alignment': Topic alignment across runs
                - 'all_models': List of all models (if return_all_models=True)
        
        Example:
            >>> results = LDAGibbsSampler.train_multiple(
            ...     documents, n_runs=5, n_topics=10, iterations=100
            ... )
            >>> print(f"Stability: {results['stability_score']:.4f}")
            >>> best_model = results['best_model']
        """
        if random_seeds is None:
            random_seeds = list(range(42, 42 + n_runs))
        elif len(random_seeds) != n_runs:
            raise ValueError(f"random_seeds length ({len(random_seeds)}) must match n_runs ({n_runs})")
        
        models = []
        coherence_scores = []
        perplexity_scores = []
        
        if verbose:
            logger.info(f"Training {n_runs} models with {n_topics} topics...")
        
        for i, seed in tqdm(enumerate(random_seeds), total=n_runs, desc="Training models"):
            if verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"Run {i+1}/{n_runs} (seed={seed})")
                logger.info('='*50)
            
            model = cls(
                n_topics=n_topics,
                random_state=seed,
                **kwargs
            )
            model.fit(documents)
            
            # Calculate metrics
            perplexity = model.perplexity()
            coherence, _ = model.coherence_umass()
            
            models.append(model)
            coherence_scores.append(coherence)
            perplexity_scores.append(perplexity)
            
            if verbose:
                logger.info(f"Perplexity: {perplexity:.2f}, UMass Coherence: {coherence:.4f}")
        
        # Find best model (highest coherence)
        best_idx = np.argmax(coherence_scores)
        best_model = models[best_idx]
        
        # Calculate stability (average pairwise topic similarity across models)
        stability_scores = []
        topic_alignments = []
        
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # Calculate topic alignment using Hungarian algorithm
                alignment, similarity = cls._align_topics(models[i], models[j])
                topic_alignments.append({
                    'run_i': i,
                    'run_j': j,
                    'alignment': alignment,
                    'similarity': similarity
                })
                stability_scores.append(similarity)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        results = {
            'best_model': best_model,
            'best_run_index': best_idx,
            'coherence_scores': coherence_scores,
            'perplexity_scores': perplexity_scores,
            'stability_score': avg_stability,
            'topic_alignments': topic_alignments,
            'coherence_mean': np.mean(coherence_scores),
            'coherence_std': np.std(coherence_scores),
            'perplexity_mean': np.mean(perplexity_scores),
            'perplexity_std': np.std(perplexity_scores)
        }
        
        if return_all_models:
            results['all_models'] = models
        
        if verbose:
            logger.info("\n" + "=" * 50)
            logger.info("ROBUSTNESS ANALYSIS RESULTS")
            logger.info("=" * 50)
            logger.info(f"Number of runs:          {n_runs}")
            logger.info(f"Best run:                {best_idx + 1} (seed={random_seeds[best_idx]})")
            logger.info(f"Coherence (mean±std):    {results['coherence_mean']:.4f} ± {results['coherence_std']:.4f}")
            logger.info(f"Perplexity (mean±std):   {results['perplexity_mean']:.2f} ± {results['perplexity_std']:.2f}")
            logger.info(f"Topic Stability:         {avg_stability:.4f}")
            logger.info("=" * 50)
            logger.info("\nInterpretation:")
            logger.info("  - Stability close to 1.0 indicates highly robust topics")
            logger.info("  - Stability below 0.5 suggests topics are unstable")
            logger.info("  - Low std in coherence indicates consistent quality")
        
        return results
    
    @staticmethod
    def _align_topics(model1: 'LDAGibbsSampler', model2: 'LDAGibbsSampler') -> Tuple[List[Tuple[int, int]], float]:
        """
        Align topics between two models using the Hungarian algorithm.
        
        Finds the optimal one-to-one mapping between topics from two models
        that maximizes the total cosine similarity.
        
        Args:
            model1: First LDA model
            model2: Second LDA model
        
        Returns:
            Tuple of:
                - List of (topic_from_model1, topic_from_model2) pairs
                - Average similarity of aligned topics
        """
        n_topics = model1.n_topics
        if model2.n_topics != n_topics:
            raise ValueError("Models must have the same number of topics")
        
        # Build similarity matrix using cosine similarity
        similarity_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(n_topics):
                # Cosine similarity between topic distributions
                p = model1.phi[i]
                q = model2.phi[j]
                similarity_matrix[i, j] = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-10)
        
        # Use Hungarian algorithm to find optimal assignment
        # We want to maximize similarity, so we negate and use linear_sum_assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        except ImportError:
            # Fallback to greedy assignment if scipy is not available
            row_ind = list(range(n_topics))
            col_ind = []
            available = set(range(n_topics))
            for i in range(n_topics):
                best_j = max(available, key=lambda j: similarity_matrix[i, j])
                col_ind.append(best_j)
                available.remove(best_j)
        
        alignment = list(zip(row_ind, col_ind))
        avg_similarity = np.mean([similarity_matrix[i, j] for i, j in alignment])
        
        return alignment, avg_similarity
    
    def visualize_documents(
        self,
        method: str = 'pca',
        n_clusters: Optional[int] = None,
        doc_labels: Optional[List[str]] = None,
        show_labels: bool = False,
        label_strategy: str = 'auto',
        use_adjusttext: bool = True,
        max_labels: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 150,
        alpha: float = 0.7,
        size: float = 50,
        cmap: str = 'tab10',
        title: Optional[str] = None,
        filename: Optional[str] = None,
        format: str = 'static',
        random_state: Optional[int] = None,
        highlight: Optional[Union[int, List[int]]] = None,
        n_topic_words: int = 4,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Visualize documents in 2D space using dimensionality reduction.
        
        Documents are automatically colored by dominant topic, or by k-means clusters if n_clusters is specified.
        
        Args:
            method: Dimensionality reduction method. Options:
                   - 'pca': Principal Component Analysis (fast, linear)
                   - 'tsne': t-SNE (slower, captures non-linear structure)
                   - 'mds': Multidimensional Scaling (moderate speed)
                   - 'umap': UMAP (requires umap-learn package, fast and effective)
            n_clusters: If specified, apply k-means clustering and color by cluster instead of topic
            doc_labels: Optional list of document names/labels (same length as number of documents)
            show_labels: Whether to show document labels on the plot
            label_strategy: How to handle label display:
                          - 'auto': Automatically decide based on number of documents
                          - 'all': Show all labels (use adjustText if available)
                          - 'sample': Show a random sample of labels (controlled by max_labels)
                          - 'none': Don't show any labels
            use_adjusttext: Use adjustText package for better label placement (if available)
            max_labels: Maximum number of labels to show per topic/cluster (used with 'sample' or 'auto' strategy)
            figsize: Figure size as (width, height). If None, automatically scales based on number of documents
            dpi: Resolution in dots per inch
            alpha: Transparency of points (0-1)
            size: Size of scatter plot points
            cmap: Colormap to use (matplotlib colormap name)
            title: Optional plot title (auto-generated if None)
            filename: If provided, save the plot to this file
            format: Output format:
                   - 'static': Static matplotlib plot
                   - 'html': Interactive HTML visualization with hover tooltips
            random_state: Random seed for reproducibility
            highlight: Topic ID(s) to highlight. Can be a single int or list of ints.
                      Only the specified topics will be colored; others will be gray.
                      In HTML format, all topics are shown in legend and can be toggled interactively.
            n_topic_words: Number of representative words to show for each topic in the legend (default: 4).
                          Increase figsize width if using many words to accommodate longer legend labels.
            **kwargs: Additional keyword arguments to pass to the dimensionality reduction method.
                     For t-SNE: perplexity, learning_rate, max_iter, etc.
                     For UMAP: n_neighbors, min_dist, metric, etc.
                     For PCA: whiten, svd_solver, tol, etc.
                     For MDS: metric, max_iter, eps, etc.
            
        Returns:
            2D coordinates array of shape (n_docs, 2) if format='static', None if format='html'
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Process highlight parameter
        if highlight is not None:
            if isinstance(highlight, int):
                highlight = [highlight]
            else:
                highlight = list(highlight)
        
        # Resolve random state (uses global seed if not specified)
        if random_state is None:
            random_state = self.random_state
        effective_seed = resolve_seed(random_state)
        viz_rng = get_rng(effective_seed)
        
        n_docs = self.theta.shape[0]
        
        # Dynamically adjust figure size based on number of documents if not provided
        if figsize is None:
            if n_docs < 100:
                figsize = (12, 10)
            elif n_docs < 500:
                figsize = (14, 12)
            elif n_docs < 1000:
                figsize = (16, 14)
            elif n_docs < 5000:
                figsize = (20, 18)
            else:  # >= 5000 documents
                figsize = (24, 22)
        
        # Dynamically adjust point size based on number of documents
        if n_docs > 1000:
            size = max(10, size * (500 / n_docs))  # Reduce point size for large datasets
        
        # Perform dimensionality reduction
        extra_info = {}  # Store additional info like variance explained
        if method == 'pca':
            from sklearn.decomposition import PCA
            # Filter kwargs to only include valid PCA parameters
            pca_kwargs = {k: v for k, v in kwargs.items() if k in ['whiten', 'svd_solver', 'tol', 'iterated_power']}
            reducer = PCA(n_components=2, random_state=random_state, **pca_kwargs)
            coords_2d = reducer.fit_transform(self.theta)
            extra_info['explained_variance'] = reducer.explained_variance_ratio_
        
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            # Set default init to 'pca' if not provided
            tsne_kwargs = {'init': 'pca', **kwargs}
            # Filter to valid t-SNE parameters
            valid_tsne_params = ['perplexity', 'early_exaggeration', 'learning_rate', 'max_iter', 
                                'n_iter_without_progress', 'min_grad_norm', 'metric', 'init', 
                                'verbose', 'method', 'angle', 'n_jobs']
            tsne_kwargs = {k: v for k, v in tsne_kwargs.items() if k in valid_tsne_params}
            reducer = TSNE(n_components=2, random_state=random_state, **tsne_kwargs)
            coords_2d = reducer.fit_transform(self.theta)
        
        elif method == 'mds':
            from sklearn.manifold import MDS
            # Set default normalized_stress if not provided
            mds_kwargs = {'normalized_stress': 'auto', **kwargs}
            # Filter to valid MDS parameters
            valid_mds_params = ['metric', 'n_init', 'max_iter', 'verbose', 'eps', 
                               'n_jobs', 'dissimilarity', 'normalized_stress']
            mds_kwargs = {k: v for k, v in mds_kwargs.items() if k in valid_mds_params}
            reducer = MDS(n_components=2, random_state=random_state, **mds_kwargs)
            coords_2d = reducer.fit_transform(self.theta)
        
        elif method == 'umap':
            try:
                import umap
                # Filter to valid UMAP parameters
                valid_umap_params = ['n_neighbors', 'min_dist', 'metric', 'n_epochs', 'learning_rate',
                                    'init', 'min_grad_norm', 'spread', 'low_memory', 'set_op_mix_ratio',
                                    'local_connectivity', 'repulsion_strength', 'negative_sample_rate',
                                    'transform_queue_size', 'a', 'b', 'angular_rp_forest', 'target_n_neighbors',
                                    'target_metric', 'target_weight', 'transform_seed', 'verbose']
                umap_kwargs = {k: v for k, v in kwargs.items() if k in valid_umap_params}
                reducer = umap.UMAP(n_components=2, random_state=random_state, **umap_kwargs)
                coords_2d = reducer.fit_transform(self.theta)
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Please install it with: pip install umap-learn"
                )
        
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'pca', 'tsne', 'mds', or 'umap'"
            )
        
        # Determine colors - infer from n_clusters
        if n_clusters is not None:
            # User specified n_clusters, so use k-means clustering
            # IMPORTANT: Apply k-means to the 2D coordinates, not the original high-dimensional space
            # This is much faster and visually consistent with what's displayed
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            colors = kmeans.fit_predict(coords_2d)
            color_label = "K-means Cluster"
            extra_info['n_clusters'] = n_clusters
        else:
            # Default: color by dominant topic
            colors = np.argmax(self.theta, axis=1)
            color_label = "Dominant Topic"
        
        # Handle document labels
        if doc_labels is not None and len(doc_labels) != n_docs:
            raise ValueError(
                f"doc_labels length ({len(doc_labels)}) must match number of documents ({n_docs})"
            )
        
        # Generate default labels if not provided
        if doc_labels is None:
            doc_labels = [f"Doc {i}" for i in range(n_docs)]
        
        # Determine which labels to show
        labels_to_show = []
        if show_labels:
            if label_strategy == 'none':
                pass
            elif label_strategy == 'all':
                labels_to_show = list(range(n_docs))
            elif label_strategy == 'sample':
                # Sample max_labels documents per topic/cluster
                if max_labels is None:
                    max_labels = 5
                
                # Group documents by their color (topic or cluster)
                labels_by_color = {}
                for doc_id, color in enumerate(colors):
                    if color not in labels_by_color:
                        labels_by_color[color] = []
                    labels_by_color[color].append(doc_id)
                
                # Sample up to max_labels from each topic/cluster
                labels_to_show = []
                for color, doc_ids in labels_by_color.items():
                    n_to_sample = min(max_labels, len(doc_ids))
                    sampled = viz_rng.choice(doc_ids, size=n_to_sample, replace=False).tolist()
                    labels_to_show.extend(sampled)
                    
            elif label_strategy == 'auto':
                if n_docs <= 20:
                    labels_to_show = list(range(n_docs))
                elif n_docs <= 100:
                    # Sample up to max_labels per topic/cluster
                    max_labels = max_labels or 3
                    
                    labels_by_color = {}
                    for doc_id, color in enumerate(colors):
                        if color not in labels_by_color:
                            labels_by_color[color] = []
                        labels_by_color[color].append(doc_id)
                    
                    labels_to_show = []
                    for color, doc_ids in labels_by_color.items():
                        n_to_sample = min(max_labels, len(doc_ids))
                        sampled = viz_rng.choice(doc_ids, size=n_to_sample, replace=False).tolist()
                        labels_to_show.extend(sampled)
                else:
                    # For large datasets, sample fewer per topic
                    max_labels = max_labels or 2
                    
                    labels_by_color = {}
                    for doc_id, color in enumerate(colors):
                        if color not in labels_by_color:
                            labels_by_color[color] = []
                        labels_by_color[color].append(doc_id)
                    
                    labels_to_show = []
                    for color, doc_ids in labels_by_color.items():
                        n_to_sample = min(max_labels, len(doc_ids))
                        sampled = viz_rng.choice(doc_ids, size=n_to_sample, replace=False).tolist()
                        labels_to_show.extend(sampled)
            else:
                raise ValueError(
                    f"Unknown label_strategy: {label_strategy}. Use 'auto', 'all', 'sample', or 'none'"
                )
        
        # Generate visualization based on format
        if format == 'static':
            return self._plot_static_scatter(
                coords_2d, colors, doc_labels, labels_to_show, 
                method, color_label, use_adjusttext,
                figsize, dpi, alpha, size, cmap, title, filename, extra_info, highlight, n_topic_words
            )
        
        elif format == 'html':
            self._plot_interactive_html(
                coords_2d, colors, doc_labels, method, color_label,
                title, filename, extra_info, highlight, n_topic_words, size
            )
            return None
        
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'static' or 'html'"
            )
    
    def _plot_static_scatter(
        self,
        coords_2d: np.ndarray,
        colors: np.ndarray,
        doc_labels: List[str],
        labels_to_show: List[int],
        method: str,
        color_label: str,
        use_adjusttext: bool,
        figsize: Tuple[int, int],
        dpi: int,
        alpha: float,
        size: float,
        cmap: str,
        title: Optional[str],
        filename: Optional[str],
        extra_info: Dict[str, Any],
        highlight: Optional[List[int]],
        n_topic_words: int = 4
    ) -> np.ndarray:
        """Create static matplotlib scatter plot."""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get topic top words for legend (if coloring by topic)
        unique_colors = np.unique(colors)
        
        # Apply highlight filter if specified
        if highlight is not None:
            # Filter unique_colors to only include highlighted topics
            unique_colors = np.array([c for c in unique_colors if c in highlight])
        
        legend_labels = []
        
        if color_label == "Dominant Topic" and self.phi is not None and self.id_to_word is not None:
            # Get top n_topic_words words for each topic
            for topic_id in unique_colors:
                topic_id = int(topic_id)
                top_word_indices = np.argsort(-self.phi[topic_id])[:n_topic_words]
                top_words = [self.id_to_word[i] for i in top_word_indices]
                legend_labels.append(f"Topic {topic_id}: {', '.join(top_words)}")
        else:
            # For k-means clusters or when no topic words available
            for c in unique_colors:
                legend_labels.append(f"{color_label} {int(c)}")
        
        # Create scatter plot with separate plots for each color to enable legend
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
        n_colors = len(unique_colors)
        
        # Plot non-highlighted points in gray if highlight is specified
        if highlight is not None:
            # Plot all non-highlighted points first in gray
            all_colors_set = set(colors)
            non_highlighted = [c for c in all_colors_set if c not in highlight]
            if non_highlighted:
                non_highlighted_mask = np.isin(colors, non_highlighted)
                ax.scatter(
                    coords_2d[non_highlighted_mask, 0], coords_2d[non_highlighted_mask, 1],
                    c='lightgray', alpha=alpha * 0.5, s=size, edgecolors='w', linewidth=0.5
                )
        
        # Plot highlighted colors (or all colors if no highlight specified)
        for i, color_val in enumerate(unique_colors):
            mask = colors == color_val
            color_rgb = colormap(i / max(n_colors - 1, 1))
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[color_rgb], alpha=alpha, s=size, edgecolors='w', linewidth=0.5,
                label=legend_labels[i]
            )
        
        # Add legend with colored dots
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, 
                 fancybox=True, shadow=True, fontsize=9)
        
        # Add labels
        if labels_to_show:
            if use_adjusttext:
                try:
                    from adjustText import adjust_text
                    texts = []
                    for idx in labels_to_show:
                        texts.append(
                            ax.text(coords_2d[idx, 0], coords_2d[idx, 1], 
                                   doc_labels[idx], fontsize=8)
                        )
                    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
                except ImportError:
                    warnings.warn(
                        "adjustText is not installed. Labels will be shown without adjustment. "
                        "Install it with: pip install adjusttext"
                    )
                    for idx in labels_to_show:
                        ax.text(coords_2d[idx, 0], coords_2d[idx, 1], 
                               doc_labels[idx], fontsize=8, alpha=0.7)
            else:
                for idx in labels_to_show:
                    ax.text(coords_2d[idx, 0], coords_2d[idx, 1], 
                           doc_labels[idx], fontsize=8, alpha=0.7)
        
        # Set title
        if title is None:
            if 'n_clusters' in extra_info:
                title = f'Document Visualization ({method.upper()}) - K-means with k={extra_info["n_clusters"]}'
            else:
                title = f'Document Visualization ({method.upper()}) - Colored by {color_label}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis labels with variance explained for PCA
        if method == 'pca' and 'explained_variance' in extra_info:
            var_exp = extra_info['explained_variance']
            ax.set_xlabel(f'{method.upper()} Component 1 ({var_exp[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Component 2 ({var_exp[1]:.1%} variance)', fontsize=12)
        else:
            ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Plot saved to: {filename}")
        
        plt.show()
        
        return coords_2d
    
    def _plot_interactive_html(
        self,
        coords_2d: np.ndarray,
        colors: np.ndarray,
        doc_labels: List[str],
        method: str,
        color_label: str,
        title: Optional[str],
        filename: Optional[str],
        extra_info: Dict[str, Any],
        highlight: Optional[List[int]],
        n_topic_words: int = 4,
        size: float = 50
    ) -> None:
        """Create interactive HTML visualization with JavaScript."""
        import json
        
        n_docs = len(coords_2d)
        
        # Get top words for each topic (configurable number of words)
        topic_top_words = []
        if self.phi is not None and self.id_to_word is not None:
            for k in range(self.n_topics):
                top_word_indices = np.argsort(-self.phi[k])[:n_topic_words]
                top_words = [self.id_to_word[i] for i in top_word_indices]
                topic_top_words.append(", ".join(top_words))
        else:
            topic_top_words = [f"Topic {k}" for k in range(self.n_topics)]
        
        # Prepare data for JavaScript
        points_data = []
        for i in range(n_docs):
            topic_dist = self.theta[i]
            top_topics = np.argsort(-topic_dist)[:3]
            topic_info_lines = []
            for t in top_topics:
                if t < len(topic_top_words):
                    topic_info_lines.append(
                        f"Topic {t} ({topic_dist[t]:.3f}): {topic_top_words[t]}"
                    )
                else:
                    topic_info_lines.append(f"Topic {t}: {topic_dist[t]:.3f}")
            topic_info = "<br>".join(topic_info_lines)
            
            # Check if this point should be highlighted
            is_highlighted = highlight is None or colors[i] in highlight
            
            points_data.append({
                'x': float(coords_2d[i, 0]),
                'y': float(coords_2d[i, 1]),
                'color': int(colors[i]),
                'label': doc_labels[i],
                'doc_id': i,
                'topic_info': topic_info,
                'highlighted': is_highlighted
            })
        
        # Generate color palette
        import matplotlib.cm as cm
        n_colors = len(np.unique(colors))
        colormap = cm.get_cmap('tab10')
        color_palette = [
            'rgb({},{},{})'.format(
                int(colormap(i / n_colors)[0] * 255),
                int(colormap(i / n_colors)[1] * 255),
                int(colormap(i / n_colors)[2] * 255)
            ) for i in range(n_colors)
        ]
        
        if title is None:
            if 'n_clusters' in extra_info:
                title = f'Interactive Document Visualization ({method.upper()}) - K-means with k={extra_info["n_clusters"]}'
            else:
                title = f'Interactive Document Visualization ({method.upper()})'
        
        # Dynamically adjust canvas size based on number of documents
        if n_docs < 100:
            canvas_width, canvas_height = 1000, 800
        elif n_docs < 500:
            canvas_width, canvas_height = 1200, 1000
        elif n_docs < 1000:
            canvas_width, canvas_height = 1400, 1200
        elif n_docs < 5000:
            canvas_width, canvas_height = 1600, 1400
        else:  # >= 5000 documents
            canvas_width, canvas_height = 2000, 1600
        
        # Load HTML template
        template_path = _TEMPLATE_DIR / "document_visualization.html"
        with open(template_path, 'r', encoding='utf-8') as f:
            html_template = f.read()
        
        # Substitute template variables
        html_content = html_template.replace('{{title}}', title)
        html_content = html_content.replace('{{canvas_width}}', str(canvas_width))
        html_content = html_content.replace('{{canvas_height}}', str(canvas_height))
        html_content = html_content.replace('{{points_data_json}}', json.dumps(points_data))
        html_content = html_content.replace('{{color_palette_json}}', json.dumps(color_palette))
        html_content = html_content.replace('{{color_label}}', color_label.replace('Dominant Topic', 'Topic'))
        html_content = html_content.replace('{{topic_words_json}}', json.dumps(topic_top_words))
        html_content = html_content.replace('{{size}}', str(size))
        
        # Save or display HTML
        if filename is None:
            filename = 'document_visualization.html'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive visualization saved to: {filename}")
        logger.info(f"Open this file in a web browser to view the interactive plot.")