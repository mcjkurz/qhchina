import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import random
import time
import warnings
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm.auto import trange
# Try to import Cython module, with auto-compilation if needed
try:
    from .cython_ext import lda_sampler
    CYTHON_AVAILABLE = True
except ImportError:
    # First attempt to compile the extensions
    try:
        from .compile_extensions import compile_cython_extensions
        if compile_cython_extensions(verbose=False):
            # If compilation succeeded, try importing again
            try:
                from .cython_ext import lda_sampler
                CYTHON_AVAILABLE = True
            except ImportError:
                CYTHON_AVAILABLE = False
        else:
            CYTHON_AVAILABLE = False
    except ImportError:
        CYTHON_AVAILABLE = False

# Issue a warning only once at import time if Cython is not available
if not CYTHON_AVAILABLE:
    warnings.warn(
        "Using pure Python LDA implementation (Cython extensions not available). "
        "This will be significantly slower than the optimized version. ",
        RuntimeWarning
    )

class LDAGibbsSampler:
    """
    Latent Dirichlet Allocation with Gibbs sampling implementation.
    
    This implementation is designed to be modular, with the core sampling logic 
    isolated to enable future optimization with Cython.
    
    Attributes:
        n_topics (int): Number of topics
        alpha (float): Dirichlet prior for document-topic distributions
        beta (float): Dirichlet prior for topic-word distributions
        iterations (int): Number of Gibbs sampling iterations
        random_state (int, optional): Random seed for reproducibility
        eval_interval (int, optional): Perform evaluation every eval iterations
        min_count (int): Minimum count of word to be included in vocabulary
        max_vocab_size (int): Maximum vocabulary size to keep
        min_length (int): Minimum length of word to be included in vocabulary
        stopwords (set): Set of words to exclude from vocabulary
    """
    
    def __init__(
        self,
        n_topics: int = 10,
        alpha: Optional[float] = None,  # Changed to None as default to use 50/n_topics heuristic
        beta: float = 0.1,
        iterations: int = 1000,
        random_state: Optional[int] = None,
        eval_interval: Optional[int] = None,
        min_count: int = 1,
        max_vocab_size: Optional[int] = None,
        min_length: int = 1,
        stopwords: Optional[set] = None
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
            eval_interval: Perform evaluation every eval iterations
            min_count: Minimum count of word to be included in vocabulary
            max_vocab_size: Maximum vocabulary size to keep
            min_length: Minimum length of word to be included in vocabulary
            stopwords: Set of words to exclude from vocabulary
        """
        self.n_topics = n_topics
        # Use Griffiths and Steyvers (2004) heuristic if alpha is None
        if alpha is None:
            self.alpha = 50.0 / n_topics
        else:
            self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.random_state = random_state
        self.eval_interval = eval_interval
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.min_length = min_length
        self.stopwords = set() if stopwords is None else set(stopwords)
        
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
    
    def run_gibbs_sampling(self) -> None:
        """
        Run Gibbs sampling for the specified number of iterations. 
        
        Uses Cython if available.
        """
        n_docs = len(self.docs_tokens)
        
        if CYTHON_AVAILABLE:
            print(f"Running Gibbs sampling for {self.iterations} iterations with Cython.")
        else:
            print(f"Running Gibbs sampling for {self.iterations} iterations (no Cython available).")
        
        progress_bar = trange(self.iterations)
        for it in range(self.iterations):
            start_time = time.time()
            
            if CYTHON_AVAILABLE:
                # Use the optimized Cython implementation
                self.z = lda_sampler.run_iteration(
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
            
            progress_bar.update(1)
            # Calculate perplexity every eval iterations
            if it > 0 and self.eval_interval and it % self.eval_interval == 0:
                elapsed = time.time() - start_time
                self._update_distributions()
                perplexity = self.perplexity()
                tokens_per_sec = sum(len(doc) for doc in self.docs_tokens) / elapsed
                print(f"Iteration {it}: Perplexity = {perplexity:.2f}, Tokens/sec = {tokens_per_sec:.1f}")
    
    def _sample_topic(self, d: int, i: int, w: int) -> int:
        """
        Sample a new topic for word w in document d at position i.
        
        Uses Cython if available.
        
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
        self.theta = (self.n_dt + self.alpha) / (doc_lengths + self.n_topics * self.alpha)
        
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
        
        # Calculate document-topic distribution
        theta_doc = (n_dt_doc + self.alpha) / (len(filtered_doc) + self.n_topics * self.alpha)
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
        
        model = cls(
            n_topics=model_data['n_topics'],
            alpha=model_data['alpha'],
            beta=model_data['beta'],
            min_length=model_data.get('min_length', 1),  # Default to 1 for backward compatibility
            stopwords=set(model_data.get('stopwords', [])) if model_data.get('stopwords') else None
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

# Add export of the class
__all__ = ['LDAGibbsSampler']
