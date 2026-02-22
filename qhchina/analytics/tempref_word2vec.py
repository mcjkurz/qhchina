"""
TempRefWord2Vec: Word2Vec with Temporal Referencing for semantic change analysis.

This module provides the TempRefWord2Vec class which extends Word2Vec to track
how word meanings change across time periods.

This implementation is based on the Temporal Referencing approach described in:

    Haim Dubossarsky, Simon Hengchen, Nina Tahmasebi, and Dominik Schlechtweg. 2019.
    Time-Out: Temporal Referencing for Robust Modeling of Lexical Semantic Change.
    In Proceedings of the 57th Annual Meeting of the Association for Computational
    Linguistics, pages 457–470, Florence, Italy. Association for Computational Linguistics.
"""

import logging
import numpy as np
from collections import Counter
from collections.abc import Iterable
from .word2vec import Word2Vec
from .word2vec_utils import LineSentenceFile, BalancedSentenceIterator, word2vec_c
from .vectors import cosine_similarity

logger = logging.getLogger("qhchina.analytics.tempref_word2vec")

__all__ = [
    'TempRefWord2Vec',
]


class TempRefWord2Vec(Word2Vec):
    """
    Word2Vec with Temporal Referencing (TR) for tracking semantic change.
    
    Implements temporal referencing where target words are tagged with time period
    indicators (e.g., "bread_1800s"). During training:
    
    - Temporal variants (e.g., "bread_1800s") are used as CENTER words in syn0 (W)
    - Base forms (e.g., "bread") are used as CONTEXT words in syn1neg (W_prime)
    - Negative samples are drawn from base forms only
    
    This design places temporal variant embeddings in W, making them directly
    comparable with each other and with regular words for semantic change analysis.
    
    Training uses balanced batch sampling - each batch contains equal numbers of
    documents from each time period, ensuring fair representation regardless of
    corpus sizes.
    
    Note:
        - Only supports Skip-gram (sg=1). CBOW is not supported.
        - Requires pre-tagged corpus files. Use ``TempRefCorpus`` to create them.
        - Training does NOT start automatically. Call ``train()`` explicitly after
          initialization.
    
    Args:
        sentences: Dictionary mapping time period labels to file paths.
            Files must be pre-tagged using ``TempRefCorpus``.
            Format: ``{"label1": "path1.txt", "label2": "path2.txt", ...}``
        targets: List of target words to trace semantic change.
        **kwargs: Arguments passed to Word2Vec. Common options:
            - vector_size (int): Dimensionality of word vectors (default: 100)
            - window (int): Context window size (default: 5)
            - min_word_count (int): Minimum word frequency (default: 5)
            - negative (int): Negative samples (default: 5)
            - epochs (int): Training epochs (default: 1)
            - batch_size (int): Tokens per batch (default: 10240)
            - alpha (float): Initial learning rate (default: 0.025)
            - verbose (bool): Log progress (default: False)
            
            Note: sg must be 1 (Skip-gram).
    
    Example:
        from qhchina import TempRefCorpus
        from qhchina.analytics import TempRefWord2Vec
        
        # Step 1: Create tagged corpus files
        targets = ["bread", "food"]
        
        corpus1 = TempRefCorpus(label="1800s", targets=targets)
        corpus1.add_many(sentences_1800s)
        corpus1.shuffle()
        corpus1.save("1800s.txt")
        
        corpus2 = TempRefCorpus(label="1900s", targets=targets)
        corpus2.add_many(sentences_1900s)
        corpus2.shuffle()
        corpus2.save("1900s.txt")
        
        # Step 2: Initialize the model
        model = TempRefWord2Vec(
            sentences={"1800s": "1800s.txt", "1900s": "1900s.txt"},
            targets=targets,
            vector_size=100,
            sg=1
        )
        
        # Step 3: Train the model (explicit call required)
        model.train()
        
        # Step 4: Analyze semantic change
        model.most_similar("bread_1800s")  # Words similar to "bread" in 1800s
        model.most_similar("bread_1900s")  # Words similar to "bread" in 1900s
    """

    def __init__(
        self,
        sentences: dict[str, str],
        targets: list[str],
        _skip_init: bool = False,
        **kwargs
    ):
        """
        Initialize TempRefWord2Vec.
        
        Training does NOT start automatically. Call ``train()`` explicitly after
        initialization to begin training.
        
        Args:
            sentences: Dictionary mapping time period labels to file paths.
                Files must be pre-tagged using ``TempRefCorpus``.
            targets: List of target words to trace semantic change.
            _skip_init: Internal. Used by load() to skip initialization.
            **kwargs: Arguments passed to Word2Vec (vector_size, window, etc.).
        """
        # _skip_init is used by load() to create an empty shell
        if _skip_init:
            self.labels = []
            self.targets = targets
            self.period_vocab_counts = {}
            self.temporal_word_map = {}
            self.reverse_temporal_map = {}
            self._file_readers = None
            super().__init__(_skip_init=True, **kwargs)
            return
        
        # Validate sentences (must be dict for TempRefWord2Vec)
        if not isinstance(sentences, dict):
            raise TypeError(
                f"sentences must be a dictionary mapping labels to file paths, "
                f"got {type(sentences).__name__}"
            )
        
        if not sentences:
            raise ValueError("sentences cannot be empty")
        
        # Validate all values are file paths (strings)
        for label, path in sentences.items():
            if not isinstance(path, str):
                raise TypeError(
                    f"sentences values must be file paths (str), got {type(path).__name__} "
                    f"for label '{label}'. Use TempRefCorpus to create tagged corpus files."
                )
        
        # Check sg=1 requirement
        if kwargs.get('sg') != 1:
            raise NotImplementedError("TempRefWord2Vec only supports Skip-gram (sg=1)")
        
        # Validate targets
        if not targets:
            raise ValueError("targets cannot be empty. Provide at least one target word for temporal tracking.")
        
        # Reject shuffle=True (balanced iterator shuffles within batches)
        if kwargs.get('shuffle') is True:
            raise ValueError(
                "shuffle=True is not supported for TempRefWord2Vec. "
                "Sentences are shuffled within each batch by the balanced iterator. "
                "For corpus-level shuffling, pre-shuffle your data files."
            )
        
        verbose = kwargs.get('verbose', False)
        
        # Store instance attributes
        self.labels = list(sentences.keys())
        self.targets = targets
        
        # Create file readers
        self._file_readers = {
            label: LineSentenceFile(path) for label, path in sentences.items()
        }
        
        if verbose:
            logger.info("Loading corpus files:")
            for label, reader in self._file_readers.items():
                logger.info(f"  {label}: {reader.sentence_count:,} documents, {reader.token_count:,} tokens")
        
        # Build temporal word mappings
        self.temporal_word_map = {
            target: [f"{target}_{label}" for label in self.labels]
            for target in targets
        }
        self.reverse_temporal_map = {
            variant: base_word
            for base_word, variants in self.temporal_word_map.items()
            for variant in variants
        }
        
        # Initialize parent (without auto-training)
        super().__init__(**kwargs)
    
    def build_vocab(self, sentences: Iterable[list[str]] | None = None) -> None:
        """
        Build vocabulary by streaming through corpus files.
        
        This override builds both period_vocab_counts and the main vocabulary
        in a single pass through the files, then adds temporal base words.
        
        Args:
            sentences: Ignored. TempRefWord2Vec uses internal file readers instead.
                Accepted for API compatibility with the parent class.
        """
        self._count_words()
        self._filter_and_map_vocab()
        self._add_temporal_base_words()
    
    def _count_words(self, sentences: Iterable[list[str]] | None = None) -> None:
        """
        Count word occurrences from corpus files with balanced token limits.
        
        Streams through files, counting only up to min_token_count per corpus
        to match the balanced training data. This ensures word frequencies,
        subsampling thresholds, and negative sampling distributions are computed
        correctly for balanced training.
        
        Args:
            sentences: Ignored. TempRefWord2Vec uses internal file readers instead.
                Accepted for API compatibility with the parent class.
        
        Raises:
            ValueError: If corpus files contain no words.
        """
        min_token_count = min(reader.token_count for reader in self._file_readers.values())
        
        self.period_vocab_counts = {}
        self.word_counts = Counter()
        
        for label, reader in self._file_readers.items():
            period_counter = Counter()
            tokens_counted = 0
            
            for sentence in reader:
                remaining = min_token_count - tokens_counted
                if remaining <= 0:
                    break
                
                # Truncate sentence if it would exceed the limit
                if len(sentence) > remaining:
                    sentence = sentence[:remaining]
                
                period_counter.update(sentence)
                tokens_counted += len(sentence)
            
            self.period_vocab_counts[label] = period_counter
            self.word_counts.update(period_counter)
        
        if not self.word_counts:
            raise ValueError("Corpus files contain no words.")
        
        if self.verbose:
            for label, counter in self.period_vocab_counts.items():
                logger.info(
                    f"  {label}: {len(counter):,} unique tokens, "
                    f"{sum(counter.values()):,} total tokens"
                )
    
    def _add_temporal_base_words(self):
        """
        Add base words to vocabulary with counts derived from their temporal variants.
        
        Base words serve as CONTEXT words (positive targets in syn1neg) during training,
        while temporal variants serve as CENTER words (in syn0). The base word's
        effective frequency equals the sum of all its variants, ensuring proper
        negative sampling probability.
        
        Reference: Dubossarsky et al. (2019) "Time-Out: Temporal Referencing for Robust 
        Modeling of Lexical Semantic Change" - base forms provide stable reference frame.
        """
        # Verify all temporal variants are in the vocabulary
        missing_variants = []
        for base_word, variants in self.temporal_word_map.items():
            for variant in variants:
                if variant not in self.vocab:
                    missing_variants.append(variant)
        
        if missing_variants:
            logger.warning(f"{len(missing_variants)} temporal variants not found in corpus:")
            logger.warning(f"Sample: {missing_variants[:10]}")
            logger.warning("These variants will not be part of the temporal analysis.")
        
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
                # Base word already in vocab - update its count
                self.word_counts[base_word] = base_count
        
        if skipped_base_words:
            logger.warning(f"Skipped {len(skipped_base_words)} base words with no variant counts: {skipped_base_words[:5]}...")
        
        if added_base_words > 0:
            self.corpus_word_count += total_base_count
        
        # Build the temporal index map for Cython acceleration
        self._build_temporal_index_map()
    
    def _build_temporal_index_map(self) -> None:
        """
        Build a numpy array for fast temporal index mapping in Cython.
        
        The array maps word indices to their base form indices:
        - For temporal variants: temporal_index_map[variant_idx] = base_word_idx
        - For regular words: temporal_index_map[word_idx] = word_idx (identity)
        
        During training, CENTER words are temporal variants (in syn0), and this map
        is applied to get the CONTEXT words (base forms as positive targets in syn1neg).
        For example, "民_宋" (center, in syn0) predicts "民" (context, in syn1neg).
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
    
    def _build_cum_table(self) -> np.ndarray:
        """
        Build cumulative table for negative sampling, excluding temporal variants.
        
        Temporal variants (e.g., "民_宋") are CENTER words (in syn0), not CONTEXT words.
        Negative samples should come from the same distribution as positive targets
        (base forms in syn1neg). Including temporal variants in negative sampling would
        sample words whose syn1neg embeddings are never trained as positive targets.
        
        This override gives temporal variants zero probability by assigning them
        the same cumulative value as the previous word (zero-width range).
        
        Returns:
            np.ndarray[uint32]: Cumulative distribution table
        """
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            self._cum_table = np.array([], dtype=np.uint32)
            return self._cum_table
        
        domain = 2**31 - 1
        self._cum_table = np.zeros(vocab_size, dtype=np.uint32)
        
        # Compute Z = sum of count^exponent for eligible words only
        train_words_pow = 0.0
        for word in self.index2word:
            if word in self.reverse_temporal_map:
                continue  # Skip temporal variants
            count = self.word_counts[word]
            train_words_pow += count ** self.ns_exponent
        
        # Build cumulative table
        # Temporal variants get the same value as the previous entry (zero-width range)
        cumulative = 0.0
        for word_index, word in enumerate(self.index2word):
            if word not in self.reverse_temporal_map:
                count = self.word_counts[word]
                cumulative += count ** self.ns_exponent
            self._cum_table[word_index] = round(cumulative / train_words_pow * domain)
        
        # Verify final value equals domain
        if vocab_size > 0:
            assert self._cum_table[-1] == domain, f"Final cum_table value {self._cum_table[-1]} != {domain}"
        
        n_excluded = len([w for w in self.index2word if w in self.reverse_temporal_map])
        logger.debug(f"Built cum_table excluding {n_excluded} temporal variants from negative sampling")
        
        return self._cum_table
    
    def _get_thread_working_mem(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Allocate private work buffer for a worker thread.
        
        TempRefWord2Vec only needs the work buffer (Skip-gram only, no neu1 needed).
        Returns (work, None) to match parent signature.
        """
        work = np.zeros(self.vector_size, dtype=self.dtype)
        return work, None
    
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
        Train on a single batch using temporal-aware training with provided work buffers.
        
        This method is called by worker threads and uses thread-private buffers.
        
        Args:
            batch: List of tokenized sentences.
            sample_ints: Subsampling thresholds array.
            alpha: Learning rate for this batch.
            random_seed: Random seed for this batch (generated by producer thread).
            calculate_loss: Whether to compute loss.
            work: Thread-private work buffer.
            neu1: Unused (temporal training is Skip-gram only).
        
        Returns:
            Tuple of (batch_loss, batch_examples, batch_words).
        """
        if not hasattr(self, 'temporal_index_map') or self.temporal_index_map is None:
            self._build_temporal_index_map()
        
        batch_loss, batch_examples, batch_words, _ = word2vec_c.train_batch_temporal(
            self.W,
            self.W_prime,
            batch,
            self.vocab,
            self.temporal_index_map,
            sample_ints,
            self._cum_table,
            work,
            self.sample > 0,
            self.window,
            self.shrink_windows,
            alpha,
            self.negative,
            random_seed,
            calculate_loss,
        )
        return batch_loss, batch_examples, batch_words
    
    def train(self) -> float | None:
        """
        Train the TempRefWord2Vec model.
        
        Uses balanced batch sampling from corpus files. Each batch contains
        equal numbers of tokens from each time period, shuffled together.
        
        All training configuration (epochs, batch_size, alpha, min_alpha, etc.) is read
        from instance attributes set during initialization via ``**kwargs``.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        # Build vocabulary if not already built
        if not self.vocab:
            self.build_vocab()
        
        # Create balanced sentence iterator from file readers
        training_corpus = BalancedSentenceIterator(
            self._file_readers,
            token_budget=self.batch_size,
            seed=self.seed
        )
        
        # Set the training corpus and call the parent's train method
        # Note: Progress bar uses corpus_word_count which may exceed actual tokens
        # yielded by balanced iterator. This is acceptable - subsampling and negative
        # sampling use the correct vocabulary-based counts.
        self._sentences = training_corpus
        return super().train()

    def _is_temporal_variant(self, word: str) -> bool:
        """Check if a word is a temporal variant (e.g., '民_宋')."""
        return word in self.reverse_temporal_map

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
                print(f"{transition}:")
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
        model = cls(
            sentences={},  # Empty dict - won't be used with _skip_init
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
        )
        
        # Restore saved model state
        model.labels = labels
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
        
        if model.verbose:
            logger.info(f"TempRefWord2Vec model loaded from {path}")
            logger.info(f"Restored data includes:")
            logger.info(f"  - Vocabulary: {len(model.vocab)} words")
            logger.info(f"  - Time periods: {len(model.labels)} ({', '.join(model.labels)})")
            logger.info(f"  - Target words: {len(model.targets)} ({', '.join(model.targets)})")
            logger.info(f"  - Period vocab counts: {len(model.period_vocab_counts)} periods")
        
        return model