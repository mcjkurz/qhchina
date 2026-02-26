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
import pickle
import numpy as np
from collections import Counter
from collections.abc import Iterable
from .word2vec_base import Word2Vec
from .word2vec_utils import BalancedSentenceIterator, word2vec_c
from ..vectors import cosine_similarity

logger = logging.getLogger("qhchina.analytics.tempref_word2vec")

__all__ = [
    'TempRefWord2Vec',
]


def _filter_words(
    words: list[str],
    filters: dict | None,
    labels: list[str],
    period_vocab_counts: dict,
) -> list[str]:
    """
    Filter a list of candidate words according to the given filter criteria.

    Args:
        words: Candidate words to filter.
        filters: Dict with zero or more of the following keys:
            - ``min_word_count`` (int or tuple): Minimum corpus count per period.
            - ``min_word_length`` (int): Minimum character length.
            - ``stopwords`` (set): Words to exclude outright.
        labels: Ordered list of period labels (must match period_vocab_counts keys).
        period_vocab_counts: Dict mapping period label -> Counter of raw word counts.

    Returns:
        Filtered list of words (same order as input).

    Raises:
        ValueError: If unknown filter keys are supplied or tuple length mismatches.
        TypeError: If ``min_word_count`` is not an int or tuple.
    """
    if not filters:
        return words

    valid_keys = {'min_word_count', 'min_word_length', 'stopwords', 'reference_words', 'vocab_top_n'}
    bad_keys = set(filters) - valid_keys
    if bad_keys:
        raise ValueError(f"Unknown filter key(s): {bad_keys}. Valid keys: {valid_keys}")

    # vocab_top_n: restrict to the union of top-N most frequent words per period
    vocab_top_n = filters.get('vocab_top_n')
    if vocab_top_n is not None:
        if isinstance(vocab_top_n, int):
            top_ns = [vocab_top_n] * len(labels)
        elif isinstance(vocab_top_n, (tuple, list)):
            if len(vocab_top_n) != len(labels):
                raise ValueError(
                    f"vocab_top_n tuple length ({len(vocab_top_n)}) must equal "
                    f"the number of time slices ({len(labels)})"
                )
            top_ns = list(vocab_top_n)
        else:
            raise TypeError("vocab_top_n must be a positive int or a tuple/list of positive ints")
        if any(n <= 0 for n in top_ns):
            raise ValueError("All vocab_top_n values must be positive integers")
        top_n_set = set()
        for label, n in zip(labels, top_ns):
            counter = period_vocab_counts.get(label, {})
            top_n_set.update(w for w, _ in Counter(counter).most_common(n))
        words = [w for w in words if w in top_n_set]

    # reference_words: explicit whitelist (applied after vocab_top_n if both given)
    ref_words = filters.get('reference_words')
    if ref_words is not None:
        ref_set = set(ref_words)
        words = [w for w in words if w in ref_set]

    # Resolve min_word_count into a per-period list
    min_wc = filters.get('min_word_count')
    if min_wc is None:
        min_counts = None
    elif isinstance(min_wc, int):
        min_counts = [min_wc] * len(labels)
    elif isinstance(min_wc, (tuple, list)):
        if len(min_wc) != len(labels):
            raise ValueError(
                f"min_word_count tuple length ({len(min_wc)}) must equal "
                f"the number of time slices ({len(labels)})"
            )
        min_counts = list(min_wc)
    else:
        raise TypeError("min_word_count must be an int or a tuple/list of ints")

    min_len   = filters.get('min_word_length')
    stopwords = filters.get('stopwords') or set()

    result = []
    for word in words:
        if word in stopwords:
            continue
        if min_len is not None and len(word) < min_len:
            continue
        if min_counts is not None:
            if any(
                period_vocab_counts.get(label, {}).get(word, 0) < mc
                for label, mc in zip(labels, min_counts)
            ):
                continue
        result.append(word)
    return result


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
    
    There are two ways to train the model:
    1. Balanced batch sampling (default) - each batch contains equal numbers of
       tokens from each time period, ensuring fair representation regardless of
       corpus sizes.
    2. Proportional batch sampling - each batch contains a proportion of
       tokens from each time period; uses all data.
    
    Note:
        - Only supports Skip-gram (sg=1). CBOW is not supported.
        - Corpora must be UNTAGGED. Tagging is done automatically during training.
        - Training does not start automatically. Call ``train()`` explicitly after
          initialization.
    
    Interpreting Results:
        ``most_similar()`` on temporal variants (e.g., "民_宋") might return
        suboptimal results given that temporal variants are only trained on period-specificsubsets of data.
        Use ``calculate_semantic_change()`` instead, which compares similarity shifts
        across periods to reveal genuine semantic drift.
    
    Args:
        sentences: Dictionary mapping time period labels to corpora. Each value must
            be an iterable of tokenized sentences.
            Format: ``{"label1": [["w1", "w2"], ...], "label2": LineSentenceFile("song.txt"), ...}``
        targets: List of target words to trace semantic change.
        sampling_strategy: How to sample from corpora during training:
            - "balanced" (default): Equal tokens from each corpus, stops at smallest corpus.
            - "proportional": Proportional tokens from each corpus, uses all data.
        **kwargs: Arguments passed to Word2Vec. Common options:
            - vector_size (int): Dimensionality of word vectors (default: 100)
            - window (int): Context window size (default: 5)
            - min_word_count (int): Minimum word frequency (default: 5)
            - negative (int): Negative samples (default: 5)
            - epochs (int): Training epochs (required)
            - batch_size (int): Tokens per batch (default: 10240)
            - alpha (float): Initial learning rate (default: 0.025)
            - verbose (bool): Log progress (default: False)
            
            Note: sg must be 1 (Skip-gram).
    
    Example:
        # Using in-memory sentences:
        from qhchina.analytics import TempRefWord2Vec
        
        # Tokenized sentences from 宋史 and 明史 (untagged)
        song_sentences = [["太祖", "建隆", "元年", "正月"], ["民", "安", "其", "业"]]
        ming_sentences = [["太祖", "洪武", "元年", "春"], ["民", "困", "于", "役"]]
        
        model = TempRefWord2Vec(
            sentences={"宋": song_sentences, "明": ming_sentences},
            targets=["民", "太祖"],
            vector_size=100,
            sg=1
        )
        model.train()
        
        # Streaming from text files:
        from qhchina.analytics import TempRefWord2Vec, LineSentenceFile
        
        model = TempRefWord2Vec(
            sentences={
                "宋": LineSentenceFile("songshi.txt"),
                "明": LineSentenceFile("mingshi.txt"),
            },
            targets=["民", "太祖"],
            vector_size=100,
            sg=1
        )
        model.train()
        
        # Analyze semantic change
        changes = model.calculate_semantic_change("民")
        
        # most_similar on temporal variants is available, however each variant
        # has been only trained on a period-specific subset of data
        model.most_similar("民_宋")
        model.most_similar("民_明")
    """

    def __init__(
        self,
        sentences: dict[str, Iterable[list[str]]],
        targets: list[str],
        sampling_strategy: str = "balanced",
        _skip_init: bool = False,
        **kwargs
    ):
        """
        Initialize TempRefWord2Vec.
        
        Training does not start automatically. Call ``train()`` explicitly after
        initialization to begin training.
        
        Args:
            sentences: Dictionary mapping time period labels to corpora.
                Each value must be an iterable of tokenized sentences
                (e.g. list[list[str]] or ``LineSentenceFile``).
                Corpora must be untagged - tagging is done automatically during training.
            targets: List of target words to trace semantic change.
            sampling_strategy: How to sample from corpora during training.
                "balanced" (default) or "proportional".
            _skip_init: Internal. Used by load() to skip initialization.
            **kwargs: Arguments passed to Word2Vec (vector_size, window, etc.).
        """
        # _skip_init is used by load() to create an empty shell
        if _skip_init:
            self.labels = []
            self.targets = list(targets) if targets else []
            self._target_set = set(self.targets)
            self.period_vocab_counts = {}
            self.temporal_word_map = {}
            self.reverse_temporal_map = {}
            self._corpora = None
            self._sampling_strategy = "balanced"
            super().__init__(_skip_init=True, **kwargs)
            return
        
        # Validate sentences (must be dict for TempRefWord2Vec)
        if not isinstance(sentences, dict):
            raise TypeError(
                f"sentences must be a dictionary mapping labels to corpora, "
                f"got {type(sentences).__name__}"
            )
        
        if not sentences:
            raise ValueError("sentences cannot be empty")
        
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
                "For corpus-level shuffling, pre-shuffle your data files or in-memory lists."
            )
        
        # Validate sampling strategy
        if sampling_strategy not in ("balanced", "proportional"):
            raise ValueError(
                f"sampling_strategy must be 'balanced' or 'proportional', got {sampling_strategy!r}"
            )
        
        verbose = kwargs.get('verbose', False)
        
        # Store instance attributes
        self.labels = list(sentences.keys())
        self.targets = list(targets)
        self._target_set = set(targets)
        self._sampling_strategy = sampling_strategy
        
        self._corpora = {}
        for label, corpus in sentences.items():
            self._corpora[label] = corpus
        
        if verbose:
            logger.info("Loading corpora:")
            for label, corpus in self._corpora.items():
                if hasattr(corpus, 'sentence_count'):
                    logger.info(f"  {label}: {corpus.sentence_count:,} documents, {corpus.token_count:,} tokens (file)")
                else:
                    logger.info(f"  {label}: {len(corpus):,} documents (in-memory)")
        
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
        Build vocabulary by iterating through corpora.
        
        This override builds both period_vocab_counts and the main vocabulary
        in a single pass through the corpora, then adds temporal base words.
        
        Args:
            sentences: Ignored. TempRefWord2Vec uses internal corpora instead.
                Accepted for API compatibility with the parent class.
        """
        self._count_words()
        self._filter_and_map_vocab()
        self._add_temporal_base_words()
    
    def _count_words(self, sentences: Iterable[list[str]] | None = None) -> None:
        """
        Count word occurrences from corpora respecting the sampling strategy.
        
        With "balanced" strategy: counts only up to min_token_count per corpus.
        With "proportional" strategy: counts all tokens from each corpus.
        
        This ensures word frequencies, subsampling thresholds, and negative 
        sampling distributions match the actual training data.
        
        Target words are tagged with their corpus label during counting (e.g.,
        "bread" from corpus "1800s" is counted as "bread_1800s").
        
        Args:
            sentences: Ignored. TempRefWord2Vec uses internal corpora instead.
                Accepted for API compatibility with the parent class.
        
        Raises:
            ValueError: If corpora contain no words.
        """
        # Determine token counts for each corpus
        token_counts = {}
        for label, corpus in self._corpora.items():
            if hasattr(corpus, 'token_count'):
                token_counts[label] = corpus.token_count
            else:
                token_counts[label] = sum(len(sent) for sent in corpus)
        
        # Set token limits based on strategy
        if self._sampling_strategy == "balanced":
            min_count = min(token_counts.values())
            token_limits = {label: min_count for label in token_counts}
        else:  # proportional
            token_limits = token_counts
        
        self.period_vocab_counts = {}
        self.word_counts = Counter()
        
        for label, corpus in self._corpora.items():
            period_counter = Counter()
            tokens_counted = 0
            suffix = "_" + label
            token_limit = token_limits[label]
            
            for sentence in corpus:
                remaining = token_limit - tokens_counted
                if remaining <= 0:
                    break
                
                # Truncate sentence if it would exceed the limit
                if len(sentence) > remaining:
                    sentence = sentence[:remaining]
                
                # Tag target words with corpus label
                tagged_sentence = [
                    tok + suffix if tok in self._target_set else tok 
                    for tok in sentence
                ]
                
                period_counter.update(tagged_sentence)
                tokens_counted += len(sentence)
            
            self.period_vocab_counts[label] = period_counter
            self.word_counts.update(period_counter)
        
        if not self.word_counts:
            raise ValueError("Corpora contain no words.")
        
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
    
    def _build_alias_table(self) -> None:
        """Build alias table excluding temporal variants from negative sampling.
        
        Temporal variants (e.g., "民_宋") are CENTER words (in syn0), not CONTEXT words.
        Negative samples should come from the same distribution as positive targets
        (base forms in syn1neg). Temporal variants get zero weight so they are
        never drawn as negative samples.
        """
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            self._alias_prob = np.array([], dtype=np.uint32)
            self._alias_index = np.array([], dtype=np.uint32)
            return
        
        weights = np.array([
            0.0 if word in self.reverse_temporal_map
            else self.word_counts[word] ** self.ns_exponent
            for word in self.index2word
        ], dtype=np.float64)
        
        self._build_alias_from_weights(weights)
        
        n_excluded = sum(1 for w in self.index2word if w in self.reverse_temporal_map)
        logger.debug(f"Built alias table excluding {n_excluded} temporal variants from negative sampling")
    
    def _get_thread_working_mem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Allocate private work buffer for a worker thread.
        
        TempRefWord2Vec only needs the work buffer (Skip-gram only, no neu1
        or context_buffer needed). Returns (work, None, None) to match parent.
        """
        work = np.zeros(self.vector_size, dtype=self.dtype)
        return work, None, None
    
    def _train_batch_worker(
        self,
        batch: list[list[str]],
        sample_ints: np.ndarray,
        alpha: float,
        random_seed: int,
        calculate_loss: bool,
        work: np.ndarray,
        neu1: np.ndarray,
        context_buffer: np.ndarray,
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
            context_buffer: Unused (temporal training is Skip-gram only).
        
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
            self._alias_prob,
            self._alias_index,
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
        
        Uses balanced batch sampling from corpora. Each batch contains
        equal numbers of tokens from each time period, shuffled together.
        Target words are automatically tagged with their corpus label during training.
        
        All training configuration (epochs, batch_size, alpha, min_alpha, etc.) is read
        from instance attributes set during initialization via ``**kwargs``.
        
        Returns:
            Final loss value if calculate_loss is True, None otherwise.
        """
        # Build vocabulary if not already built
        if not self.vocab:
            self.build_vocab()
        
        # Create sentence iterator with automatic tagging
        training_corpus = BalancedSentenceIterator(
            self._corpora,
            token_budget=self.batch_size,
            targets=self._target_set,
            seed=self.seed,
            strategy=self._sampling_strategy
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

    def calculate_semantic_change(
        self,
        word: str,
        filters: dict | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Calculate semantic change by comparing cosine similarities across time periods.

        This is the recommended way to analyze temporal embeddings. It compares
        cosine similarity shifts for ``word`` across every adjacent pair of
        periods stored in the model (in the order they were provided at initialisation).

        Args:
            word: Target word to analyze (must be one of the targets specified
                during initialization).
            filters: Optional dict to restrict which reference words are considered.
                Supported keys:

                - ``min_word_count`` (int or tuple): Minimum number of occurrences a
                  word must have in **each** period to be included. Pass an int to use
                  the same threshold for every period, or a tuple whose length equals
                  the number of time slices to set per-period thresholds.
                - ``min_word_length`` (int): Minimum character length of a word.
                - ``stopwords`` (set): Words to exclude from the reference set.
                - ``reference_words`` (list/set): Explicit whitelist of reference words.
                - ``vocab_top_n`` (int): Only consider the union of the top-N most
                  frequent words from each period's corpus.

        Returns:
            Dict mapping transition names (e.g. ``"宋_to_明"``) to lists of
            ``(word, change)`` tuples sorted by descending change score.

        Example:
            changes = model.calculate_semantic_change(
                "人民",
                filters={"vocab_top_n": 5000, "min_word_length": 2},
            )
            for transition, word_changes in changes.items():
                print(f"{transition}:")
                print("Words moved towards:", word_changes[:5])
                print("Words moved away:", word_changes[-5:])
        """
        if word not in self.targets:
            raise ValueError(
                f"'{word}' was not specified as a target during model initialization. "
                f"Available targets: {self.targets}"
            )

        labels = self.labels  # always use the stored order

        # Candidate reference words: base forms only (no temporal variants)
        candidate_words = [w for w in self.vocab if w not in self.reverse_temporal_map]

        # Apply filters
        candidate_words = _filter_words(candidate_words, filters, labels, self.period_vocab_counts)

        if not candidate_words:
            raise ValueError("No reference words remain after applying filters.")

        results = {}
        for i in range(len(labels) - 1):
            from_period = labels[i]
            to_period   = labels[i + 1]
            transition  = f"{from_period}_to_{to_period}"

            from_variant = f"{word}_{from_period}"
            to_variant   = f"{word}_{to_period}"

            if from_variant not in self.vocab or to_variant not in self.vocab:
                logger.warning(
                    f"{from_variant} or {to_variant} not found in vocabulary. "
                    f"Skipping transition {transition}."
                )
                continue

            # Only include words that actually appear in both adjacent periods.
            # This ensures reference words are contextually grounded in the periods
            # being compared, preventing spurious similarity scores.
            from_counts = self.period_vocab_counts.get(from_period, {})
            to_counts   = self.period_vocab_counts.get(to_period, {})
            transition_words = [
                w for w in candidate_words
                if from_counts.get(w, 0) > 0 and to_counts.get(w, 0) > 0
            ]

            if not transition_words:
                results[transition] = []
                continue

            transition_vectors = np.array([self.get_vector(w) for w in transition_words])

            from_vector = self.get_vector(from_variant).reshape(1, -1)
            to_vector   = self.get_vector(to_variant).reshape(1, -1)

            from_sims = cosine_similarity(from_vector, transition_vectors)[0]
            to_sims   = cosine_similarity(to_vector,   transition_vectors)[0]
            sim_diffs = to_sims - from_sims

            word_changes = [(transition_words[j], float(sim_diffs[j])) for j in range(len(transition_words))]
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
            '_sampling_strategy': self._sampling_strategy,
            'model_type': 'TempRefWord2Vec'
        }
        
        # Combine all data
        model_data.update(tempref_data)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if this is a TempRefWord2Vec model
        if model_data.get('model_type') != 'TempRefWord2Vec':
            raise ValueError("The loaded file does not contain a TempRefWord2Vec model. "
                           "Use Word2Vec.load() for regular Word2Vec models.")
        
        # Extract TempRefWord2Vec-specific data
        labels = model_data['labels']
        targets = model_data['targets']
        if '_sampling_strategy' not in model_data:
            raise ValueError(
                "Invalid TempRefWord2Vec model format: missing '_sampling_strategy'. "
                "Re-save the model with the current qhchina version."
            )
        
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
        model._sampling_strategy = model_data['_sampling_strategy']
        
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