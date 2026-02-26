"""
Utility classes for Word2Vec training.

Provides balanced batch sampling for temporal training and Cython extension access.
"""

import logging
import numpy as np
from collections.abc import Iterable
from ...config import get_rng, resolve_seed

logger = logging.getLogger("qhchina.analytics.word2vec")

__all__ = [
    'BalancedSentenceIterator',
    'SingleCorpusTemporalIterator',
    'TemporalSentence',
    'TemporalSentenceIterator',
    'CYTHON_AVAILABLE',
    'word2vec_c',
]

# Cython extension - required for Word2Vec
try:
    from ..cython_ext import word2vec as word2vec_c
    CYTHON_AVAILABLE = True
except ImportError:
    raise ImportError(
        "Word2Vec requires Cython extensions which are not compiled. "
        "Please run: python setup.py build_ext --inplace"
    )


def _count_tokens(corpus: Iterable[list[str]]) -> int:
    """Count total tokens in a corpus by iterating through it once."""
    return sum(len(sentence) for sentence in corpus)


class TemporalSentence(list):
    """A sentence (list of tokens) annotated with a time slice index.

    Subclasses ``list`` so it passes through ``iter_batches`` and the parent
    ``Word2Vec`` training loop unchanged, while carrying an integer
    ``time_idx`` that the Cython kernel uses to select the correct
    ``U[t]`` / ``V[t]`` embedding slice.

    Args:
        tokens: The tokenized sentence.
        time_idx: Zero-based index of the time slice this sentence belongs to.
    """

    __slots__ = ('time_idx',)

    def __init__(self, tokens: list[str], time_idx: int):
        super().__init__(tokens)
        self.time_idx = time_idx


class TemporalSentenceIterator:
    """Streaming iterator that interleaves sentences from multiple corpora with time metadata.

    Follows the same balanced/proportional sampling logic as
    :class:`BalancedSentenceIterator` but yields :class:`TemporalSentence`
    objects (``list`` subclass with a ``time_idx`` attribute) instead of
    tagging words.  Only one budget-sized chunk is materialised at a time,
    so memory usage stays constant regardless of corpus size.

    Args:
        corpora: Dict mapping period labels to sentence iterables.
        label2idx: Dict mapping period labels to integer time-slice indices.
        token_budget: Tokens to collect per chunk before shuffling and yielding.
        seed: Random seed for reproducible chunk shuffling.
        strategy: ``"balanced"`` (equal tokens per corpus, stops at smallest)
            or ``"proportional"`` (all data, proportional contribution).

    Attributes:
        token_counts: Dict mapping labels to total token counts per corpus.
    """

    def __init__(
        self,
        corpora: dict[str, Iterable[list[str]]],
        label2idx: dict[str, int],
        token_budget: int,
        seed: int | None = None,
        strategy: str = "balanced",
    ):
        self._corpora = corpora
        self._label2idx = label2idx
        self._labels = list(corpora.keys())
        self._base_seed = resolve_seed(seed)
        self.token_budget = token_budget
        self._epoch = 0

        if strategy not in ("balanced", "proportional"):
            raise ValueError(f"strategy must be 'balanced' or 'proportional', got {strategy!r}")
        self._strategy = strategy

        self.token_counts = {
            label: corpus.token_count if hasattr(corpus, 'token_count') else _count_tokens(corpus)
            for label, corpus in corpora.items()
        }

        if strategy == "balanced":
            min_count = min(self.token_counts.values())
            self._token_limits = {label: min_count for label in self._labels}
        else:
            self._token_limits = self.token_counts.copy()

        self._total_tokens = sum(self._token_limits.values())

    def reset(self) -> None:
        """Reset epoch counter for reproducible iteration from the start."""
        self._epoch = 0

    def __iter__(self):
        """Yield :class:`TemporalSentence` objects from all corpora, interleaved and shuffled."""
        epoch_seed = (self._base_seed + self._epoch) if self._base_seed is not None else None
        rng = get_rng(epoch_seed)
        self._epoch += 1

        corpus_iters = {label: iter(corpus) for label, corpus in self._corpora.items()}
        tokens_yielded = {label: 0 for label in self._labels}
        corpora_exhausted: set[str] = set()

        num_corpora = len(self._labels)
        if self._strategy == "balanced":
            cycle_budgets = {label: self.token_budget // num_corpora for label in self._labels}
        else:
            cycle_budgets = {
                label: max(1, int(self.token_budget * self._token_limits[label] / self._total_tokens))
                for label in self._labels
            }

        total_yielded = 0

        while True:
            chunk: list[TemporalSentence] = []

            for label in self._labels:
                if label in corpora_exhausted:
                    continue

                time_idx = self._label2idx[label]
                corpus_iter = corpus_iters[label]
                cycle_tokens = 0
                cycle_budget = cycle_budgets[label]

                while cycle_tokens < cycle_budget:
                    remaining = self._token_limits[label] - tokens_yielded[label]
                    if remaining <= 0:
                        corpora_exhausted.add(label)
                        break

                    try:
                        sentence = next(corpus_iter)
                        if len(sentence) > remaining:
                            sentence = sentence[:remaining]

                        chunk.append(TemporalSentence(sentence, time_idx))
                        sentence_tokens = len(sentence)
                        cycle_tokens += sentence_tokens
                        tokens_yielded[label] += sentence_tokens
                    except StopIteration:
                        corpora_exhausted.add(label)
                        break

            if chunk:
                rng.shuffle(chunk)
                for sent in chunk:
                    total_yielded += len(sent)
                    yield sent

            if self._strategy == "balanced":
                if corpora_exhausted:
                    return
            else:
                if len(corpora_exhausted) == num_corpora:
                    return

            if total_yielded >= self._total_tokens:
                return

            if not chunk:
                return


class SingleCorpusTemporalIterator:
    """Streaming iterator over a single corpus that yields :class:`TemporalSentence` objects.

    Used by :class:`DynamicWord2Vec` in sequential training mode, where each
    time slice is trained independently.  Streams through the corpus without
    materialising it, respecting an optional token limit.

    Args:
        corpus: Sentence iterable for one period.
        time_idx: Integer time-slice index to attach to every sentence.
        token_limit: Maximum tokens to yield (``None`` = unlimited).
    """

    def __init__(
        self,
        corpus: Iterable[list[str]],
        time_idx: int,
        token_limit: int | None = None,
    ):
        self._corpus = corpus
        self._time_idx = time_idx
        self._token_limit = token_limit

    def __iter__(self):
        tokens_yielded = 0
        for sentence in self._corpus:
            if self._token_limit is not None:
                remaining = self._token_limit - tokens_yielded
                if remaining <= 0:
                    return
                if len(sentence) > remaining:
                    sentence = sentence[:remaining]
            yield TemporalSentence(sentence, self._time_idx)
            tokens_yielded += len(sentence)


class BalancedSentenceIterator:
    """
    Iterator that streams sentences from multiple corpus sources with configurable sampling.
    
    Used by TempRefWord2Vec for training. Collects sentences from each corpus 
    until a token budget is reached, shuffles them, then yields the sentences.
    
    When ``targets`` is provided, target words are automatically tagged with their
    corpus label (e.g., "民" from corpus "宋" becomes "民_宋").
    
    Args:
        corpora: Dictionary mapping labels to sentence iterables (LineSentenceFile 
            or list[list[str]]).
        token_budget: Target number of tokens to collect before yielding.
        targets: Set of target words to tag with corpus labels.
        seed: Random seed for reproducible sentence shuffling.
        strategy: Sampling strategy - "balanced" or "proportional".
            - "balanced": Each corpus contributes equal tokens (stops at smallest corpus).
            - "proportional": Each corpus contributes proportionally to its size (uses all data).
    
    Attributes:
        token_counts: Dictionary mapping labels to token counts for each corpus.
    """
    
    def __init__(
        self, 
        corpora: dict[str, Iterable[list[str]]], 
        token_budget: int,
        targets: set[str] | None = None,
        seed: int | None = None,
        strategy: str = "balanced"
    ):
        self._corpora = corpora
        self._base_seed = resolve_seed(seed)
        self._labels = list(corpora.keys())
        self.token_budget = token_budget
        self._targets = targets if targets else None
        self._epoch = 0
        
        if strategy not in ("balanced", "proportional"):
            raise ValueError(f"strategy must be 'balanced' or 'proportional', got {strategy!r}")
        self._strategy = strategy
        
        # Count tokens for each corpus
        self.token_counts = {
            label: corpus.token_count if hasattr(corpus, 'token_count') else _count_tokens(corpus)
            for label, corpus in corpora.items()
        }
        
        # Set token limits based on strategy
        if strategy == "balanced":
            min_count = min(self.token_counts.values())
            self._token_limits = {label: min_count for label in self._labels}
        else:  # proportional
            self._token_limits = self.token_counts.copy()
        
        self._total_tokens = sum(self._token_limits.values())
    
    def reset(self) -> None:
        """Reset epoch counter to 0 for reproducible iteration from the start."""
        self._epoch = 0
    
    def _tag_sentence(self, sentence: list[str], label: str) -> list[str]:
        """Tag target words in a sentence with the corpus label."""
        suffix = "_" + label
        return [tok + suffix if tok in self._targets else tok for tok in sentence]
    
    def __iter__(self):
        """Yield sentences from all corpora in a shuffled manner."""
        if self._base_seed is not None:
            epoch_seed = self._base_seed + self._epoch
        else:
            epoch_seed = None
        rng = get_rng(epoch_seed)
        self._epoch += 1
        
        corpus_iters = {label: iter(corpus) for label, corpus in self._corpora.items()}
        tokens_yielded = {label: 0 for label in self._labels}
        corpora_exhausted = set()
        
        # Calculate per-corpus budget for each collection cycle
        num_corpora = len(self._labels)
        if self._strategy == "balanced":
            cycle_budgets = {label: self.token_budget // num_corpora for label in self._labels}
        else:  # proportional
            cycle_budgets = {
                label: max(1, int(self.token_budget * self._token_limits[label] / self._total_tokens))
                for label in self._labels
            }
        
        # Safeguard: track total tokens yielded to prevent infinite loop
        total_yielded = 0
        max_total = self._total_tokens
        
        while True:
            collected_sentences = []
            
            for label in self._labels:
                if label in corpora_exhausted:
                    continue
                    
                corpus_iter = corpus_iters[label]
                cycle_tokens = 0
                cycle_budget = cycle_budgets[label]
                
                while cycle_tokens < cycle_budget:
                    remaining = self._token_limits[label] - tokens_yielded[label]
                    if remaining <= 0:
                        corpora_exhausted.add(label)
                        break
                    
                    try:
                        sentence = next(corpus_iter)
                        
                        if len(sentence) > remaining:
                            sentence = sentence[:remaining]
                        
                        if self._targets is not None:
                            sentence = self._tag_sentence(sentence, label)
                        
                        collected_sentences.append(sentence)
                        sentence_tokens = len(sentence)
                        cycle_tokens += sentence_tokens
                        tokens_yielded[label] += sentence_tokens
                    except StopIteration:
                        corpora_exhausted.add(label)
                        break
            
            if collected_sentences:
                rng.shuffle(collected_sentences)
                for sent in collected_sentences:
                    total_yielded += len(sent)
                    yield sent
            
            # Stop conditions
            if self._strategy == "balanced":
                # Stop when ANY corpus is exhausted
                if corpora_exhausted:
                    return
            else:  # proportional
                # Stop when ALL corpora are exhausted
                if len(corpora_exhausted) == num_corpora:
                    return
            
            # Safeguard: stop if we've yielded all expected tokens (prevents infinite loop)
            if total_yielded >= max_total:
                return
            
            # Safeguard: stop if no sentences were collected this cycle
            if not collected_sentences:
                return
