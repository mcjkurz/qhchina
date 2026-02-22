"""
Utility classes for Word2Vec training.

Provides file-based corpus streaming and balanced batch sampling for temporal training.
"""

import logging
import numpy as np
from ..config import get_rng, resolve_seed

logger = logging.getLogger("qhchina.analytics.word2vec")

__all__ = [
    'LineSentenceFile',
    'BalancedSentenceIterator',
    'CYTHON_AVAILABLE',
    'word2vec_c',
]

# Cython extension - required for Word2Vec
try:
    from .cython_ext import word2vec as word2vec_c
    CYTHON_AVAILABLE = True
except ImportError:
    raise ImportError(
        "Word2Vec requires Cython extensions which are not compiled. "
        "Please run: python setup.py build_ext --inplace"
    )


class LineSentenceFile:
    """
    Restartable iterable that streams sentences from a text file.
    
    This class enables memory-efficient training on large corpora by reading
    sentences directly from disk instead of loading everything into memory.
    
    File format:
        Line 1: sentence_count token_count [optional_params...]
        Line 2+: space-separated tokens (one sentence per line)
    
    The header must contain at least sentence_count and token_count as the first
    two space-separated values. Additional values may be present and are ignored.
    
    The file should be pre-shuffled if random sentence order is desired during
    training, as sentences are read sequentially.
    
    Args:
        filepath: Path to the corpus file.
    
    Attributes:
        filepath: Path to the corpus file.
        sentence_count: Number of sentences in the file (from header).
        token_count: Total number of tokens in the file (from header).
    
    Example:
        # Create a corpus file using Corpus
        corpus = Corpus(my_sentences)
        corpus.save("corpus.txt")
        
        # Use with Word2Vec
        model = Word2Vec("corpus.txt", vector_size=100)
        model.train()
        
        # Or iterate directly
        reader = LineSentenceFile("corpus.txt")
        for sentence in reader:
            print(sentence)
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.sentence_count, self.token_count = self._read_header()
    
    def _read_header(self) -> tuple[int, int]:
        """Read and parse the header line containing counts.
        
        The header must have at least two space-separated integers (sentence_count
        and token_count). Additional values are ignored for forward compatibility.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            parts = header.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid file header in {self.filepath}. "
                    f"Expected at least 'sentence_count token_count', got: {header!r}"
                )
            try:
                return int(parts[0]), int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid file header in {self.filepath}. "
                    f"First two values must be integers, got: {header!r}"
                ) from e
    
    def __iter__(self):
        """
        Yield sentences one at a time.
        
        Each call to __iter__ opens the file fresh, making this a restartable
        iterable suitable for multi-epoch training.
        
        Yields:
            list[str]: A sentence as a list of tokens.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            f.readline()  # Skip header
            for line in f:
                line = line.rstrip('\n\r')
                if line:
                    yield line.split(' ')
    
    def __len__(self) -> int:
        """Return the number of sentences in the file."""
        return self.sentence_count
    
    def __repr__(self) -> str:
        return (
            f"LineSentenceFile({self.filepath!r}, "
            f"sentences={self.sentence_count:,}, tokens={self.token_count:,})"
        )


class BalancedSentenceIterator:
    """
    Iterator that streams token-balanced sentences from multiple corpus sources.
    
    Used by TempRefWord2Vec for training. Reads from multiple LineSentenceFile 
    readers, collecting sentences from each corpus until a token budget is reached,
    then shuffles and yields the collected sentences.
    
    Token-based balancing ensures each corpus contributes roughly equal numbers of
    tokens per collection cycle (and overall), regardless of differences in average 
    sentence length. This is important for temporal referencing where periods should 
    have equal influence on the shared embedding space.
    
    Each call to ``__iter__`` increments an internal epoch counter, producing a 
    different shuffle order. Call ``reset()`` to restart from epoch 0.
    
    Args:
        readers: Dictionary mapping labels to LineSentenceFile instances.
        token_budget: Target number of tokens to collect before yielding. Each corpus 
            contributes approximately token_budget // num_corpora tokens per cycle.
        seed: Random seed for reproducible sentence shuffling.
    
    Attributes:
        token_count: Total tokens yielded per epoch (min_corpus_tokens * num_corpora).
    """
    
    def __init__(
        self, 
        readers: dict[str, LineSentenceFile], 
        token_budget: int,
        seed: int | None = None
    ):
        self.readers = readers
        self._base_seed = resolve_seed(seed)
        self._labels = list(readers.keys())
        self.token_budget = token_budget
        self._epoch = 0
        
        # Determine minimum token count across corpora for balanced training
        self._min_token_count = min(reader.token_count for reader in readers.values())
    
    def reset(self) -> None:
        """Reset epoch counter to 0 for reproducible iteration from the start."""
        self._epoch = 0
    
    def __iter__(self):
        """
        Yield sentences from all files in a token-balanced manner.
        
        Collects sentences from each corpus until the token budget is reached,
        shuffles them, then yields individual sentences. This process repeats
        until any corpus is exhausted.
        
        Each call creates fresh file iterators. The RNG seed is derived from
        base_seed + epoch_number, ensuring:
        - Deterministic shuffling (same seed = same results)
        - Different shuffle order each epoch
        
        Iteration stops when any corpus reaches the minimum token count,
        ensuring all corpora contribute equally.
        
        Yields:
            list[str]: Individual sentences (as token lists), yielded one at a time
                from shuffled collections.
        """
        epoch_seed = self._base_seed + self._epoch
        rng = get_rng(epoch_seed)
        self._epoch += 1
        
        file_iters = {label: iter(reader) for label, reader in self.readers.items()}
        
        # Track cumulative tokens yielded per corpus
        tokens_yielded = {label: 0 for label in self._labels}
        
        # Token budget per corpus per collection cycle
        num_corpora = len(self._labels)
        tokens_per_corpus = self.token_budget // num_corpora
        
        if tokens_per_corpus < 1:
            raise ValueError(
                f"token_budget ({self.token_budget}) must be at least {num_corpora} "
                f"(number of corpora)"
            )
        
        while True:
            collected_sentences = []
            corpora_exhausted = set()
            
            # Collect sentences from each corpus until per-corpus budget is reached
            for label in self._labels:
                file_iter = file_iters[label]
                cycle_tokens = 0
                
                while cycle_tokens < tokens_per_corpus:
                    # Check if this corpus has reached its total token limit
                    remaining_for_corpus = self._min_token_count - tokens_yielded[label]
                    if remaining_for_corpus <= 0:
                        corpora_exhausted.add(label)
                        break
                    
                    try:
                        sentence = next(file_iter)
                        
                        # Truncate sentence if it would exceed corpus token limit
                        if len(sentence) > remaining_for_corpus:
                            sentence = sentence[:remaining_for_corpus]
                        
                        collected_sentences.append(sentence)
                        sentence_tokens = len(sentence)
                        cycle_tokens += sentence_tokens
                        tokens_yielded[label] += sentence_tokens
                    except StopIteration:
                        corpora_exhausted.add(label)
                        break
            
            # Shuffle and yield collected sentences
            if collected_sentences:
                rng.shuffle(collected_sentences)
                yield from collected_sentences
            
            # Stop when ANY corpus is exhausted (ensures balanced training)
            if corpora_exhausted:
                return
