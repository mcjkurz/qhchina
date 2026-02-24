"""
Utility functions for the qhchina package.

This module provides common utilities used across multiple modules.
"""

import logging
from collections.abc import Generator, Iterable
from typing import Any

import numpy as np

logger = logging.getLogger("qhchina.utils")


__all__ = [
    'LineSentenceFile',
    'validate_filters',
    'apply_p_value_correction',
    'iter_batches',
    'build_vocab_from_iter',
    'VALID_CORRECTIONS',
]


class LineSentenceFile:
    """
    Restartable iterable that streams sentences from a text file.
    
    Enables memory-efficient training on large corpora by reading sentences 
    directly from disk. File format is one sentence per line, with tokens 
    separated by spaces.
    
    Args:
        filepath: Path to the corpus file.
        limit: Maximum number of sentences to read. None reads the entire file.
    
    Attributes:
        filepath: Path to the corpus file.
        limit: Maximum number of sentences, or None for unlimited.
        sentence_count: Number of sentences (respects *limit*).
        token_count: Total number of tokens (respects *limit*).
    
    Example:
        reader = LineSentenceFile("corpus.txt")
        for sentence in reader:
            print(sentence)
        
        # Read only the first 1000 sentences
        reader = LineSentenceFile("corpus.txt", limit=1000)
    """
    
    def __init__(self, filepath: str, limit: int | None = None):
        self.filepath = filepath
        self.limit = limit
        self.sentence_count, self.token_count = self._count_file()
    
    def _count_file(self) -> tuple[int, int]:
        """Count sentences and tokens by iterating through the file once."""
        sentence_count = 0
        token_count = 0
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                if line:
                    sentence_count += 1
                    token_count += len(line.split(' '))
                    if self.limit is not None and sentence_count >= self.limit:
                        break
        return sentence_count, token_count
    
    def __iter__(self):
        """Yield sentences one at a time."""
        count = 0
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                if line:
                    yield line.split(' ')
                    count += 1
                    if self.limit is not None and count >= self.limit:
                        return
    
    def __len__(self) -> int:
        """Return the number of sentences in the file."""
        return self.sentence_count
    
    def __repr__(self) -> str:
        limit_str = f", limit={self.limit}" if self.limit is not None else ""
        return (
            f"LineSentenceFile({self.filepath!r}, "
            f"sentences={self.sentence_count:,}, tokens={self.token_count:,}{limit_str})"
        )


VALID_CORRECTIONS = ('bonferroni', 'fdr_bh')


def apply_p_value_correction(
    p_values: list[float],
    method: str,
) -> np.ndarray:
    """
    Apply multiple testing correction to a list of p-values.
    
    Args:
        p_values: List or array of raw p-values.
        method: Correction method. Options:
            - 'bonferroni': Bonferroni correction (controls family-wise error rate).
              Adjusted p = p * n_tests, capped at 1.0.
            - 'fdr_bh': Benjamini-Hochberg procedure (controls false discovery rate).
              Generally less conservative than Bonferroni.
    
    Returns:
        numpy array of adjusted p-values (same length as input).
    
    Raises:
        ValueError: If method is not recognized or p_values is empty.
    """
    if method not in VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction method '{method}'. "
            f"Valid methods are: {VALID_CORRECTIONS}"
        )
    
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    
    if n == 0:
        return p_arr
    
    if method == 'bonferroni':
        adjusted = np.minimum(p_arr * n, 1.0)
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(p_arr)
        sorted_p = p_arr[sorted_indices]
        
        # Compute adjusted p-values: p_adjusted[i] = p[i] * n / rank
        ranks = np.arange(1, n + 1, dtype=np.float64)
        adjusted_sorted = sorted_p * n / ranks
        
        # Enforce monotonicity (working backwards from largest rank)
        # Each adjusted p cannot be larger than the one at the next higher rank
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        
        # Cap at 1.0
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        
        # Restore original order
        adjusted = np.empty(n, dtype=np.float64)
        adjusted[sorted_indices] = adjusted_sorted
    
    return adjusted


def validate_filters(
    filters: dict[str, Any] | None,
    valid_keys: set[str],
    context: str = "function"
) -> None:
    """
    Validate that all filter keys are recognized.
    
    Args:
        filters: Dictionary of filter parameters to validate.
        valid_keys: Set of valid/recognized filter keys.
        context: String describing the calling context for error messages.
    
    Raises:
        ValueError: If filters contains unrecognized keys.
    
    Example:
        validate_filters(
        ...     {'min_count': 5, 'max_p': 0.05, 'invalid_key': 'value'},
        ...     {'min_count', 'max_p', 'stopwords'},
        ...     context='compare_corpora'
        ... )
        ValueError: Unknown filter keys in compare_corpora: {'invalid_key'}. 
                    Valid keys are: {'max_p', 'min_count', 'stopwords'}
    """
    if filters is None:
        return
    
    if not isinstance(filters, dict):
        raise TypeError(f"filters must be a dictionary, got {type(filters).__name__}")
    
    unknown_keys = set(filters.keys()) - valid_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown filter keys in {context}: {unknown_keys}. "
            f"Valid keys are: {valid_keys}"
        )


def iter_batches(
    texts: Iterable[list[str]],
    batch_words: int = 100_000,
    max_length: int | None = 256,
) -> Generator[list[list[str]], None, None]:
    """
    Yield batches of tokenized texts grouped by total token count.
    
    Streams through the iterable without materializing the full corpus.
    Texts longer than *max_length* are truncated; empty texts are skipped.
    
    Args:
        texts: Iterable of tokenized texts (sentences or documents).
        batch_words: Target token count per batch.
        max_length: Truncate texts longer than this. None disables truncation.
    
    Yields:
        list[list[str]]: Batches where total tokens <= *batch_words*.
    """
    batch: list[list[str]] = []
    word_count = 0
    validated = False
    
    for text in texts:
        if not validated and text:
            if not isinstance(text, list):
                raise ValueError(
                    "input must be an iterable of lists (tokenized texts), "
                    f"but got an iterable of {type(text).__name__}"
                )
            validated = True
        if not text:
            continue
        if max_length is not None:
            text = text[:max_length]
        text_len = len(text)
        
        if word_count + text_len <= batch_words:
            batch.append(text)
            word_count += text_len
        else:
            if batch:
                yield batch
            batch = [text]
            word_count = text_len
    
    if batch:
        yield batch


def build_vocab_from_iter(
    texts: Iterable[list[str]],
    max_length: int | None = 256,
) -> tuple:
    """
    Build vocabulary statistics by streaming through tokenized texts (pass 1 of 2).
    
    Collects word counts, document-frequency counts, and total text count
    in a single pass. Texts longer than *max_length* are truncated.
    
    Args:
        texts: Restartable iterable of tokenized texts (sentences or documents).
        max_length: Truncate texts longer than this. None disables truncation.
    
    Returns:
        tuple: (word_counts, doc_counts, n_texts)
            - word_counts (Counter): Total token counts across all texts.
            - doc_counts (Counter): Number of texts each word appears in.
            - n_texts (int): Total number of non-empty texts.
    
    Raises:
        ValueError: If the iterable is empty or yields only empty texts.
    """
    from collections import Counter
    
    word_counts: Counter = Counter()
    doc_counts: Counter = Counter()
    n_texts = 0
    total_seen = 0
    validated = False
    
    for text in texts:
        total_seen += 1
        if not validated and text:
            if not isinstance(text, list):
                raise ValueError(
                    "input must be an iterable of lists (tokenized texts), "
                    f"but got an iterable of {type(text).__name__}"
                )
            validated = True
        if not text:
            continue
        if max_length is not None:
            text = text[:max_length]
        n_texts += 1
        word_counts.update(text)
        doc_counts.update(set(text))
    
    if total_seen == 0:
        raise ValueError("input cannot be empty")
    if n_texts == 0:
        raise ValueError("all input texts are empty")
    
    return word_counts, doc_counts, n_texts
