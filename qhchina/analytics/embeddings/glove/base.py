"""
GloVe implementation with Python API and Cython compute kernel.

This module follows the same API conventions as the existing embedding models:
- explicit ``train()`` call
- ``get_vector`` / ``most_similar`` / ``similarity`` inference methods
- pickle ``save()`` / ``load()`` model persistence

Co-occurrence construction supports two modes:
- ``in_memory``: accumulate weighted pair counts in RAM (fastest with enough memory)
- ``disk``: aggregate pair counts into sorted shard files and train by streaming
  k-way merge (lower RAM usage, no full matrix materialization)
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import heapq
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from ....config import resolve_seed
from ..._vector_ops import cosine_similarity
from ..word2vec.base import Word2Vec

try:
    from ...cython_ext import glove as glove_c
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    glove_c = None

logger = logging.getLogger("qhchina.analytics.embeddings.glove")

__all__ = ["GloVe", "CYTHON_AVAILABLE", "glove_c"]


class GloVe(Word2Vec):
    """
    Global Vectors (GloVe) model with sparse co-occurrence training.

    This implementation follows the same high-level API shape as ``Word2Vec``
    (explicit ``train()``, vector querying, pickle save/load), while optimizing
    a GloVe weighted least-squares objective with AdaGrad updates in Cython.

    Two co-occurrence backends are available:
    - ``mode="in_memory"``: fastest when memory is sufficient; accumulates all
      sparse pairs in RAM before epoch updates.
    - ``mode="disk"``: memory-bounded external aggregation; writes sorted shard
      files and trains from a streaming k-way merge without materializing the
      full sparse matrix.

    Parameters:
        sentences: Restartable iterable of tokenized sentences.
        vector_size: Embedding dimensionality.
        window: Symmetric context window radius.
        min_word_count: Minimum token frequency for vocabulary inclusion.
        max_vocab_size: Optional cap on retained vocabulary size.
        seed: RNG seed for deterministic initialization/training order.
        alpha: Learning rate for AdaGrad updates.
        min_alpha: Accepted for API compatibility (not used for internal
            GloVe decay schedule).
        epochs: Number of full passes over co-occurrence pairs.
        workers: Accepted for compatibility with Word2Vec API.
        verbose: If True, logs progress and backend details.
        calculate_loss: If True, ``train()`` returns average epoch loss.
        mode: ``"in_memory"`` or ``"disk"`` co-occurrence backend.
        x_max: GloVe weighting cutoff in ``f(x)``.
        power: GloVe weighting exponent in ``f(x)``.
        min_cooc_count: Drop co-occurrence pairs below this weight.
        shard_sentence_count: In disk mode, flush local pair map every N
            sentences.
        cooc_train_chunk_size: Number of merged pairs passed per Cython
            update chunk in disk mode.
        max_cooc_entries_in_memory: Safety cap for local pair-map size.
        combine_vectors: If True, expose ``(W + W_tilde)/2`` as ``self.W``;
            otherwise expose ``W`` only.

    Notes:
        * Base (single-corpus) GloVe only.
        * ``workers`` is accepted for API consistency, but training updates are
          currently executed as one shared update stream.
        * Vectors returned by ``get_vector`` and ``most_similar`` come from
          either ``(W + W_tilde) / 2`` (default) or ``W`` only when
          ``combine_vectors=False``.
    """

    def __init__(
        self,
        sentences: Iterable[list[str]] | None = None,
        vector_size: int = 100,
        window: int = 5,
        min_word_count: int = 5,
        max_vocab_size: int | None = None,
        seed: int | None = None,
        alpha: float = 0.05,
        min_alpha: float | None = None,
        epochs: int | None = None,
        workers: int = 1,
        verbose: bool = False,
        calculate_loss: bool = True,
        mode: str = "in_memory",
        x_max: float = 100.0,
        power: float = 0.75,
        min_cooc_count: float = 0.0,
        shard_sentence_count: int = 50000,
        cooc_train_chunk_size: int = 200000,
        max_cooc_entries_in_memory: int = 5_000_000,
        combine_vectors: bool = True,
        _skip_init: bool = False,
    ):
        """
        Initialize a GloVe model.

        Args:
            sentences: Restartable iterable of tokenized sentences.
            vector_size: Embedding dimensionality.
            window: Symmetric context window radius.
            min_word_count: Minimum token frequency for vocabulary inclusion.
            max_vocab_size: Optional cap on retained vocabulary size.
            seed: RNG seed for deterministic initialization/training order.
            alpha: Learning rate for AdaGrad updates.
            min_alpha: Accepted for API compatibility (not used for internal
                GloVe decay schedule).
            epochs: Number of full passes over co-occurrence pairs.
            workers: Accepted for compatibility with Word2Vec API.
            verbose: If True, logs progress and backend details.
            calculate_loss: If True, ``train()`` returns average epoch loss.
            mode: ``"in_memory"`` or ``"disk"`` co-occurrence backend.
            x_max: GloVe weighting cutoff in ``f(x)``.
            power: GloVe weighting exponent in ``f(x)``.
            min_cooc_count: Drop co-occurrence pairs below this weight.
            shard_sentence_count: In disk mode, flush local pair map every N
                sentences.
            cooc_train_chunk_size: Number of merged pairs passed per Cython
                update chunk in disk mode.
            max_cooc_entries_in_memory: Safety cap for local pair-map size.
            combine_vectors: If True, expose ``(W + W_tilde)/2`` as ``self.W``;
                otherwise expose ``W`` only.
            _skip_init: Internal flag used by ``load()``.
        """
        # Reuse Word2Vec validation + serialization layout where possible.
        super().__init__(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_word_count=min_word_count,
            negative=1,
            ns_exponent=0.75,
            cbow_mean=True,
            sg=1,
            seed=seed,
            alpha=alpha,
            min_alpha=min_alpha,
            sample=0.0,
            shrink_windows=False,
            max_vocab_size=max_vocab_size,
            verbose=verbose,
            epochs=epochs,
            batch_size=10240,
            workers=workers,
            callbacks=None,
            calculate_loss=calculate_loss,
            shuffle=True,
            _skip_init=_skip_init,
        )

        self.mode = mode
        self.x_max = x_max
        self.power = power
        self.min_cooc_count = min_cooc_count
        self.shard_sentence_count = shard_sentence_count
        self.cooc_train_chunk_size = cooc_train_chunk_size
        self.max_cooc_entries_in_memory = max_cooc_entries_in_memory
        self.combine_vectors = combine_vectors

        # Internal trainable parameters and AdaGrad state.
        self._W_input: np.ndarray | None = None
        self._W_context: np.ndarray | None = None
        self._bias_input: np.ndarray | None = None
        self._bias_context: np.ndarray | None = None
        self._grad_sq_input: np.ndarray | None = None
        self._grad_sq_context: np.ndarray | None = None
        self._grad_sq_bias_input: np.ndarray | None = None
        self._grad_sq_bias_context: np.ndarray | None = None

        self._validate_glove_hyperparameters()
        if not CYTHON_AVAILABLE:
            raise ImportError(
                "GloVe requires Cython extensions which are not compiled. "
                "Please run: ./venv/bin/python setup.py build_ext --inplace"
            )

    def _validate_glove_hyperparameters(self) -> None:
        if self.mode not in ("in_memory", "disk"):
            raise ValueError("mode must be 'in_memory' or 'disk'")
        if not isinstance(self.x_max, (int, float)) or self.x_max <= 0:
            raise ValueError("x_max must be a positive number")
        if not isinstance(self.power, (int, float)) or self.power <= 0 or self.power > 1:
            raise ValueError("power must be in the interval (0, 1]")
        if not isinstance(self.min_cooc_count, (int, float)) or self.min_cooc_count < 0:
            raise ValueError("min_cooc_count must be non-negative")
        if not isinstance(self.shard_sentence_count, int) or self.shard_sentence_count <= 0:
            raise ValueError("shard_sentence_count must be a positive integer")
        if (
            not isinstance(self.cooc_train_chunk_size, int)
            or self.cooc_train_chunk_size <= 0
        ):
            raise ValueError("cooc_train_chunk_size must be a positive integer")
        if (
            not isinstance(self.max_cooc_entries_in_memory, int)
            or self.max_cooc_entries_in_memory <= 0
        ):
            raise ValueError("max_cooc_entries_in_memory must be a positive integer")

    def _initialize_glove_params(self) -> None:
        """Initialize trainable matrices and AdaGrad accumulators."""
        vocab_size = len(self.vocab)
        init_rng = np.random.default_rng(seed=resolve_seed(self.seed))
        scale = 1.0 / max(self.vector_size, 1)

        self._W_input = (
            (init_rng.random((vocab_size, self.vector_size), dtype=np.float32) - 0.5) * scale
        ).astype(np.float32, copy=False)
        self._W_context = (
            (init_rng.random((vocab_size, self.vector_size), dtype=np.float32) - 0.5) * scale
        ).astype(np.float32, copy=False)

        self._bias_input = np.zeros(vocab_size, dtype=np.float32)
        self._bias_context = np.zeros(vocab_size, dtype=np.float32)

        # Start accumulators at 1 to avoid divide-by-zero and huge first updates.
        self._grad_sq_input = np.ones((vocab_size, self.vector_size), dtype=np.float32)
        self._grad_sq_context = np.ones((vocab_size, self.vector_size), dtype=np.float32)
        self._grad_sq_bias_input = np.ones(vocab_size, dtype=np.float32)
        self._grad_sq_bias_context = np.ones(vocab_size, dtype=np.float32)

        self._refresh_exposed_vectors()

    def _refresh_exposed_vectors(self) -> None:
        """Refresh public ``W`` / ``W_prime`` views after updates."""
        if self._W_input is None or self._W_context is None:
            return
        self.W_prime = self._W_context
        if self.combine_vectors:
            self.W = ((self._W_input + self._W_context) * 0.5).astype(np.float32, copy=False)
        else:
            self.W = self._W_input

    def _build_cooc_in_memory(
        self, sentences: Iterable[list[str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build weighted co-occurrence triplets in memory."""
        cooc: dict[tuple[int, int], float] = defaultdict(float)
        for sentence in sentences:
            if not sentence:
                continue
            mapped = [self.vocab[w] for w in sentence if w in self.vocab]
            sent_len = len(mapped)
            for center_pos, center_idx in enumerate(mapped):
                left = max(0, center_pos - self.window)
                right = min(sent_len, center_pos + self.window + 1)
                for ctx_pos in range(left, right):
                    if ctx_pos == center_pos:
                        continue
                    ctx_idx = mapped[ctx_pos]
                    distance = abs(center_pos - ctx_pos)
                    cooc[(center_idx, ctx_idx)] += 1.0 / float(distance)
            if len(cooc) > self.max_cooc_entries_in_memory:
                raise MemoryError(
                    "In-memory co-occurrence map exceeded max_cooc_entries_in_memory. "
                    "Use mode='disk' or increase the cap."
                )
        return self._cooc_dict_to_arrays(cooc)

    def _flush_cooc_shard(
        self,
        local: dict[int, float],
        shard_dir: str,
        shard_pairs: list[tuple[str, str]],
    ) -> None:
        """
        Write one sorted shard to disk and clear the local accumulator.

        The shard is stored as two aligned ``.npy`` files:
        - ``*_keys.npy``: int64 composite keys ``(i * vocab_size + j)``
        - ``*_vals.npy``: float32 aggregated co-occurrence weights
        """
        if not local:
            return
        n = len(local)
        keys = np.fromiter(local.keys(), dtype=np.int64, count=n)
        values = np.fromiter(local.values(), dtype=np.float32, count=n)
        order = np.argsort(keys)
        keys = keys[order]
        values = values[order]
        base = os.path.join(shard_dir, f"cooc_{len(shard_pairs):06d}")
        key_path = f"{base}_keys.npy"
        val_path = f"{base}_vals.npy"
        np.save(key_path, keys, allow_pickle=False)
        np.save(val_path, values, allow_pickle=False)
        shard_pairs.append((key_path, val_path))
        local.clear()

    def _build_cooc_disk(
        self, sentences: Iterable[list[str]]
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Build sorted co-occurrence shards and return ``(shard_dir, shard_files)``.

        Shards contain aligned ``keys`` and ``values`` arrays sorted by key,
        where ``key = center_idx * vocab_size + context_idx``.
        """
        shard_dir = tempfile.mkdtemp(prefix="qhchina_glove_shards_")
        local: dict[int, float] = defaultdict(float)
        shard_pairs: list[tuple[str, str]] = []
        seen_sentences = 0
        vocab_size = len(self.vocab)
        vs = int(vocab_size)

        try:
            for sentence in sentences:
                if not sentence:
                    continue
                mapped = [self.vocab[w] for w in sentence if w in self.vocab]
                sent_len = len(mapped)
                for center_pos, center_idx in enumerate(mapped):
                    left = max(0, center_pos - self.window)
                    right = min(sent_len, center_pos + self.window + 1)
                    for ctx_pos in range(left, right):
                        if ctx_pos == center_pos:
                            continue
                        ctx_idx = mapped[ctx_pos]
                        distance = abs(center_pos - ctx_pos)
                        key = int(center_idx) * vs + int(ctx_idx)
                        local[key] += 1.0 / float(distance)

                seen_sentences += 1
                if (
                    seen_sentences % self.shard_sentence_count == 0
                    or len(local) >= self.max_cooc_entries_in_memory
                ):
                    self._flush_cooc_shard(local, shard_dir, shard_pairs)

            self._flush_cooc_shard(local, shard_dir, shard_pairs)
            return shard_dir, shard_pairs
        except Exception:
            shutil.rmtree(shard_dir, ignore_errors=True)
            raise

    def _iter_merged_pairs(
        self, shard_pairs: list[tuple[str, str]]
    ) -> Iterable[tuple[int, int, float]]:
        """
        Yield globally merged ``(row_idx, col_idx, value)`` pairs from shards.

        Uses a memory-bounded k-way merge over sorted shard streams loaded with
        ``mmap_mode='r'``.
        """
        vocab_size = len(self.vocab)
        key_arrays = []
        val_arrays = []
        positions = []
        heap: list[tuple[int, int]] = []

        for shard_idx, (key_path, val_path) in enumerate(shard_pairs):
            keys = np.load(key_path, mmap_mode="r")
            vals = np.load(val_path, mmap_mode="r")
            if len(keys) != len(vals):
                raise ValueError("Corrupt co-occurrence shard: keys/values length mismatch")
            key_arrays.append(keys)
            val_arrays.append(vals)
            positions.append(0)
            if len(keys) > 0:
                heapq.heappush(heap, (int(keys[0]), shard_idx))

        while heap:
            key, shard_idx = heapq.heappop(heap)
            agg = 0.0
            while True:
                keys = key_arrays[shard_idx]
                vals = val_arrays[shard_idx]
                pos = positions[shard_idx]
                agg += float(vals[pos])
                positions[shard_idx] += 1
                next_pos = positions[shard_idx]
                if next_pos < len(keys):
                    heapq.heappush(heap, (int(keys[next_pos]), shard_idx))

                if not heap or heap[0][0] != key:
                    break
                _, shard_idx = heapq.heappop(heap)

            if self.min_cooc_count > 0 and agg < self.min_cooc_count:
                continue

            row_idx = key // vocab_size
            col_idx = key % vocab_size
            yield row_idx, col_idx, agg

    def _cooc_dict_to_arrays(
        self, cooc: dict[tuple[int, int], float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not cooc:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        items = list(cooc.items())
        row_idx = np.fromiter((key[0] for key, _ in items), dtype=np.int32, count=len(items))
        col_idx = np.fromiter((key[1] for key, _ in items), dtype=np.int32, count=len(items))
        values = np.fromiter((value for _, value in items), dtype=np.float32, count=len(items))
        if self.min_cooc_count > 0:
            keep = values >= self.min_cooc_count
            row_idx = row_idx[keep]
            col_idx = col_idx[keep]
            values = values[keep]
        return row_idx, col_idx, values

    def _build_cooccurrence(
        self, sentences: Iterable[list[str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.verbose:
            logger.info("Building co-occurrence matrix in %s mode...", self.mode)
        if self.mode == "in_memory":
            return self._build_cooc_in_memory(sentences)
        raise RuntimeError("Disk mode should use streaming training path")

    def _train_from_disk_cooc(
        self,
        shard_dir: str,
        shard_pairs: list[tuple[str, str]],
        epochs: int,
    ) -> tuple[float, int]:
        """
        Train by streaming merged shard pairs in chunks.

        Returns:
            Tuple ``(avg_loss, retained_pairs)`` where ``retained_pairs`` is the
            number of merged pairs that passed ``min_cooc_count`` in epoch 1.
        """
        total_loss = 0.0
        retained_pairs = 0
        chunk_size = self.cooc_train_chunk_size

        rows = np.empty(chunk_size, dtype=np.int32)
        cols = np.empty(chunk_size, dtype=np.int32)
        vals = np.empty(chunk_size, dtype=np.float32)

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_pairs = 0
                fill = 0
                epoch_seed = int(self._get_random_state() % (2**31 - 1))

                for row_idx, col_idx, value in self._iter_merged_pairs(shard_pairs):
                    rows[fill] = row_idx
                    cols[fill] = col_idx
                    vals[fill] = value
                    fill += 1

                    if fill == chunk_size:
                        chunk_seed = int((epoch_seed + epoch_pairs) % (2**31 - 1))
                        chunk_loss, chunk_pairs = glove_c.train_epoch_glove(
                            self._W_input,
                            self._W_context,
                            self._bias_input,
                            self._bias_context,
                            self._grad_sq_input,
                            self._grad_sq_context,
                            self._grad_sq_bias_input,
                            self._grad_sq_bias_context,
                            rows,
                            cols,
                            vals,
                            float(self.alpha),
                            float(self.x_max),
                            float(self.power),
                            chunk_seed,
                            bool(self.shuffle),
                        )
                        epoch_loss += float(chunk_loss)
                        epoch_pairs += int(chunk_pairs)
                        fill = 0

                if fill > 0:
                    chunk_seed = int((epoch_seed + epoch_pairs) % (2**31 - 1))
                    chunk_loss, chunk_pairs = glove_c.train_epoch_glove(
                        self._W_input,
                        self._W_context,
                        self._bias_input,
                        self._bias_context,
                        self._grad_sq_input,
                        self._grad_sq_context,
                        self._grad_sq_bias_input,
                        self._grad_sq_bias_context,
                        rows[:fill],
                        cols[:fill],
                        vals[:fill],
                        float(self.alpha),
                        float(self.x_max),
                        float(self.power),
                        chunk_seed,
                        bool(self.shuffle),
                    )
                    epoch_loss += float(chunk_loss)
                    epoch_pairs += int(chunk_pairs)

                if epoch == 0:
                    retained_pairs = epoch_pairs
                total_loss += epoch_loss
                if self.verbose:
                    logger.info(
                        "Epoch %d/%d loss=%.6f pairs=%d (shard-streamed)",
                        epoch + 1,
                        epochs,
                        epoch_loss,
                        epoch_pairs,
                    )

            return total_loss / float(epochs), retained_pairs
        finally:
            shutil.rmtree(shard_dir, ignore_errors=True)

    def train(
        self,
        sentences: Iterable[list[str]] | None = None,
        epochs: int | None = None,
        update_vocab: bool = False,
        reset_lr: bool = True,
    ) -> float | None:
        """
        Train GloVe on a restartable sentence iterable.

        Workflow:
        1. Build or reuse vocabulary.
        2. Build weighted co-occurrence pairs via ``mode`` backend.
        3. Run Cython AdaGrad updates for ``epochs`` passes.

        Memory behavior:
        - ``in_memory`` keeps all retained sparse pairs in RAM.
        - ``disk`` writes sorted shards and streams merged pairs by chunk,
          avoiding full sparse-matrix materialization.

        Args:
            sentences: Optional restartable corpus override.
            epochs: Number of epochs. If None, uses ``self.epochs`` from model
                initialization.
            update_vocab: Not implemented for GloVe (raises when requested on
                initialized vocabulary).
            reset_lr: If True, reset ``alpha`` to constructor value.

        Returns:
            Average epoch loss when ``calculate_loss=True``; otherwise ``None``.
        """
        if sentences is None:
            sentences = self._sentences
        if sentences is None:
            raise ValueError(
                "No sentences provided. Pass sentences to train() or provide them at "
                "GloVe() initialization."
            )
        if iter(sentences) is sentences:
            raise ValueError(
                "sentences must be restartable because vocabulary counting, "
                "co-occurrence building, and multi-epoch training traverse the corpus "
                "more than once. Use a list or LineSentenceFile."
            )
        if epochs is None:
            epochs = self.epochs
        if epochs is None:
            raise ValueError("epochs must be specified either at init or in train()")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        if update_vocab and self.vocab:
            raise NotImplementedError("GloVe update_vocab=True is not implemented yet")

        if reset_lr and hasattr(self, "_initial_alpha"):
            self.alpha = self._initial_alpha

        if not self.vocab:
            self.build_vocab(sentences)
        if not self.vocab:
            raise ValueError("No vocabulary after filtering. Lower min_word_count.")

        if (
            self._W_input is None
            or self._W_context is None
            or self._bias_input is None
            or self._bias_context is None
        ):
            self._initialize_glove_params()

        if self.workers > 1 and self.verbose:
            logger.warning(
                "GloVe currently updates a single shared stream; workers=%d is accepted "
                "for API compatibility but does not parallelize updates.",
                self.workers,
            )
        if self.mode == "in_memory":
            row_idx, col_idx, values = self._build_cooccurrence(sentences)
            if len(values) == 0:
                raise ValueError(
                    "No co-occurrence pairs available for training. "
                    "Try reducing min_word_count or min_cooc_count."
                )
            if self.verbose:
                logger.info(
                    "GloVe training start: %d epochs, %d pairs, mode=%s",
                    epochs,
                    len(values),
                    self.mode,
                )
            total_loss = 0.0
            for epoch in range(epochs):
                seed = int(self._get_random_state() % (2**31 - 1))
                epoch_loss, _ = glove_c.train_epoch_glove(
                    self._W_input,
                    self._W_context,
                    self._bias_input,
                    self._bias_context,
                    self._grad_sq_input,
                    self._grad_sq_context,
                    self._grad_sq_bias_input,
                    self._grad_sq_bias_context,
                    row_idx,
                    col_idx,
                    values,
                    float(self.alpha),
                    float(self.x_max),
                    float(self.power),
                    seed,
                    bool(self.shuffle),
                )
                total_loss += float(epoch_loss)
                if self.verbose:
                    logger.info("Epoch %d/%d loss=%.6f", epoch + 1, epochs, float(epoch_loss))
            avg_loss = total_loss / float(epochs)
        else:
            shard_dir, shard_pairs = self._build_cooc_disk(sentences)
            if not shard_pairs:
                shutil.rmtree(shard_dir, ignore_errors=True)
                raise ValueError(
                    "No co-occurrence pairs available for training. "
                    "Try reducing min_word_count or min_cooc_count."
                )
            if self.verbose:
                logger.info(
                    "GloVe training start: %d epochs, mode=%s, chunk_size=%d, shards=%d",
                    epochs,
                    self.mode,
                    self.cooc_train_chunk_size,
                    len(shard_pairs),
                )
            avg_loss, retained_pairs = self._train_from_disk_cooc(
                shard_dir,
                shard_pairs,
                epochs,
            )
            if retained_pairs <= 0:
                raise ValueError(
                    "No co-occurrence pairs available for training. "
                    "Try reducing min_word_count or min_cooc_count."
                )

        self._refresh_exposed_vectors()
        return avg_loss if self.calculate_loss else None

    def similarity(self, word1: str, word2: str, cross_space: bool = False) -> float:
        if word1 not in self.vocab:
            raise KeyError(f"Word '{word1}' not found in vocabulary")
        if word2 not in self.vocab:
            raise KeyError(f"Word '{word2}' not found in vocabulary")
        vec1 = self.W[self.vocab[word1]]
        vec2 = self.W_prime[self.vocab[word2]] if cross_space else self.W[self.vocab[word2]]
        return float(cosine_similarity(vec1, vec2))

    def save(self, path: str) -> None:
        """
        Persist full GloVe state including optimizer accumulators.

        Stores model configuration, vocabulary statistics, trainable parameters,
        and AdaGrad state so training can continue after ``load()``.
        """
        model_data = {
            "model_type": "glove",
            "vocab": self.vocab,
            "index2word": self.index2word,
            "word_counts": dict(self.word_counts),
            "corpus_word_count": self.corpus_word_count,
            "total_corpus_tokens": self.total_corpus_tokens,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_word_count": self.min_word_count,
            "max_vocab_size": self.max_vocab_size,
            "seed": self.seed,
            "alpha": self.alpha,
            "min_alpha": self.min_alpha,
            "epochs": self.epochs,
            "workers": self.workers,
            "verbose": self.verbose,
            "calculate_loss": self.calculate_loss,
            "mode": self.mode,
            "x_max": self.x_max,
            "power": self.power,
            "min_cooc_count": self.min_cooc_count,
            "shard_sentence_count": self.shard_sentence_count,
            "cooc_train_chunk_size": self.cooc_train_chunk_size,
            "max_cooc_entries_in_memory": self.max_cooc_entries_in_memory,
            "combine_vectors": self.combine_vectors,
            "_W_input": self._W_input,
            "_W_context": self._W_context,
            "_bias_input": self._bias_input,
            "_bias_context": self._bias_context,
            "_grad_sq_input": self._grad_sq_input,
            "_grad_sq_context": self._grad_sq_context,
            "_grad_sq_bias_input": self._grad_sq_bias_input,
            "_grad_sq_bias_context": self._grad_sq_bias_context,
        }
        with open(path, "wb") as handle:
            pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "GloVe":
        """
        Load a previously saved GloVe model from pickle.

        Raises:
            ValueError: if ``model_type`` in the file is not ``"glove"``.
        """
        with open(path, "rb") as handle:
            model_data = pickle.load(handle)
        if model_data.get("model_type") != "glove":
            raise ValueError(
                f"Invalid model type: expected 'glove', got {model_data.get('model_type')!r}"
            )

        model = cls(
            vector_size=model_data["vector_size"],
            window=model_data["window"],
            min_word_count=model_data["min_word_count"],
            max_vocab_size=model_data.get("max_vocab_size"),
            seed=model_data.get("seed"),
            alpha=model_data.get("alpha", 0.05),
            min_alpha=model_data.get("min_alpha"),
            epochs=model_data.get("epochs"),
            workers=model_data.get("workers", 1),
            verbose=model_data.get("verbose", False),
            calculate_loss=model_data.get("calculate_loss", True),
            mode=model_data.get("mode", "in_memory"),
            x_max=model_data.get("x_max", 100.0),
            power=model_data.get("power", 0.75),
            min_cooc_count=model_data.get("min_cooc_count", 0.0),
            shard_sentence_count=model_data.get("shard_sentence_count", 50000),
            cooc_train_chunk_size=model_data.get("cooc_train_chunk_size", 200000),
            max_cooc_entries_in_memory=model_data.get("max_cooc_entries_in_memory", 5_000_000),
            combine_vectors=model_data.get("combine_vectors", True),
            _skip_init=True,
        )

        model.vocab = model_data["vocab"]
        model.index2word = model_data["index2word"]
        model.word_counts = Counter(model_data.get("word_counts", {}))
        model.corpus_word_count = model_data.get(
            "corpus_word_count", sum(model.word_counts.values())
        )
        model.total_corpus_tokens = model_data.get(
            "total_corpus_tokens", model.corpus_word_count
        )

        model._W_input = model_data.get("_W_input")
        model._W_context = model_data.get("_W_context")
        model._bias_input = model_data.get("_bias_input")
        model._bias_context = model_data.get("_bias_context")
        model._grad_sq_input = model_data.get("_grad_sq_input")
        model._grad_sq_context = model_data.get("_grad_sq_context")
        model._grad_sq_bias_input = model_data.get("_grad_sq_bias_input")
        model._grad_sq_bias_context = model_data.get("_grad_sq_bias_context")
        model._refresh_exposed_vectors()
        return model
