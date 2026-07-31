"""
Internal vector math utilities used by analytics models.

This module is intentionally plotting-free so it can be safely imported by
training/inference code without pulling visualization dependencies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

__all__ = [
    "cosine_similarity",
    "cosine_distance",
    "most_similar",
    "align_vectors",
]


def cosine_similarity(
    v1: np.ndarray | list[float],
    v2: np.ndarray | list[float],
) -> float | np.ndarray:
    """
    Compute cosine similarity between vectors.

    If v1 and v2 are single vectors, computes similarity between them.
    If either is a matrix of vectors, uses sklearn's implementation.
    Returns 0.0 if either vector has zero norm.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    if v1.ndim == 1 and v2.ndim == 1:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    return sklearn_cosine_similarity(v1, v2)


def cosine_distance(
    v1: np.ndarray | list[float],
    v2: np.ndarray | list[float],
) -> float | np.ndarray:
    """Compute cosine distance between vectors (1 - cosine_similarity)."""
    return 1.0 - cosine_similarity(v1, v2)


def most_similar(
    target_vector: np.ndarray,
    vectors: list[np.ndarray] | np.ndarray,
    labels: list[str] | None = None,
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "cosine",
    top_n: int | None = None,
) -> list[tuple[str | int, float]]:
    """Find vectors most similar to target_vector using selected metric."""
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    if callable(metric):
        similarity_func = metric
    elif metric == "cosine":
        similarity_func = None
    else:
        raise ValueError("metric must be 'cosine' or a callable function")

    if similarity_func is None:
        target_2d = np.asarray(target_vector).reshape(1, -1)
        similarities = sklearn_cosine_similarity(target_2d, vectors).ravel().tolist()
    else:
        similarities = [similarity_func(target_vector, vec) for vec in vectors]

    if labels:
        if len(labels) != len(vectors):
            raise ValueError("Number of labels must match number of vectors")
        pairs = list(zip(labels, similarities))
    else:
        pairs = list(enumerate(similarities))

    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    if top_n is not None:
        return sorted_pairs[:top_n]
    return sorted_pairs


def align_vectors(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align source vectors with target vectors using Procrustes analysis.

    Returns a tuple of (aligned_vectors, transformation_matrix).
    """
    source_centered = source_vectors - np.mean(source_vectors, axis=0)
    target_centered = target_vectors - np.mean(target_vectors, axis=0)

    covariance = np.dot(target_centered.T, source_centered)
    u, _, vt = np.linalg.svd(covariance)
    rotation = np.dot(u, vt)
    aligned_vectors = np.dot(source_vectors, rotation.T)
    return aligned_vectors, rotation
