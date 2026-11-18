"""Group passage embeddings into candidate frames with spherical k-means.

Plain NumPy so the workflow needs no extra dependency. Vectors are assumed
L2-normalized (every embedding backend guarantees this), so cosine similarity
is a dot product and centroids are re-normalized means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clustering:
    labels: np.ndarray
    """``(n,)`` cluster index per passage; clusters are numbered largest first."""
    centroids: np.ndarray
    """``(k, dim)`` unit-length centroids."""
    similarity: np.ndarray
    """``(n,)`` cosine similarity of each passage to its own centroid."""

    @property
    def k(self) -> int:
        return len(self.centroids)

    def members(self, cluster: int) -> np.ndarray:
        """Passage indexes in ``cluster``, most central first."""
        idx = np.flatnonzero(self.labels == cluster)
        return idx[np.argsort(-self.similarity[idx], kind="stable")]


def cluster_passages(
    vectors: np.ndarray,
    max_clusters: int,
    *,
    min_size: int = 1,
    seed: int = 0,
    max_iter: int = 100,
) -> Clustering:
    """Cluster unit vectors into at most ``max_clusters`` groups of at least ``min_size``.

    The number of clusters is ``min(max_clusters, n // min_size)``. Clusters that
    end up smaller than ``min_size`` are dissolved and their passages reassigned
    to the nearest surviving centroid, so a frame always has real support.
    Deterministic for a given ``seed``.
    """
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("cluster_passages needs a non-empty 2-D array of vectors")
    n = len(x)
    k = max(1, min(max_clusters, n // max(min_size, 1)))
    rng = np.random.default_rng(seed)

    centroids = _kmeans_pp(x, k, rng)
    labels = np.full(n, -1)
    for _ in range(max_iter):
        new_labels = np.argmax(x @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = _recompute(x, labels, centroids)

    # Dissolve undersized clusters, smallest first, until all meet min_size.
    while len(centroids) > 1:
        sizes = np.bincount(labels, minlength=len(centroids))
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_size:
            break
        centroids = np.delete(centroids, smallest, axis=0)
        labels = np.argmax(x @ centroids.T, axis=1)
        centroids = _recompute(x, labels, centroids)

    # Renumber largest-first (ties broken by original index) and drop empties.
    sizes = np.bincount(labels, minlength=len(centroids))
    order = [c for c in sorted(range(len(centroids)), key=lambda c: (-sizes[c], c)) if sizes[c] > 0]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(c)] for c in labels])
    centroids = centroids[order]
    similarity = np.einsum("ij,ij->i", x, centroids[labels])
    return Clustering(labels=labels, centroids=centroids, similarity=similarity)


def _kmeans_pp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centroids = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        dist = 1.0 - np.max(x @ np.array(centroids).T, axis=1)
        dist = np.clip(dist, 0.0, None)
        total = dist.sum()
        if total <= 0:  # every point coincides with a centroid already
            centroids.append(x[rng.integers(len(x))])
            continue
        centroids.append(x[rng.choice(len(x), p=dist / total)])
    return _normalize(np.array(centroids))


def _recompute(x: np.ndarray, labels: np.ndarray, previous: np.ndarray) -> np.ndarray:
    centroids = previous.copy()
    for c in range(len(previous)):
        members = x[labels == c]
        if len(members):
            centroids[c] = members.mean(axis=0)
    return _normalize(centroids)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)
