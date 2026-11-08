"""Unsupervised exploration of peptide embeddings (k-means + t-SNE).

Used to check whether positive and negative peptides separate in ESM-2
embedding space before any supervised model is trained.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def kmeans_clustering(embeddings: np.ndarray, k: int = 2, random_state: int = 42) -> np.ndarray:
    """Cluster embeddings with k-means.

    Args:
        embeddings: Array of shape ``(n_points, dim)``.
        k: Number of clusters.
        random_state: Seed for reproducible clustering.

    Returns:
        A per-point cluster-label array of length ``n_points``.
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return km.fit_predict(np.asarray(embeddings))


def tsne_reduction(embeddings: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to a low-dimensional layout with t-SNE.

    Args:
        embeddings: Array of shape ``(n_points, dim)``.
        n_components: Target dimensionality (2 for plotting).
        random_state: Seed for reproducible embedding.

    Returns:
        An array of shape ``(n_points, n_components)``.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(np.asarray(embeddings))
