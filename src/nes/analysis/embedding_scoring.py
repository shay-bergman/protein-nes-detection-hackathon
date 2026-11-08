"""Embedding-similarity scoring of candidate segments.

Each segment is scored by how much closer it sits, in ESM-2 embedding space, to
known positive NES peptides than to matched negative peptides. The score is a
signed log-fold difference of mean distances, so larger means more NES-like.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from nes.data.peptides import get_peptide_distances


def score_segments_with_embeddings(
    segment_embeddings: Dict[str, np.ndarray],
    positive_reference_embeddings: Sequence[np.ndarray],
    negative_reference_embeddings: Sequence[np.ndarray],
) -> Dict[str, float]:
    """Score segments by log-fold distance difference to reference peptides.

    Args:
        segment_embeddings: Mapping of segment key -> pooled segment embedding.
        positive_reference_embeddings: Embeddings of known positive peptides.
        negative_reference_embeddings: Embeddings of matched negative peptides.

    Returns:
        Segment key -> ``log1p(dist_to_neg) - log1p(dist_to_pos)`` (larger is
        more NES-like).
    """
    keys = list(segment_embeddings)
    embeddings = [segment_embeddings[key] for key in keys]

    dist_to_pos = get_peptide_distances(embeddings, positive_reference_embeddings, np.mean)
    dist_to_neg = get_peptide_distances(embeddings, negative_reference_embeddings, np.mean)
    scores = np.log1p(dist_to_neg) - np.log1p(dist_to_pos)

    return dict(zip(keys, scores))
