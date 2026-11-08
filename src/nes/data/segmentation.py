"""Protein segmentation via change-point detection on ESM-2 embeddings."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import ruptures as rpt

Segment = List[int]  # [start, end], end-exclusive


def get_protein_segments(
    emb_dict: Dict[str, np.ndarray],
    max_bkps_per100aa: int = 3,
) -> Dict[str, List[Segment]]:
    """Find contiguous segment boundaries for each protein.

    Each protein is split wherever its per-residue embedding changes character,
    giving a small number of candidate regions instead of exhaustive windows.

    Args:
        emb_dict: UniProt accession -> per-residue embedding matrix ``(L, d)``.
        max_bkps_per100aa: Maximum number of break-points per 100 residues.

    Returns:
        Accession -> list of ``[start, end]`` segments, or the string
        ``"Failed"`` if the change-point search did not converge.
    """
    protein_segments: Dict[str, List[Segment]] = {}

    for protein, emb in emb_dict.items():
        n_boundaries = max(int(emb.shape[0] * max_bkps_per100aa / 100), 1)

        try:
            algo = rpt.Window(width=30, model="rbf", jump=1).fit(emb)
            boundaries = algo.predict(n_bkps=n_boundaries)
            segments: List[Segment] = [[0, boundaries[0]]]
            segments.extend(
                [boundaries[i], boundaries[i + 1]] for i in range(len(boundaries) - 1)
            )
        except Exception:
            # The search can fail when the break-point budget is infeasible.
            print(f"Segmentation failed for {protein} (n_boundaries={n_boundaries})")
            segments = "Failed"

        protein_segments[protein] = segments

    return protein_segments


def get_protein_segment_embeddings(
    emb_dict: Dict[str, np.ndarray],
    protein_segment_boundaries: Dict[str, List[Segment]],
) -> Dict[str, np.ndarray]:
    """Mean-pool per-residue embeddings within each segment.

    Args:
        emb_dict: Accession -> per-residue embedding matrix.
        protein_segment_boundaries: Output of ``get_protein_segments``.

    Returns:
        ``"<accession> <start>-<end>"`` -> pooled segment embedding vector.
    """
    segment_embeddings: Dict[str, np.ndarray] = {}

    for protein, emb in emb_dict.items():
        boundaries = protein_segment_boundaries[protein]
        if boundaries == "Failed":
            continue
        for start, end in boundaries:
            key = f"{protein} {start}-{end}"
            segment_embeddings[key] = np.mean(emb[start:end, :], axis=0)

    return segment_embeddings
