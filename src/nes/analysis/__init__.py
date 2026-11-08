"""Scoring and analysis of candidate NES segments.

Three complementary scorers are provided -- a rule-based consensus-pattern
scorer, an ESM embedding-similarity scorer and the trained neural network --
together with utilities to combine, threshold and evaluate them.
"""

from nes.analysis.combine import combine_scores_and_rank
from nes.analysis.consensus import (
    extract_segment_sequences,
    filter_segments_by_length,
    score_nes_consensus_pattern,
)
from nes.analysis.embedding_scoring import score_segments_with_embeddings

__all__ = [
    "filter_segments_by_length",
    "extract_segment_sequences",
    "score_nes_consensus_pattern",
    "score_segments_with_embeddings",
    "combine_scores_and_rank",
]
