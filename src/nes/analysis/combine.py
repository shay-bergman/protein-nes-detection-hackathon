"""Combine the three scorers into a single ranked list of NES candidates."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

RankedSegment = Tuple[str, float, dict]


def _normalize(values: Sequence[float]) -> List[float]:
    """Min-max normalise ``values`` into ``[0, 1]`` (constant input -> 0/1 by sign)."""
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - min_val) / (max_val - min_val) for v in values]


def combine_scores_and_rank(
    consensus_scores: Dict[str, dict],
    embedding_scores: Dict[str, float],
    neural_net_scores: Dict[str, float],
    consensus_weight: float = 0.4,
    embedding_weight: float = 0.4,
    neural_net_weight: float = 0.3,
    union_keys: bool = True,
) -> List[RankedSegment]:
    """Normalise, weight and combine the three scorers.

    Args:
        consensus_scores: Segment key -> ``{"score": ..., "matches": ...}``.
        embedding_scores: Segment key -> scalar embedding-similarity score.
        neural_net_scores: Segment key -> scalar neural-network logit.
        consensus_weight: Weight of the consensus scorer in the combination.
        embedding_weight: Weight of the embedding scorer.
        neural_net_weight: Weight of the neural-network scorer.
        union_keys: If ``True`` rank every segment scored by any method
            (missing scores treated as 0); if ``False`` rank only segments
            scored by all methods.

    Returns:
        A list of ``(key, combined_score, details)`` sorted by combined score,
        highest first.
    """
    if union_keys:
        keys = set(consensus_scores) | set(embedding_scores) | set(neural_net_scores)
    else:
        keys = set(consensus_scores) & set(embedding_scores) & set(neural_net_scores)
    if not keys:
        return []

    keys = list(keys)
    consensus_values = [consensus_scores.get(k, {}).get("score", 0) for k in keys]
    embedding_values = [embedding_scores.get(k, 0) for k in keys]
    neural_values = [neural_net_scores.get(k, 0) for k in keys]

    norm_consensus = _normalize(consensus_values)
    norm_embedding = _normalize(embedding_values)
    norm_neural = _normalize(neural_values)

    results: List[RankedSegment] = []
    for i, key in enumerate(keys):
        combined = (
            consensus_weight * norm_consensus[i]
            + embedding_weight * norm_embedding[i]
            + neural_net_weight * norm_neural[i]
        )
        details = {
            "combined_score": combined,
            "raw_consensus": consensus_scores.get(key, {}).get("score", 0),
            "raw_embedding": embedding_scores.get(key, 0),
            "raw_neural_net": neural_net_scores.get(key, 0),
            "pattern_matches": consensus_scores.get(key, {}).get("matches", []),
        }
        results.append((key, combined, details))

    return sorted(results, key=lambda item: item[1], reverse=True)
