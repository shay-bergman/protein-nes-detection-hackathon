"""Rule-based scoring of segments against classical NES consensus patterns.

Nuclear export signals recognised by CRM1 follow well-characterised hydrophobic
spacing patterns (e.g. the Kosugi classes and the PKI/Rev-type signals). This
module filters candidate segments to a plausible NES length window, recovers
their amino-acid sequences and scores them against a library of such patterns.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from nes.config import MAX_SEGMENT_LEN, MIN_SEGMENT_LEN

# Residue classes used by the consensus patterns.
CORE = set("LIVFM")          # hydrophobic core residues
RELAXED = CORE | set("AT")   # positions that tolerate small residues
STRUCTURAL = CORE | set("AYWP")
ACIDIC = set("DE")

# (name, window_length, positions, allowed-residue sets per position, weight)
_FORWARD_PATTERNS = [
    ("Kosugi_1a", 10, [0, 4, 7, 9], [RELAXED, RELAXED, CORE, CORE], 4),
    ("Kosugi_1b", 9, [0, 3, 6, 8], [RELAXED, RELAXED, CORE, CORE], 3),
    ("Kosugi_1c", 11, [0, 4, 8, 10], [RELAXED, RELAXED, CORE, CORE], 3),
    ("Kosugi_1d", 10, [0, 3, 7, 9], [RELAXED, RELAXED, CORE, CORE], 3),
    ("Kosugi_2", 8, [0, 2, 5, 7], [RELAXED, RELAXED, CORE, CORE], 3),
    ("Kosugi_3", 10, [0, 3, 7, 9], [RELAXED, RELAXED, CORE, CORE], 2),
]


def _all_patterns():
    """Return forward patterns plus their reverse-orientation counterparts."""
    patterns = list(_FORWARD_PATTERNS)
    for name, length, positions, residue_sets, weight in _FORWARD_PATTERNS:
        rev_positions = [length - 1 - p for p in positions]
        patterns.append((f"{name}_rev", length, rev_positions, residue_sets, weight // 2))
    return patterns


_PATTERNS = _all_patterns()


def filter_segments_by_length(
    protein_segment_boundaries: Dict[str, List],
    min_length: int = MIN_SEGMENT_LEN,
    max_length: int = MAX_SEGMENT_LEN,
) -> Dict[str, List[List[int]]]:
    """Keep only segments whose length falls in the NES-plausible window.

    Args:
        protein_segment_boundaries: Accession -> segments (or ``"Failed"``).
        min_length: Minimum segment length to keep, in residues.
        max_length: Maximum segment length to keep, in residues.

    Returns:
        Accession -> list of ``[start, end]`` segments within the window.
    """
    filtered: Dict[str, List[List[int]]] = {}
    for protein_id, segments in protein_segment_boundaries.items():
        if segments == "Failed":
            continue
        # Segment coordinates are end-exclusive, so the residue length is
        # ``end - start`` (matching the slice used to extract the sequence).
        kept = [
            [start, end]
            for start, end in segments
            if min_length <= (end - start) <= max_length
        ]
        if kept:
            filtered[protein_id] = kept
    return filtered


def extract_segment_sequences(
    protein_sequences: Dict[str, str],
    filtered_boundaries: Dict[str, List[List[int]]],
) -> Dict[str, str]:
    """Recover the amino-acid subsequence for each filtered segment.

    Args:
        protein_sequences: Accession -> full protein sequence.
        filtered_boundaries: Accession -> list of ``[start, end]`` segments.

    Returns:
        Segment key ``"<accession> <start>-<end>"`` -> amino-acid subsequence.
    """
    segment_sequences: Dict[str, str] = {}
    for protein_id, segments in filtered_boundaries.items():
        full_sequence = protein_sequences.get(protein_id)
        if full_sequence is None:
            continue
        for start, end in segments:
            segment_sequences[f"{protein_id} {start}-{end}"] = full_sequence[start:end]
    return segment_sequences


def _score_sequence(seq: str) -> Tuple[int, List[str]]:
    """Score a single segment sequence, returning ``(score, matched_pattern_tags)``."""
    score = 0
    matches: List[str] = []

    # Library patterns (forward and reverse).
    for name, length, positions, residue_sets, weight in _PATTERNS:
        for i in range(len(seq) - length + 1):
            window = seq[i : i + length]
            if all(window[p] in residue_sets[j] for j, p in enumerate(positions)):
                score += weight
                matches.append(f"{name}@{i}")

    # PKI-type signal: four core residues at a characteristic spacing.
    for i in range(len(seq) - 11):
        for offset in range(4):
            indices = [i, i + offset + 1, i + offset + 5, i + offset + 9, i + offset + 12]
            if all(idx < len(seq) and seq[idx] in CORE for idx in indices):
                score += 3
                matches.append(f"PKI@{i}")

    # Rev-type signal.
    for i in range(len(seq) - 9):
        w = seq[i : i + 9]
        if w[0] in STRUCTURAL and w[1] == "P" and w[3] in CORE and w[6] in CORE and w[8] in CORE:
            score += 3
            matches.append(f"Rev@{i}")

    # Overall hydrophobicity bonus.
    ratio = sum(1 for aa in seq if aa in CORE) / len(seq)
    if ratio >= 0.5:
        score += 2
    elif ratio >= 0.35:
        score += 1

    # Prolines disrupt the alpha-helix NES fold.
    if seq.count("P") > 2:
        score -= 1

    # Acidic residues often flank the signal.
    if seq[0] in ACIDIC:
        score += 1
    if seq[-1] in ACIDIC:
        score += 1

    return max(score, 0), matches


def score_nes_consensus_pattern(segment_sequences: Dict[str, str]) -> Dict[str, dict]:
    """Score every segment against the NES consensus-pattern library.

    Args:
        segment_sequences: Mapping of segment key -> amino-acid sequence.

    Returns:
        Segment key -> ``{"score": int, "matches": list[str]}``.
    """
    results: Dict[str, dict] = {}
    for key, seq in segment_sequences.items():
        score, matches = _score_sequence(seq)
        results[key] = {"score": score, "matches": matches}
    return results
