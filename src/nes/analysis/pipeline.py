"""End-to-end NES scoring and evaluation pipeline.

Loads the segmented database and reference embeddings, scores every candidate
segment with the three methods (consensus, embedding similarity, neural
network), combines them into a ranked list and -- when ground-truth labels are
available -- evaluates each method and writes the result tables and figures.

Run as a script::

    python -m nes.analysis.pipeline
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from nes import config
from nes.analysis.combine import combine_scores_and_rank
from nes.analysis.consensus import (
    extract_segment_sequences,
    filter_segments_by_length,
    score_nes_consensus_pattern,
)
from nes.analysis.embedding_scoring import score_segments_with_embeddings
from nes.analysis.metrics import evaluate_accuracy, load_true_labels, print_results
from nes.data.peptides import load_peptide_data_lists
from nes.embeddings import get_esm_embeddings, get_esm_model
from nes.viz.plots import plot_roc_curve


def _load_reference_embeddings(embedding_size: int, embedding_layer: int, data_csv: Path):
    """Embed the positive and negative reference peptides with ESM-2."""
    pos_pep, neg_pep, _ = load_peptide_data_lists(data_csv=data_csv)
    model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)
    kwargs = dict(embedding_layer=embedding_layer, sequence_embedding=True)
    positive = get_esm_embeddings(pos_pep, model, alphabet, batch_converter, device, **kwargs)
    negative = get_esm_embeddings(neg_pep, model, alphabet, batch_converter, device, **kwargs)
    return positive, negative


def _load_neural_scores(data_dir: Path, segment_keys, union_keys: bool) -> Dict[str, float]:
    """Load pre-computed neural-network logits, restricted to current segments."""
    name = "network_all_scores.pkl" if union_keys else "network_test_scores.pkl"
    with open(data_dir / name, "rb") as fh:
        scores = pickle.load(fh)
    return {k: v for k, v in scores.items() if k in segment_keys}


def run_pipeline(
    data_dir: Path = config.DATA_DIR,
    results_dir: Path = config.RESULTS_DIR,
    data_csv: Path = config.DB_CSV,
    embedding_size: int = config.EMBEDDING_SIZE,
    embedding_layer: int = config.EMBEDDING_LAYER,
    min_length: int = config.MIN_SEGMENT_LEN,
    max_length: int = config.MAX_SEGMENT_LEN,
    consensus_weight: float = 0.4,
    embedding_weight: float = 0.4,
    neural_net_weight: float = 0.3,
    top_n_display: int = 10,
    union_keys: bool = True,
    save_results: bool = True,
) -> dict:
    """Run the full scoring and evaluation pipeline.

    Args:
        data_dir: Directory with the segmented database and network scores.
        results_dir: Directory for output tables and figures.
        data_csv: Path to the NesDB CSV (for reference peptide embeddings).
        embedding_size: ESM-2 embedding width for reference peptides.
        embedding_layer: ESM-2 layer to pool for reference peptides.
        min_length: Minimum candidate segment length (residues).
        max_length: Maximum candidate segment length (residues).
        consensus_weight: Weight of the consensus scorer in the combination.
        embedding_weight: Weight of the embedding scorer.
        neural_net_weight: Weight of the neural-network scorer.
        top_n_display: Number of top candidates to print.
        union_keys: Score segments covered by any method (``True``) or all
            methods (``False``).
        save_results: Whether to write the candidate and accuracy tables.

    Returns:
        A dict with the ranked segments, per-method scores and evaluation.
    """
    data_dir = Path(data_dir)
    results_dir = config.ensure_dir(Path(results_dir))

    # --- Load segmentation data and references -----------------------------
    with open(data_dir / "protein_segments.pkl", "rb") as fh:
        segment_boundaries = pickle.load(fh)
    with open(data_dir / "protein_segment_embeddings.pkl", "rb") as fh:
        segment_embeddings = pickle.load(fh)
    with open(data_dir / "protein_sequences.pkl", "rb") as fh:
        protein_sequences = pickle.load(fh)

    positive_ref, negative_ref = _load_reference_embeddings(
        embedding_size, embedding_layer, data_csv
    )
    true_labels = load_true_labels(data_dir / "zps_segments.jsonl")

    # --- Filter candidate segments -----------------------------------------
    filtered = filter_segments_by_length(segment_boundaries, min_length, max_length)
    segment_sequences = extract_segment_sequences(protein_sequences, filtered)
    print(f"Candidate segments after length filter: {len(segment_sequences)}")

    # --- Three scorers ------------------------------------------------------
    consensus_results = score_nes_consensus_pattern(segment_sequences)

    filtered_embeddings = {
        k: v for k, v in segment_embeddings.items() if k in segment_sequences
    }
    embedding_scores = score_segments_with_embeddings(
        filtered_embeddings, positive_ref, negative_ref
    )

    neural_scores = _load_neural_scores(data_dir, segment_sequences, union_keys)

    # --- Combine and rank ---------------------------------------------------
    ranked = combine_scores_and_rank(
        consensus_results, embedding_scores, neural_scores,
        consensus_weight=consensus_weight,
        embedding_weight=embedding_weight,
        neural_net_weight=neural_net_weight,
        union_keys=union_keys,
    )
    print(f"Ranked {len(ranked)} segments")

    print(f"\nTop {top_n_display} NES candidates:")
    for rank, (key, score, details) in enumerate(ranked[:top_n_display], start=1):
        print(f"  #{rank} {key}  seq={segment_sequences[key]}  score={score:.4f}")

    # --- Evaluation ---------------------------------------------------------
    accuracy_results = {}
    if true_labels:
        combined_scores = {key: score for key, score, _ in ranked}
        accuracy_results = evaluate_accuracy(
            segment_keys=list(segment_sequences),
            true_labels_dict=true_labels,
            method_scores={
                "consensus": {k: v["score"] for k, v in consensus_results.items()},
                "embedding": embedding_scores,
                "neural": neural_scores,
                "combined": combined_scores,
            },
            optimization_metric="f1",
            results_dir=results_dir,
        )
        print_results(accuracy_results)
        _plot_roc_curves(accuracy_results, results_dir)

    if save_results:
        _save_candidate_table(ranked, segment_sequences, results_dir)
        _save_accuracy_table(accuracy_results, results_dir)

    return {
        "ranked_segments": ranked,
        "segment_sequences": segment_sequences,
        "consensus_results": consensus_results,
        "embedding_scores": embedding_scores,
        "neural_scores": neural_scores,
        "accuracy_results": accuracy_results,
    }


def _plot_roc_curves(accuracy_results: dict, results_dir: Path) -> None:
    for method in ("consensus", "embedding", "neural", "combined"):
        metrics = accuracy_results.get(method)
        if metrics:
            plot_roc_curve(
                accuracy_results["true_labels"],
                metrics["scores"],
                results_dir / f"{method}_roc_curve.png",
            )


def _save_candidate_table(ranked, segment_sequences, results_dir: Path) -> None:
    rows = []
    for rank, (key, score, details) in enumerate(ranked, start=1):
        protein_id, positions = key.split()
        sequence = segment_sequences[key]
        rows.append({
            "rank": rank,
            "segment_id": key,
            "protein_id": protein_id,
            "positions": positions,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "combined_score": score,
            "consensus_score": details["raw_consensus"],
            "embedding_score": details["raw_embedding"],
            "neural_score": details["raw_neural_net"],
            "num_pattern_matches": len(details["pattern_matches"]),
        })
    out = Path(results_dir) / "nes_candidates.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved candidate table -> {out}")


def _save_accuracy_table(accuracy_results: dict, results_dir: Path) -> None:
    if not accuracy_results:
        return
    summary = {}
    for method in ("consensus", "embedding", "neural", "combined"):
        metrics = accuracy_results.get(method)
        if metrics:
            summary[method] = {
                k: v for k, v in metrics.items()
                if k not in ("scores", "predictions", "all_optimal_thresholds")
            }
    out = Path(results_dir) / "accuracy_results.csv"
    pd.DataFrame(summary).T.to_csv(out)
    print(f"Saved accuracy table -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NES scoring pipeline.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--data-csv", type=Path, default=config.DB_CSV)
    parser.add_argument("--consensus-weight", type=float, default=0.4)
    parser.add_argument("--embedding-weight", type=float, default=0.4)
    parser.add_argument("--neural-net-weight", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        data_csv=args.data_csv,
        consensus_weight=args.consensus_weight,
        embedding_weight=args.embedding_weight,
        neural_net_weight=args.neural_net_weight,
        top_n_display=args.top_n,
    )


if __name__ == "__main__":
    main()
