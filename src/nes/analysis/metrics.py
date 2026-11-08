"""Threshold tuning and accuracy evaluation of the scoring methods.

Each scorer produces a continuous score; this module finds the decision
threshold that maximises a chosen metric, reports the usual classification
metrics at that threshold and draws per-method threshold-analysis figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_METRIC_FUNCS = {
    "accuracy": lambda y, p: accuracy_score(y, p),
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
}


def load_true_labels(jsonl_file_path: Path) -> Dict[str, int]:
    """Load ground-truth segment labels from a segment JSONL file.

    Args:
        jsonl_file_path: Path to a ``zps_segments*.jsonl`` file.

    Returns:
        Segment key -> integer label. Empty (with a warning) if the file is
        missing, which lets the pipeline skip evaluation rather than crash.
    """
    labels: Dict[str, int] = {}
    try:
        with open(jsonl_file_path) as fh:
            for line in fh:
                seg = json.loads(line)
                start, end = seg["coords"]
                labels[f"{seg['id']} {start}-{end}"] = seg["label"]
        print(f"Loaded true labels for {len(labels)} segments")
    except FileNotFoundError:
        print("Warning: true-labels file not found; accuracy evaluation skipped.")
    return labels


def find_optimal_threshold(
    true_labels: np.ndarray,
    predicted_scores: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Scan thresholds and return the one maximising ``metric``.

    Args:
        true_labels: True binary labels.
        predicted_scores: Continuous scores (larger is more positive-like).
        metric: One of ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``.
        n_thresholds: Number of thresholds sampled across the score range.

    Returns:
        A tuple ``(best_threshold, best_score, thresholds, scores)``.
    """
    thresholds = np.linspace(np.min(predicted_scores), np.max(predicted_scores), n_thresholds)
    metric_func = _METRIC_FUNCS[metric]
    scores = np.array([
        metric_func(true_labels, (predicted_scores >= t).astype(int)) for t in thresholds
    ])
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx]), thresholds, scores


def plot_threshold_analysis(
    true_labels: np.ndarray,
    predicted_scores: np.ndarray,
    method_name: str,
    save_path: Path = None,
) -> Dict[str, dict]:
    """Plot every metric versus threshold for one scoring method.

    Args:
        true_labels: True binary labels.
        predicted_scores: Continuous scores for the method.
        method_name: Human-readable method name for the figure title.
        save_path: Destination PNG path; the figure is not saved if ``None``.

    Returns:
        ``metric -> {"threshold": ..., "score": ...}`` optima.
    """
    optima: Dict[str, dict] = {}
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for metric in _METRIC_FUNCS:
        best_t, best_s, thresholds, scores = find_optimal_threshold(
            true_labels, predicted_scores, metric=metric
        )
        optima[metric] = {"threshold": best_t, "score": best_s}
        curves[metric] = (thresholds, scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, (thresholds, scores) in curves.items():
        line, = ax.plot(thresholds, scores, label=metric.title())
        best = optima[metric]
        ax.axvline(best["threshold"], color=line.get_color(), linestyle="--", alpha=0.6)
        ax.plot(best["threshold"], best["score"], "o", color=line.get_color())
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold analysis - {method_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return optima


def _evaluate_single(
    true_labels: np.ndarray,
    scores: np.ndarray,
    method_name: str,
    optimization_metric: str,
    save_path: Path = None,
) -> dict:
    """Evaluate one scorer at its metric-optimal threshold."""
    best_thresh, best_value, _, _ = find_optimal_threshold(
        true_labels, scores, metric=optimization_metric
    )
    optima = plot_threshold_analysis(true_labels, scores, method_name, save_path=save_path)
    predictions = (scores >= best_thresh).astype(int)
    multiclass = len(np.unique(true_labels)) > 1
    return {
        "optimal_threshold": best_thresh,
        "optimal_metric_value": best_value,
        "all_optimal_thresholds": optima,
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "auc": roc_auc_score(true_labels, scores) if multiclass else 0.0,
        "scores": scores,
        "predictions": predictions,
    }


def evaluate_accuracy(
    segment_keys: List[str],
    true_labels_dict: Dict[str, int],
    method_scores: Dict[str, Dict[str, float]],
    optimization_metric: str = "f1",
    results_dir: Path = Path("."),
) -> Dict[str, object]:
    """Evaluate all scoring methods on segments with ground-truth labels.

    Args:
        segment_keys: Candidate segment keys (only labelled ones are scored).
        true_labels_dict: Segment key -> ground-truth label.
        method_scores: ``method name -> {segment key -> score}`` to evaluate.
        optimization_metric: Metric whose optimum selects each threshold.
        results_dir: Directory for the per-method threshold figures.

    Returns:
        Per-method metric dictionaries plus ``true_labels``, ``segment_keys``
        and ``optimization_metric`` entries.
    """
    keys = [k for k in segment_keys if k in true_labels_dict]
    if not keys:
        print("No labelled segments to evaluate.")
        return {}

    print(f"\nEvaluating {len(keys)} labelled segments (optimising {optimization_metric})")
    true_labels = np.array([true_labels_dict[k] for k in keys])
    results: Dict[str, object] = {
        "true_labels": true_labels,
        "segment_keys": keys,
        "optimization_metric": optimization_metric,
    }

    for method, scores in method_scores.items():
        score_array = np.array([scores.get(k, 0) for k in keys])
        results[method] = _evaluate_single(
            true_labels, score_array, method.title(), optimization_metric,
            save_path=Path(results_dir) / f"{method}_threshold_analysis.png",
        )
    return results


def print_results(results: Dict[str, object]) -> None:
    """Pretty-print the per-method evaluation metrics."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 70)
    print("THRESHOLD-OPTIMISED EVALUATION")
    print("=" * 70)
    print(f"Optimisation metric: {results['optimization_metric']}")
    print(f"Segments evaluated:  {len(results['segment_keys'])}")

    for method in ("consensus", "embedding", "neural", "combined"):
        metrics = results.get(method)
        if not metrics:
            continue
        print(f"\n{method.upper()} SCORING")
        print(f"  optimal threshold : {metrics['optimal_threshold']:.4f}")
        print(f"  accuracy          : {metrics['accuracy']:.4f}")
        print(f"  precision         : {metrics['precision']:.4f}")
        print(f"  recall            : {metrics['recall']:.4f}")
        print(f"  f1                : {metrics['f1']:.4f}")
        print(f"  roc auc           : {metrics['auc']:.4f}")
