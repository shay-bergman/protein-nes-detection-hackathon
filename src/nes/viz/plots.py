"""Reusable matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

PathLike = Union[str, Path]


def plot_embedding_heatmap(
    amino_acid_embeddings: np.ndarray,
    out_file_path: PathLike = "heatmap.png",
    n_features: int = 100,
) -> None:
    """Plot a heatmap of per-residue embeddings.

    Args:
        amino_acid_embeddings: Per-residue matrix of shape ``(n_residues, n_features)``.
        out_file_path: Destination PNG path.
        n_features: Number of leading embedding features to show, for readability.
    """
    plt.figure(figsize=(10, 10))
    plt.matshow(amino_acid_embeddings[:, :n_features], fignum=1)
    plt.xlabel("Features")
    plt.ylabel("Amino acid")
    plt.title("Per-residue embedding heatmap")
    plt.savefig(out_file_path)
    plt.close()


def plot_boxplot(
    data_dict: Dict[str, Sequence[float]], out_file_path: PathLike = "boxplot.png"
) -> None:
    """Draw one box per group in a dictionary.

    Args:
        data_dict: Mapping from group label to its sequence of scores.
        out_file_path: Destination PNG path.
    """
    labels = list(data_dict.keys())
    plot_data = [data_dict[key] for key in labels]

    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, patch_artist=True, labels=labels)
    plt.xlabel("Group")
    plt.ylabel("Score")
    plt.title("Score distribution")
    plt.grid(True)
    plt.savefig(out_file_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, out_file_path: PathLike = "roc_curve.png") -> float:
    """Plot a ROC curve and return its AUC.

    Args:
        y_true: True binary labels (0 = negative, 1 = positive).
        y_scores: Continuous scores where larger means more positive-like.
        out_file_path: Destination PNG path.

    Returns:
        The area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Receiver operating characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_file_path)
    plt.close()
    return roc_auc


def plot_2d_reduction(coords_2d: np.ndarray, labels: Sequence, out_file_path: PathLike) -> None:
    """Scatter 2-D coordinates, colouring points by label.

    Args:
        coords_2d: Array of shape ``(n_points, 2)``.
        labels: Class or cluster label per point, used for colouring.
        out_file_path: Destination PNG path.
    """
    labels = np.asarray(labels)
    coords_2d = np.asarray(coords_2d)

    plt.figure(figsize=(6, 5))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(coords_2d[idx, 0], coords_2d[idx, 1], label=str(lab))
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Label", loc="best")
    plt.tight_layout()
    plt.savefig(out_file_path)
    plt.close()


def plot_loss_curve(losses: Sequence[float], out_file_path: PathLike = "training_loss.png") -> None:
    """Plot a training-loss curve over epochs.

    Args:
        losses: One loss value per epoch.
        out_file_path: Destination PNG path.
    """
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.savefig(out_file_path)
    plt.close()
