"""Train the NES classifier on segment embeddings and export its scores.

Loads the ``.npz`` splits created by :mod:`nes.data.splits`, trains
:class:`~nes.model.network.SimpleDenseNet` with class-balanced sampling, tunes a
decision threshold on the validation set, reports test accuracy and writes the
per-segment logits used by the combined-scoring analysis.

Run as a script::

    python -m nes.model.train
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from nes import config
from nes.model.network import (
    SimpleDenseNet,
    get_net_scores,
    prepare_balanced_loader,
    train_net,
)
from nes.viz.plots import plot_loss_curve, plot_roc_curve


@dataclass
class TrainingConfig:
    """Hyper-parameters for a training run."""

    embedding_size: int = config.EMBEDDING_SIZE
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-4
    hidden_dim: int = 512
    dropout: float = 0.4


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray, thresh: float = 0.5) -> float:
    """Compute binary accuracy at a decision threshold.

    Args:
        probs: Predicted positive-class probabilities.
        labels: True binary labels.
        thresh: Decision threshold applied to ``probs``.

    Returns:
        The fraction of correctly classified examples.
    """
    return ((probs >= thresh).astype(int) == labels).mean()


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximises F1.

    Args:
        labels: True binary labels.
        probs: Predicted positive-class probabilities.

    Returns:
        A tuple ``(best_threshold, best_f1)``.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]), float(f1[best_idx])


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def run_training(cfg: TrainingConfig, data_dir: Path, results_dir: Path) -> SimpleDenseNet:
    """Train the classifier and persist its scores and plots.

    Args:
        cfg: Training hyper-parameters.
        data_dir: Directory holding the ``splits/`` ``.npz`` files.
        results_dir: Directory for loss and ROC figures.

    Returns:
        The trained ``SimpleDenseNet``.
    """
    data_dir, results_dir = Path(data_dir), config.ensure_dir(Path(results_dir))
    splits_dir = data_dir / "splits"

    train = np.load(splits_dir / "train.npz", allow_pickle=True)
    val = np.load(splits_dir / "val.npz", allow_pickle=True)
    test = np.load(splits_dir / "test.npz", allow_pickle=True)

    n_pos = int((train["y"] == 1).sum())
    n_neg = int((train["y"] == 0).sum())

    train_loader = prepare_balanced_loader(
        train["X"], train["y"], n_neg, n_pos, batch_size=cfg.batch_size
    )
    model = SimpleDenseNet(
        esm_emb_dim=cfg.embedding_size, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout
    )
    model = train_net(model, train_loader, num_epochs=cfg.epochs, lr=cfg.lr)
    plot_loss_curve(model.training_losses_, results_dir / "training_loss.png")

    # Tune the decision threshold on the validation set.
    val_probs = _sigmoid(get_net_scores(model, val["X"]))
    best_t, best_f1 = best_f1_threshold(val["y"], val_probs)
    print(f"Best validation threshold: {best_t:.3f} (F1={best_f1:.3f})")
    plot_roc_curve(val["y"], val_probs, results_dir / "validation_roc_curve.png")

    test_probs = _sigmoid(get_net_scores(model, test["X"]))
    test_acc = accuracy_from_probs(test_probs, test["y"], thresh=best_t)
    print(f"Test accuracy: {test_acc:.3f}")

    print("Top 5 test predictions:")
    for i in np.argsort(-test_probs)[:5]:
        print(f"  {test['keys'][i]}: {test_probs[i]:.3f}  label={test['y'][i]}")

    _export_scores(model, train, val, test, data_dir)
    return model


def _export_scores(model, train, val, test, data_dir: Path) -> None:
    """Save per-segment logits for every split and for the held-out splits."""
    all_keys = np.concatenate([train["keys"], val["keys"], test["keys"]])
    all_logits = get_net_scores(model, np.concatenate([train["X"], val["X"], test["X"]]))
    with open(data_dir / "network_all_scores.pkl", "wb") as fh:
        pickle.dump(dict(zip(all_keys, all_logits)), fh)

    held_keys = np.concatenate([val["keys"], test["keys"]])
    held_logits = get_net_scores(model, np.concatenate([val["X"], test["X"]]))
    with open(data_dir / "network_test_scores.pkl", "wb") as fh:
        pickle.dump(dict(zip(held_keys, held_logits)), fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NES classifier.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=TrainingConfig.dropout)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    args = parser.parse_args()

    cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    run_training(cfg, data_dir=args.data_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
