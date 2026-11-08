"""Feed-forward classifier over ESM-2 segment embeddings."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def _device() -> str:
    """Return ``"cuda"`` if a GPU is available, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class SimpleDenseNet(nn.Module):
    """Small fully-connected binary classifier.

    The network maps an embedding to a single raw logit (no sigmoid) for use
    with ``BCEWithLogitsLoss``. Its architecture is three hidden blocks of
    ``Linear -> ReLU -> Dropout`` followed by a linear read-out.
    """

    def __init__(self, esm_emb_dim: int = 1280, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        """Build the network.

        Args:
            esm_emb_dim: Dimensionality of the input embeddings.
            hidden_dim: Width of the first two hidden layers.
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Float tensor of shape ``[batch, emb_dim]``.

        Returns:
            A tensor of shape ``[batch]`` with one raw logit per example.
        """
        return self.net(x).squeeze(1)


def prepare_loader(x: np.ndarray, y: np.ndarray, batch_size: int = 64) -> DataLoader:
    """Build a plain shuffled DataLoader.

    Args:
        x: Feature matrix of shape ``[n, emb_dim]``.
        y: Label vector of shape ``[n]``.
        batch_size: Mini-batch size.

    Returns:
        A shuffled ``DataLoader`` over ``(x, y)``.
    """
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def prepare_balanced_loader(
    x: np.ndarray, y: np.ndarray, n_neg: int, n_pos: int, batch_size: int = 64
) -> DataLoader:
    """Build a DataLoader that oversamples the minority class.

    Args:
        x: Feature matrix of shape ``[n, emb_dim]``.
        y: Label vector of shape ``[n]``.
        n_neg: Number of negative examples (for inverse-frequency weighting).
        n_pos: Number of positive examples.
        batch_size: Mini-batch size.

    Returns:
        A ``DataLoader`` using a ``WeightedRandomSampler``.
    """
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(x_t, y_t)

    class_weights = torch.tensor([1.0 / n_neg, 1.0 / n_pos])
    sample_weights = class_weights[y_t.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


def train_net(
    model: SimpleDenseNet,
    loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    pos_weight: Optional[float] = None,
) -> SimpleDenseNet:
    """Train the classifier with ``BCEWithLogitsLoss`` and Adam.

    Args:
        model: The network to train.
        loader: DataLoader yielding ``(features, labels)`` batches.
        num_epochs: Number of training epochs.
        lr: Adam learning rate.
        pos_weight: Positive-class loss weight; ``None`` weights classes
            equally (use together with a balanced loader).

    Returns:
        The trained model, with per-epoch losses stored on
        ``model.training_losses_``.
    """
    device = _device()
    pw = None if pos_weight is None else torch.tensor([float(pos_weight)], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device).train()
    losses: List[float] = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg = running_loss / len(loader.dataset)
        losses.append(avg)
        print(f"Epoch {epoch + 1:02d}/{num_epochs} - loss: {avg:.4f}")

    model.training_losses_ = losses
    return model


@torch.no_grad()
def get_net_scores(trained_net: SimpleDenseNet, esm_seq_embeddings: np.ndarray) -> np.ndarray:
    """Return raw logits for a batch of embeddings.

    Args:
        trained_net: A trained ``SimpleDenseNet``.
        esm_seq_embeddings: Embedding matrix of shape ``[n, emb_dim]``.

    Returns:
        A numpy array of ``n`` raw logits.
    """
    device = _device()
    trained_net.eval().to(device)
    x = torch.as_tensor(np.asarray(esm_seq_embeddings), dtype=torch.float32, device=device)
    return trained_net(x).cpu().numpy()
