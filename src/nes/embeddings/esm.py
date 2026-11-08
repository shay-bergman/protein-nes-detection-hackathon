"""Wrapper around the pre-trained ESM-2 family of protein language models."""

from __future__ import annotations

from typing import List, Tuple

import esm
import numpy as np
import torch

# Pre-trained ESM-2 checkpoints keyed by their embedding dimensionality.
ESM_MODELS = {
    320: esm.pretrained.esm2_t6_8M_UR50D,
    480: esm.pretrained.esm2_t12_35M_UR50D,
    640: esm.pretrained.esm2_t30_150M_UR50D,
    1280: esm.pretrained.esm2_t33_650M_UR50D,
    2560: esm.pretrained.esm2_t36_3B_UR50D,
    5120: esm.pretrained.esm2_t48_15B_UR50D,
}


def get_esm_model(embedding_size: int = 1280) -> Tuple[object, object, object, str]:
    """Load a pre-trained ESM-2 model by embedding width.

    Args:
        embedding_size: Embedding dimensionality; must be a key of ``ESM_MODELS``.

    Returns:
        A tuple ``(model, alphabet, batch_converter, device)`` ready for inference.

    Raises:
        ValueError: If no model exists for the requested embedding size.
    """
    if embedding_size not in ESM_MODELS:
        raise ValueError(
            f"No ESM-2 model with embedding size {embedding_size}. "
            f"Choose one of {sorted(ESM_MODELS)}."
        )

    model, alphabet = ESM_MODELS[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disable dropout so embeddings are deterministic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"ESM-2 model (dim={embedding_size}) loaded on {device}")
    return model, alphabet, batch_converter, device


def get_esm_embeddings(
    pep_tuple_list: List[Tuple[str, str]],
    esm_model: object,
    alphabet: object,
    batch_converter: object,
    device: str,
    embedding_layer: int = 33,
    sequence_embedding: bool = True,
) -> List[np.ndarray]:
    """Compute ESM-2 embeddings for a batch of sequences.

    Args:
        pep_tuple_list: Sequences as ``[(name_1, seq_1), (name_2, seq_2), ...]``.
        esm_model: Model returned by ``get_esm_model``.
        alphabet: Alphabet returned by ``get_esm_model``.
        batch_converter: Batch converter returned by ``get_esm_model``.
        device: Device string, ``"cuda"`` or ``"cpu"``.
        embedding_layer: Transformer layer whose representations are extracted.
        sequence_embedding: If ``True`` return one mean-pooled vector per
            sequence, otherwise the full per-residue matrix.

    Returns:
        One numpy array per input sequence.
    """
    _, _, batch_tokens = batch_converter(pep_tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[embedding_layer])
    token_representations = results["representations"][embedding_layer]

    # Token 0 is the beginning-of-sequence token, so residues start at index 1.
    representations = []
    for i, tokens_len in enumerate(batch_lens):
        embedding = token_representations[i, 1 : tokens_len - 1]
        if sequence_embedding:
            embedding = embedding.mean(dim=0)
        representations.append(embedding.cpu().numpy())

    return representations
