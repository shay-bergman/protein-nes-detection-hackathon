"""Build the segmented NES database.

Pipeline:

1. load full-length sequences and their ground-truth NES ranges;
2. embed every protein once with ESM-2 (cached per protein on disk);
3. split each protein into segments with change-point detection (ZPS);
4. mean-pool and label each segment (positive if it overlaps a known NES),
   writing one JSON object per segment to ``zps_segments.jsonl``.

Run as a script::

    python -m nes.data.build_database
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import tqdm

from nes import config
from nes.data.peptides import load_full_sequences
from nes.data.segmentation import get_protein_segment_embeddings, get_protein_segments
from nes.embeddings import get_esm_embeddings, get_esm_model

Range = List[int]


def has_nes_overlap(start, end, nes_ranges, mode: str = "ratio", thresh: float = 0.5) -> bool:
    """Decide whether a ``[start, end)`` segment counts as a positive NES hit.

    Args:
        start: Segment start index (inclusive).
        end: Segment end index (exclusive).
        nes_ranges: Ground-truth NES ranges for the protein.
        mode: ``"contains"`` (segment fully covers a NES range), ``"any"``
            (at least one residue of overlap), or ``"ratio"`` (overlap covers
            at least ``thresh`` of a NES range).
        thresh: Overlap fraction required when ``mode="ratio"``.

    Returns:
        ``True`` if the segment is labelled a positive NES hit.
    """
    for ns, ne in nes_ranges:
        overlap = max(0, min(end, ne) - max(start, ns))
        if mode == "contains" and start <= ns and end >= ne:
            return True
        if mode == "any" and overlap > 0:
            return True
        if mode == "ratio" and (ne - ns) > 0 and overlap / (ne - ns) >= thresh:
            return True
    return False


def build_database(
    data_csv: Path = config.DB_CSV,
    out_dir: Path = config.DATA_DIR,
    cache_dir: Path = config.CACHE_DIR,
    embedding_size: int = config.EMBEDDING_SIZE,
    embedding_layer: int = config.EMBEDDING_LAYER,
    max_bkps_per100aa: int = config.MAX_BKPS_PER_100AA,
    overlap_mode: str = config.NES_OVERLAP_MODE,
    overlap_thresh: float = config.NES_OVERLAP_THRESHOLD,
) -> Path:
    """Build the segmented database from raw sequences.

    Args:
        data_csv: Path to the NesDB CSV.
        out_dir: Directory for generated artefacts (pickles and JSONL).
        cache_dir: Directory for cached per-protein embedding matrices.
        embedding_size: ESM-2 embedding width.
        embedding_layer: ESM-2 layer to pool.
        max_bkps_per100aa: Change-point budget per 100 residues.
        overlap_mode: Labelling mode passed to ``has_nes_overlap``.
        overlap_thresh: Overlap fraction for ``ratio`` labelling.

    Returns:
        The path to the written ``zps_segments.jsonl`` file.
    """
    out_dir = config.ensure_dir(Path(out_dir))
    # Key the embedding cache by model config so changing the ESM size or layer
    # never silently reuses vectors computed for a different configuration.
    cache_dir = config.ensure_dir(Path(cache_dir) / f"esm{embedding_size}_l{embedding_layer}")
    jsonl_out = out_dir / "zps_segments.jsonl"

    print("Loading full-length sequences ...")
    sequences, truth = load_full_sequences(data_csv)
    with open(out_dir / "protein_sequences.pkl", "wb") as fh:
        pickle.dump(sequences, fh)

    print("Loading ESM-2 model ...")
    model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    # Embed each protein once, caching the per-residue matrix on disk.
    prot_emb = {}
    for acc, seq in tqdm.tqdm(sequences.items(), desc="embedding proteins"):
        npy_path = cache_dir / f"{acc}.npy"
        if npy_path.exists():
            prot_emb[acc] = np.load(npy_path)
            continue
        vec = get_esm_embeddings(
            [(acc, seq)], model, alphabet, batch_converter, device,
            embedding_layer=embedding_layer, sequence_embedding=False,
        )[0].astype("float32")
        np.save(npy_path, vec)
        prot_emb[acc] = vec
    print(f"Per-residue cache ready ({len(prot_emb)} proteins)")

    print("Running ZPS change-point detection ...")
    segments = get_protein_segments(prot_emb, max_bkps_per100aa=max_bkps_per100aa)

    print("Pooling segments and writing JSONL ...")
    with open(jsonl_out, "w") as fh:
        for acc, seg_list in segments.items():
            if seg_list == "Failed":
                continue
            for start, end in seg_list:
                vec = prot_emb[acc][start:end].mean(0)
                hit = has_nes_overlap(start, end, truth[acc], mode=overlap_mode, thresh=overlap_thresh)
                fh.write(json.dumps({
                    "id": acc,
                    "coords": [int(start), int(end)],
                    "label": int(hit),
                    "embedding": vec.tolist(),
                }) + "\n")

    n_segments = sum(1 for _ in open(jsonl_out))
    print(f"Finished: {jsonl_out} contains {n_segments} candidate segments")

    segment_embeddings = get_protein_segment_embeddings(prot_emb, segments)
    with open(out_dir / "protein_segments.pkl", "wb") as fh:
        pickle.dump(segments, fh)
    with open(out_dir / "protein_segment_embeddings.pkl", "wb") as fh:
        pickle.dump(segment_embeddings, fh)

    return jsonl_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the segmented NES database.")
    parser.add_argument("--data-csv", type=Path, default=config.DB_CSV)
    parser.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--embedding-size", type=int, default=config.EMBEDDING_SIZE)
    parser.add_argument("--embedding-layer", type=int, default=config.EMBEDDING_LAYER)
    parser.add_argument("--max-bkps-per-100aa", type=int, default=config.MAX_BKPS_PER_100AA)
    args = parser.parse_args()

    build_database(
        data_csv=args.data_csv,
        out_dir=args.out_dir,
        embedding_size=args.embedding_size,
        embedding_layer=args.embedding_layer,
        max_bkps_per100aa=args.max_bkps_per_100aa,
    )


if __name__ == "__main__":
    main()
