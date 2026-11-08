"""Protein-level train / validation / test splitting.

Splits are made over **proteins**, never over individual segments, so that no
protein contributes segments to more than one split (which would leak
information between training and evaluation). The segment JSONL produced by
:mod:`nes.data.build_database` is partitioned accordingly and then converted to
compact ``.npz`` arrays for the neural network.

Run as a script::

    python -m nes.data.splits
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from nes import config
from nes.data.peptides import load_peptide_data


def split_proteins_by_id(
    protein_ids,
    test_size: float = 0.15,
    dev_size: float = 0.15,
    random_state: int = 17,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Partition protein IDs into disjoint train/val/test ID sets.

    Args:
        protein_ids: Iterable of protein accession strings.
        test_size: Fraction of proteins held out for testing.
        dev_size: Fraction of proteins held out for validation.
        random_state: Seed for the shuffled split.

    Returns:
        A tuple ``(train_ids, val_ids, test_ids)`` of accession sets.
    """
    proteins = sorted(set(protein_ids))

    train_prot, temp_prot = train_test_split(
        proteins, test_size=(dev_size + test_size),
        random_state=random_state, shuffle=True,
    )
    rel_test = test_size / (dev_size + test_size)
    dev_prot, test_prot = train_test_split(
        temp_prot, test_size=rel_test, random_state=random_state, shuffle=True,
    )
    return set(train_prot), set(dev_prot), set(test_prot)


def partition_segments_jsonl(
    segments_jsonl: Path,
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
    out_dir: Path,
) -> Dict[str, Path]:
    """Dispatch each segment record to its split by protein accession.

    Args:
        segments_jsonl: Path to the combined segment JSONL file.
        train_ids: Accessions assigned to the training split.
        val_ids: Accessions assigned to the validation split.
        test_ids: Accessions assigned to the test split.
        out_dir: Directory for the per-split JSONL files.

    Returns:
        A mapping from split name to its written JSONL path.
    """
    config.ensure_dir(out_dir)
    paths = {
        "train": out_dir / "zps_segments_train.jsonl",
        "val": out_dir / "zps_segments_val.jsonl",
        "test": out_dir / "zps_segments_test.jsonl",
    }
    handles = {name: path.open("w") for name, path in paths.items()}
    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    with open(segments_jsonl) as fh:
        for line in fh:
            acc = json.loads(line)["id"]
            if acc in train_ids:
                bucket = "train"
            elif acc in val_ids:
                bucket = "val"
            elif acc in test_ids:
                bucket = "test"
            else:
                counts["skipped"] += 1
                continue
            handles[bucket].write(line)
            counts[bucket] += 1

    for handle in handles.values():
        handle.close()

    print("ZPS-segment rows dispatched:")
    for name, count in counts.items():
        print(f"  {name:8s}: {count}")
    return paths


def jsonl_to_npz(jsonl_path: Path, npz_path: Path) -> None:
    """Convert a segment JSONL file to a compressed ``{X, y, keys}.npz``.

    Args:
        jsonl_path: Path to a per-split segment JSONL file.
        npz_path: Output ``.npz`` path.
    """
    jsonl_path, npz_path = Path(jsonl_path), Path(npz_path)
    keys, feats, labels = [], [], []
    with open(jsonl_path) as fh:
        for line in fh:
            obj = json.loads(line)
            keys.append(f"{obj['id']} {obj['coords'][0]}-{obj['coords'][1]}")
            feats.append(obj["embedding"])
            labels.append(obj["label"])

    np.savez_compressed(
        npz_path,
        X=np.asarray(feats, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int8),
        keys=np.asarray(keys, dtype=object),
    )
    print(f"{npz_path.name:16s} -> {len(labels):6,d} rows")


def build_splits(
    data_csv: Path = config.DB_CSV,
    data_dir: Path = config.DATA_DIR,
    test_size: float = 0.15,
    dev_size: float = 0.15,
    random_state: int = 17,
) -> None:
    """Create protein-level splits of the segment database.

    Args:
        data_csv: Path to the NesDB CSV (used to enumerate proteins).
        data_dir: Directory holding ``zps_segments.jsonl`` and split outputs.
        test_size: Fraction of proteins held out for testing.
        dev_size: Fraction of proteins held out for validation.
        random_state: Seed for the shuffled split.
    """
    data_dir = Path(data_dir)

    # Collect every protein accession that appears in the peptide table.
    pos_dict, neg_dict, _ = load_peptide_data(
        data_csv=data_csv, include_nesdoubt=True, include_nodoubt=True
    )
    prot_to_pos, prot_to_neg = defaultdict(list), defaultdict(list)
    for acc, _ in pos_dict:
        prot_to_pos[acc].append(acc)
    for acc, _ in neg_dict:
        prot_to_neg[acc].append(acc)
    protein_ids = list(prot_to_pos) + list(prot_to_neg)

    train_ids, val_ids, test_ids = split_proteins_by_id(
        protein_ids, test_size=test_size, dev_size=dev_size, random_state=random_state
    )
    print(
        f"proteins  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}"
    )

    split_dir = config.ensure_dir(data_dir / "zps_split")
    split_paths = partition_segments_jsonl(
        data_dir / "zps_segments.jsonl", train_ids, val_ids, test_ids, split_dir
    )

    npz_dir = config.ensure_dir(data_dir / "splits")
    for name, jsonl_path in split_paths.items():
        jsonl_to_npz(jsonl_path, npz_dir / f"{name}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create protein-level data splits.")
    parser.add_argument("--data-csv", type=Path, default=config.DB_CSV)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()
    build_splits(
        data_csv=args.data_csv,
        data_dir=args.data_dir,
        test_size=args.test_size,
        dev_size=args.dev_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
