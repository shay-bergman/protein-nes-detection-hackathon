"""Parsing and loading of the NesDB CRM1 nuclear-export-signal dataset."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from nes.config import DB_CSV

Range = Tuple[int, int]


def extract_uniprot(row: pd.Series) -> str:
    """Return a clean UniProt accession for a NesDB row.

    The FASTA header is tried first (``sp|P15336|...`` style), then the first
    token of the ``ID`` column. The result is sanitised to letters, digits and
    underscores so it is safe as a dictionary key and filename.

    Args:
        row: A single row of the NesDB table.

    Returns:
        A sanitised UniProt accession string.
    """
    header = str(row["Fasta Header"])

    match = re.search(r"\|[st]r?\|([A-Z0-9]{4,10})(?:\.\d+)?\|", header)
    if not match:
        match = re.search(r"\|([A-Z0-9]{4,10})_[A-Z0-9]+", header)

    if match:
        acc = match.group(1)
    else:
        acc = re.sub(r"\(NES ID:.*", "", str(row["ID"])).strip().split()[0]

    return re.sub(r"[^A-Za-z0-9_]", "_", acc)


def parse_ranges(range_str: str) -> List[Range]:
    """Parse a NES range specification into 0-based, end-exclusive tuples.

    Accepts formats such as ``'405-413'``, ``'88-95; 101-112'`` and
    ``'[(400, 415)]'``.

    Args:
        range_str: The raw range string from the dataset.

    Returns:
        A list of ``(start, end)`` tuples, 0-based and end-exclusive.
    """
    matches = re.findall(r"(\d+)\D+?(\d+)", str(range_str))
    return [(int(a) - 1, int(b)) for a, b in matches]


def load_full_sequences(data_csv: str = DB_CSV) -> Tuple[Dict[str, str], Dict[str, List[Range]]]:
    """Load full-length protein sequences and their known NES ranges.

    Args:
        data_csv: Path to the NesDB CSV.

    Returns:
        A tuple ``(sequences, truth)`` keyed by UniProt accession, where
        ``truth`` maps each accession to its list of ground-truth NES ranges.
    """
    df = pd.read_csv(data_csv).dropna(subset=["Sequence", "Merged ranges"])
    sequences: Dict[str, str] = {}
    truth: Dict[str, List[Range]] = {}
    for _, row in df.iterrows():
        acc = extract_uniprot(row)
        sequences[acc] = row["Sequence"]
        truth[acc] = parse_ranges(row["Merged ranges"])
    return sequences, truth


def load_peptide_data(
    data_csv: str = DB_CSV,
    include_nesdoubt: bool = True,
    include_nodoubt: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    """Load positive/negative peptide pairs keyed by ``(accession, peptide)``.

    Args:
        data_csv: Path to the NesDB CSV.
        include_nesdoubt: Keep rows flagged as uncertain in NesDB.
        include_nodoubt: Keep rows flagged as confident in NesDB.

    Returns:
        A tuple ``(pos_dict, neg_dict, doubt_dict)`` sharing the same keys.
    """
    df = pd.read_csv(data_csv).dropna(
        subset=["Peptide_sequence", "Negative_sequence", "Sequence"]
    )

    if not include_nesdoubt:
        df = df[df["is_NesDB_doubt"] != True]  # noqa: E712 (pandas boolean mask)
    if not include_nodoubt:
        df = df[df["is_NesDB_doubt"] != False]  # noqa: E712

    pos_dict, neg_dict, doubt_dict = {}, {}, {}
    for _, row in df.iterrows():
        pep = row["Peptide_sequence"]
        neg = row["Negative_sequence"]
        if pep == "" or neg == "":
            continue

        key = (extract_uniprot(row), pep)
        pos_dict[key] = pep
        neg_dict[key] = neg
        doubt_dict[key] = bool(row["is_NesDB_doubt"])

    return pos_dict, neg_dict, doubt_dict


def load_peptide_data_lists(
    data_csv: str = DB_CSV,
    include_nesdoubt: bool = True,
    include_nodoubt: bool = True,
    max_peptide_len: int = 100,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[bool]]:
    """Load positive/negative peptides as ``(name, sequence)`` lists.

    This is the ``(name, seq)`` form expected by ``get_esm_embeddings``.

    Args:
        data_csv: Path to the NesDB CSV.
        include_nesdoubt: Keep rows flagged as uncertain in NesDB.
        include_nodoubt: Keep rows flagged as confident in NesDB.
        max_peptide_len: Discard peptides longer than this many residues.

    Returns:
        A tuple ``(positive_peptides, negative_peptides, is_doubt_flags)``.
    """
    df = pd.read_csv(data_csv).dropna(
        subset=["Peptide_sequence", "Negative_sequence", "Sequence"]
    )
    if not include_nesdoubt:
        df = df[df["is_NesDB_doubt"] != True].reset_index(drop=True)  # noqa: E712
    if not include_nodoubt:
        df = df[df["is_NesDB_doubt"] != False].reset_index(drop=True)  # noqa: E712

    pos_pep, neg_pep, doubt = [], [], []
    counter = 0
    for _, row in df.iterrows():
        pep, neg = row["Peptide_sequence"], row["Negative_sequence"]
        if pep and neg and len(pep) <= max_peptide_len and len(neg) <= max_peptide_len:
            pos_pep.append((str(counter), pep))
            neg_pep.append((str(counter), neg))
            doubt.append(row["is_NesDB_doubt"])
            counter += 1

    return pos_pep, neg_pep, doubt


def get_peptide_distances(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reduce_func: Callable = np.mean,
) -> np.ndarray:
    """Reduced Euclidean distance from each query to all reference embeddings.

    Args:
        query_embeddings: Embeddings to score (test peptides or segments).
        reference_embeddings: Reference embeddings (e.g. positive peptides).
        reduce_func: Reduction over the reference axis (``np.mean``, ``np.median``).

    Returns:
        One reduced distance per query embedding.
    """
    distances = cdist(
        np.asarray(query_embeddings), np.asarray(reference_embeddings), metric="euclidean"
    )
    return reduce_func(distances, axis=-1)
