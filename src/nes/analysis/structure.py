"""Structure-based scoring of candidate peptides from predicted 3D models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from scipy.spatial.distance import cdist
from tqdm import tqdm

from nes.viz.plots import plot_boxplot, plot_roc_curve


def load_structure(path: Path):
    """Read a PDB or mmCIF file.

    Args:
        path: Path to a ``.pdb`` or ``.cif`` structure file.

    Returns:
        A ``Bio.PDB.Structure`` object parsed from the file.
    """
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def get_peptide_atoms(structure, chain_id: str) -> List:
    """Return all atoms of a peptide chain in the first model.

    Args:
        structure: A parsed ``Bio.PDB.Structure``.
        chain_id: Chain identifier of the peptide.

    Returns:
        A list of ``Bio.PDB.Atom`` objects for the chain.
    """
    chain = structure[0][chain_id]  # first model assumed
    return [atom for residue in chain for atom in residue]


def get_interface_atoms(structure, pep_chain_id: str, prot_chain_id: str, cutoff: float = 5.0) -> List:
    """Return peptide atoms within ``cutoff`` angstroms of the protein chain.

    Args:
        structure: A parsed ``Bio.PDB.Structure``.
        pep_chain_id: Chain identifier of the peptide.
        prot_chain_id: Chain identifier of the receptor protein.
        cutoff: Distance cutoff in angstroms defining the interface.

    Returns:
        A list of peptide ``Bio.PDB.Atom`` objects at the interface.
    """
    pep_atoms = get_peptide_atoms(structure, pep_chain_id)
    prot_atoms = get_peptide_atoms(structure, prot_chain_id)
    dist = cdist([a.coord for a in pep_atoms], [a.coord for a in prot_atoms], metric="euclidean")
    mask = dist.min(axis=1) < cutoff
    return [atom for atom, keep in zip(pep_atoms, mask) if keep]


def mean_plddt(atoms: Iterable) -> float:
    """Return the mean pLDDT confidence, read from atom B-factors.

    Args:
        atoms: Iterable of ``Bio.PDB.Atom`` objects.

    Returns:
        The mean B-factor (pLDDT) across the atoms.
    """
    return float(np.mean([atom.get_bfactor() for atom in atoms]))


def centre_of_mass(atoms: Iterable) -> np.ndarray:
    """Return the geometric centre of a set of atoms.

    Args:
        atoms: Iterable of ``Bio.PDB.Atom`` objects.

    Returns:
        A length-3 numpy array with the centre-of-mass coordinates (angstroms).
    """
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    return coords.mean(axis=0)


def score_structures(
    reference_pdb: Path,
    positives_dir: Path,
    negatives_dir: Path,
    reference_chain: str = "D",
    model_chain: str = "B",
    results_dir: Path = Path("."),
) -> dict:
    """Score predicted peptide models by centre-of-mass distance and pLDDT.

    For every predicted model the peptide centre of mass is compared with the
    reference bound-state centre of mass; a smaller displacement is treated as
    more NES-like. Mean pLDDT is evaluated as a second, confidence-based signal.

    Args:
        reference_pdb: Path to the reference complex structure.
        positives_dir: Directory of predicted positive peptide structures.
        negatives_dir: Directory of predicted negative peptide structures.
        reference_chain: Peptide chain identifier in the reference structure.
        model_chain: Peptide chain identifier in the predicted models.
        results_dir: Directory where ROC curves and box plots are written.

    Returns:
        A dict with arrays ``distances``, ``plddts`` and ``labels``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ref_atoms = get_peptide_atoms(load_structure(reference_pdb), reference_chain)
    ref_com = centre_of_mass(ref_atoms)

    distances, plddts, labels = [], [], []
    for label, folder in ((1, positives_dir), (0, negatives_dir)):
        models = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".pdb", ".cif"))
        ]
        for model_path in tqdm(models, desc=f"class {label}"):
            atoms = get_peptide_atoms(load_structure(model_path), model_chain)
            plddts.append(mean_plddt(atoms))
            distances.append(float(np.linalg.norm(centre_of_mass(atoms) - ref_com)))
            labels.append(label)

    distances = np.array(distances)
    plddts = np.array(plddts)
    labels = np.array(labels)
    neg_distance_score = -distances  # larger score should mean more NES-like

    plot_roc_curve(labels, neg_distance_score, results_dir / "com_roc_curve.png")
    plot_roc_curve(labels, plddts, results_dir / "plddt_roc_curve.png")
    plot_boxplot(
        {"Positive": neg_distance_score[labels == 1], "Negative": neg_distance_score[labels == 0]},
        results_dir / "com_boxplot.png",
    )
    plot_boxplot(
        {"Positive": plddts[labels == 1], "Negative": plddts[labels == 0]},
        results_dir / "plddt_boxplot.png",
    )
    return {"distances": distances, "plddts": plddts, "labels": labels}
