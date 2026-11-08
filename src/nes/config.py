"""Central configuration: filesystem layout and default hyperparameters.

All paths are resolved relative to the project root (two levels above the
``nes`` package) and can be overridden with environment variables so that the
code runs unchanged on a different machine or data location.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(var: str, default: Path) -> Path:
    """Resolve a path from an environment variable.

    Args:
        var: Environment variable name.
        default: Path used when the variable is unset or empty.

    Returns:
        ``$var`` as a ``Path`` if set, otherwise ``default``.
    """
    value = os.environ.get(var)
    return Path(value) if value else default


# Raw NesDB CSV (not committed; see the Data section of the README).
DB_CSV = _path_from_env("NES_DB_CSV", PROJECT_ROOT / "data" / "NesDB_all_CRM1_with_peptides_train.csv")

# Generated artefacts (embeddings, segments, splits, scores).
DATA_DIR = _path_from_env("NES_DATA_DIR", PROJECT_ROOT / "data")
CACHE_DIR = _path_from_env("NES_CACHE_DIR", PROJECT_ROOT / "cache" / "per_residue")
RESULTS_DIR = _path_from_env("NES_RESULTS_DIR", PROJECT_ROOT / "results")

# A small, version-controlled example that lets the loaders run out of the box.
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample" / "nesdb_sample.csv"

# ---------------------------------------------------------------------------
# ESM-2 defaults
# ---------------------------------------------------------------------------
EMBEDDING_SIZE = 1280   # ESM-2 650M (esm2_t33_650M_UR50D)
EMBEDDING_LAYER = 33    # transformer layer whose representations are pooled

# ---------------------------------------------------------------------------
# Segmentation / labelling defaults
# ---------------------------------------------------------------------------
MAX_BKPS_PER_100AA = 10     # change-point budget for ZPS segmentation
NES_OVERLAP_MODE = "ratio"  # how a segment is labelled positive vs. a known NES
NES_OVERLAP_THRESHOLD = 0.5
MIN_SEGMENT_LEN = 8         # candidate NES length window (amino acids)
MAX_SEGMENT_LEN = 25


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed.

    Args:
        path: Directory to create (parents included).

    Returns:
        The same ``path``, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
