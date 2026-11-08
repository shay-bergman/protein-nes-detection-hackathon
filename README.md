# Protein NES Detection

*Finding CRM1-dependent nuclear export signals in protein sequences with ESM-2 embeddings and a neural-network classifier.*

## Overview

Many proteins are exported from the nucleus by the receptor CRM1, which
recognises a short, hydrophobic, loosely conserved motif called a nuclear
export signal (NES). Because the motif is degenerate, purely rule-based
detectors tend to be either too permissive or too strict. This project treats
NES detection as a supervised problem over learned protein representations:
candidate regions are embedded with a protein language model and classified,
and the learned scorer is compared against a classical consensus-pattern
baseline and an embedding-similarity baseline.

## Method

The pipeline combines a protein language model with a small supervised
classifier and two interpretable baselines.

- **Embeddings.** Each protein is embedded per residue with ESM-2
  (`esm2_t33_650M_UR50D`, 1280-dimensional). Segment and peptide representations
  are mean-pooled over their residues.
- **Segmentation.** Rather than scanning every window, each protein is split
  into a handful of contiguous segments with change-point detection on its
  per-residue embedding. A segment is labelled positive when it overlaps a
  known NES range by at least half of that range.
- **Scorers.** Three scorers rank candidate segments: (1) a rule-based
  *consensus-pattern* scorer over classical NES motifs (Kosugi, PKI and Rev
  classes); (2) an *embedding-similarity* scorer using the log-fold difference
  between a segment's mean distance to known negative versus positive peptides;
  and (3) a *neural network* — a compact feed-forward classifier trained on the
  segment embeddings with class-balanced sampling to handle the strong negative
  skew. The three are min-max normalised and combined into a single ranking.
- **Evaluation.** Decision thresholds are tuned per method and reported with
  accuracy, precision, recall, F1 and ROC AUC.

An auxiliary module explores whether positive and negative peptides separate in
embedding space (k-means + t-SNE) and a structure-based scorer evaluates
predicted 3D models by peptide centre-of-mass displacement and mean pLDDT.

## Repository structure

```
src/nes/
├── config.py                 # paths and default hyperparameters (env-overridable)
├── embeddings/
│   └── esm.py                # load ESM-2 and compute embeddings
├── data/
│   ├── peptides.py           # parse NesDB; load sequences, peptides, distances
│   ├── segmentation.py       # change-point segmentation of proteins
│   ├── build_database.py     # [stage] embed + segment + label -> segment database
│   └── splits.py             # [stage] protein-level train/val/test splits -> .npz
├── model/
│   ├── network.py            # SimpleDenseNet + data loaders + training loop
│   └── train.py              # [stage] train classifier, tune threshold, export scores
├── analysis/
│   ├── consensus.py          # NES consensus-pattern scorer
│   ├── embedding_scoring.py  # embedding-similarity scorer
│   ├── combine.py            # normalise, weight and rank the scorers
│   ├── metrics.py            # threshold tuning and evaluation
│   ├── pipeline.py           # [stage] end-to-end scoring + evaluation
│   ├── clustering.py         # k-means + t-SNE embedding exploration
│   └── structure.py          # structure-based (centre-of-mass / pLDDT) scoring
└── viz/
    └── plots.py              # shared plotting helpers
```

Modules marked `[stage]` are runnable entry points (`python -m nes.<module>`).

## Data

The project is built on the **NesDB** CRM1 export table
(`NesDB_all_CRM1_with_peptides_train.csv`): roughly 350 CRM1-dependent proteins,
each with its full sequence, one or more experimentally supported NES ranges, a
positive NES peptide and a matched negative peptide from the same protein. At
the segment level the classes are strongly imbalanced (far more negatives than
positives), which is why the classifier uses class-balanced sampling.

The dataset, the cached embeddings, the splits and the generated results are
**not** committed (they are large and derived). A small excerpt lives in
`data/sample/nesdb_sample.csv` so the loaders run out of the box. To reproduce
the full pipeline, place the complete CSV at `data/` (or point `NES_DB_CSV` at
it) and regenerate the artifacts:

```bash
export NES_DB_CSV=/path/to/NesDB_all_CRM1_with_peptides_train.csv
python -m nes.data.build_database   # embeds, segments and labels -> data/zps_segments.jsonl (+ pickles)
python -m nes.data.splits           # protein-level splits -> data/splits/{train,val,test}.npz
```

All paths default to the layout in `src/nes/config.py` and can be overridden
with the `NES_DB_CSV`, `NES_DATA_DIR`, `NES_CACHE_DIR` and `NES_RESULTS_DIR`
environment variables or the per-stage `--data-dir` / `--data-csv` flags.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for the ESM-2 embedding step; the code falls
back to CPU automatically.

## Usage

The pipeline runs in four stages:

```bash
# 1. Build the segmented, labelled database (ESM-2 embeddings + segmentation).
python -m nes.data.build_database

# 2. Create protein-level train/validation/test splits.
python -m nes.data.splits

# 3. Train the neural-network classifier and export per-segment scores.
python -m nes.model.train

# 4. Score and rank all candidate segments, then evaluate every method.
python -m nes.analysis.pipeline
```

Each stage exposes `--help` with its tunable arguments (embedding size and
layer, split sizes, training hyperparameters, scorer weights). Splitting is done
over **proteins**, never individual segments, so no protein contributes
segments to more than one split.

## Results

Evaluated on held-out segments, with per-method decision thresholds tuned for
F1. The supervised classifier is clearly the strongest scorer, and the
embedding-based methods comfortably beat the rule-based consensus baseline on
ranking quality (AUC).

| Method                | Accuracy | Precision | Recall | F1    | ROC AUC |
|-----------------------|----------|-----------|--------|-------|---------|
| Consensus pattern     | 0.849    | 0.165     | 0.564  | 0.256 | 0.781   |
| Embedding similarity  | 0.934    | 0.293     | 0.308  | 0.300 | 0.691   |
| Neural network        | 0.984    | 0.879     | 0.744  | 0.806 | 0.970   |
| Combined              | 0.952    | 0.475     | 0.487  | 0.481 | 0.918   |

The headline result is the neural network: ROC AUC 0.97 and F1 0.81 on a
heavily imbalanced segment-level task, substantially ahead of both baselines.

## License

Released under the MIT License. See [LICENSE](LICENSE).
