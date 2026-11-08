"""Nuclear Export Signal (NES) detection with ESM-2 embeddings.

The package is organised into focused sub-modules:

* ``nes.embeddings`` -- load ESM-2 and compute per-residue / per-sequence embeddings.
* ``nes.data``       -- parse NesDB, build the segmented database, create splits.
* ``nes.model``      -- the neural-network classifier and its training loop.
* ``nes.analysis``   -- consensus-pattern, embedding and combined scoring plus clustering
                        and structure-based analysis.
* ``nes.viz``        -- plotting helpers shared across the project.
"""

__version__ = "1.0.0"
