"""Strategy 2 (E2-B): graph- and embedding-based asthma detection in EHRs.

A weakly-supervised screening pipeline that detects asthma from free-text
Spanish clinical notes by:

1. embedding ICD-10 codes and reference asthma terms with a Spanish clinical
   transformer (``bsc-bio-ehr-es``);
2. propagating code embeddings over an ICD-10 knowledge graph;
3. selecting a discriminative subspace with the ``C x S`` criterion;
4. scoring patients by semantic recall against the reference terminology;
5. calibrating a single threshold under stratified cross-validation.

See the module docstrings and ``notebooks/results.ipynb`` for details.
"""

from __future__ import annotations

from .config import CONFIG, PATHS, REFERENCE_TERMS, Config, Paths
from .criterion import compute_CS
from .evaluation import cv_evaluate
from .graph import build_graph
from .propagation import propagate_embeddings
from .scoring import asthma_centroid, score_all_patients, score_patient

__all__ = [
    "CONFIG",
    "PATHS",
    "REFERENCE_TERMS",
    "Config",
    "Paths",
    "compute_CS",
    "cv_evaluate",
    "build_graph",
    "propagate_embeddings",
    "asthma_centroid",
    "score_all_patients",
    "score_patient",
]

__version__ = "1.0.0"
