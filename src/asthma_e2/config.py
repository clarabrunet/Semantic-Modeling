"""Central configuration for the Strategy 2 """

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout                                                              #
# --------------------------------------------------------------------------- #
# Repository root = two levels up from this file (src/asthma_e2/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ASTHMA_E2_DATA", ROOT_DIR / "data"))
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


@dataclass(frozen=True)
class Paths:
    """File-system locations of the (private) input artefacts.

    The clinical data are **not** distributed with this repository.
    """

    data_dir: Path = DATA_DIR
    # ICD-10 catalogue (CSV) used to build the knowledge graph.
    icd10_catalogue: Path = DATA_DIR / "dic_diagnostic.csv"
    # Pre-computed [CLS] embeddings of every patient's tokens.
    patient_embeddings: Path = DATA_DIR / "word_embeddings_cls.pkl"
    # Cached ICD-10 code embeddings ({"codes": [...], "emb_matrix": ndarray}).
    icd10_embeddings: Path = DATA_DIR / "emb_icd10_raw.pkl"
    # NHC -> folder-id mapping (CSV with NHC, id and DODx columns).
    nhc_mapping: Path = DATA_DIR / "nhc_mapping.csv"
    # Output locations.
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR


@dataclass(frozen=True)
class Config:
    """Hyperparameters of the pipline"""

    # Language model used to embed reference terms (and, offline, patient text).
    model_name: str = "PlanTL-GOB-ES/bsc-bio-ehr-es"
    max_length: int = 32
    batch_size: int = 64

    # Reference ICD-10 knowledge graph.
    graph_threshold: float = 0.75
    graph_k_neighbors: int = 10

    # Embedding propagation over the graph.
    propagation_steps: int = 2
    propagation_alpha: float = 0.7

    # C x S discriminative criterion and patient scoring.
    margin_gamma: float = 1.0
    top_n_tokens: int = 50

    # Cross-validation.
    cv_n_splits: int = 5
    random_state: int = 42

    # Sweeps used by the ablation experiments.
    umbral_range: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    k_range: tuple[int, ...] = (10, 20, 30, 40, 50, 75, 100, 150, 200, 768)


REFERENCE_TERMS: tuple[str, ...] = (
    "tos",
    "disnea",
    "asma",
    "exacerbacion",
    "inhaladores",
    "sibilancias",
    "expectoracion",
    "agudizacion",
    "astenia",
    "hiperreactividad bronquial",
    "hipoxemia",
    "fiebre",
    "taquipnea",
    "broncoespasmo",
    "crisis asmatica",
    "uso de musculatura accesoria",
    "tiraje",
    "febricula",
    "bronquitis asmatica",
)


CONFIG = Config()
PATHS = Paths()
