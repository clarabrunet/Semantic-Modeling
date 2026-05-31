from __future__ import annotations

import pickle
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, PATHS


@dataclass
class PatientData:

    ids: list[str]
    labels: np.ndarray  # shape (n_patients,), {0, 1}
    embeddings: dict[str, np.ndarray]  # patient_id -> (n_tokens, hidden_size)
    words: dict[str, list[str]]  # patient_id -> token strings
    nhc: dict[str, str]  # patient_id -> original NHC


def load_icd10_catalogue(path: Path = PATHS.icd10_catalogue) -> pd.DataFrame:

    with open(path, encoding="latin1") as f:
        fixed_lines = []
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            fixed_lines.append(line)

    df = pd.read_csv(StringIO("\n".join(fixed_lines)))
    df = df[df["catalog"] == 12].copy()
    df = df.loc[
        (df["diag_descr"] != "")
        & (df["diag_descr"] != "P.D.")
        & (df["diag_descr"] != "N.D.")
        & (~df["diag_descr"].str.contains(r"gestac|embarazo", regex=True))
    ].reset_index(drop=True)
    df["code"] = df["code"].astype(str)
    df["category"] = df["code"].str[:3]
    return df


def load_icd10_embeddings(
    path: Path = PATHS.icd10_embeddings,
) -> tuple[list[str], np.ndarray]:

    with open(path, "rb") as f:
        cache = pickle.load(f)
    codes = list(cache["codes"])
    emb_matrix = np.asarray(cache["emb_matrix"], dtype="float32")
    return codes, emb_matrix


def load_nhc_mapping(path: Path = PATHS.nhc_mapping) -> dict[str, str]:
 
    df = pd.read_csv(path, sep=None, engine="python")
    df["DODx"] = pd.to_datetime(df["DODx"], dayfirst=True)
    df = df.sort_values("DODx").drop_duplicates(subset="NHC", keep="first")
    return dict(zip(df["NHC"].astype(str), df["id"].astype(str)))


def load_patient_data(
    path: Path = PATHS.patient_embeddings,
    nhc_mapping: dict[str, str] | None = None,
    exclude_patients: frozenset[str] = CONFIG.exclude_patients,
) -> PatientData:
    """Load per-patient token embeddings from the pre-computed pickle"""

    if nhc_mapping is None:
        nhc_mapping = {}

    with open(path, "rb") as f:
        word_emb_cls = pickle.load(f)

    embeddings: dict[str, np.ndarray] = {}
    words: dict[str, list[str]] = {}
    nhc: dict[str, str] = {}
    labels: list[int] = []
    ids: list[str] = []

    for raw_nhc, record in word_emb_cls.items():
        nhc_str = str(raw_nhc)
        if nhc_str in exclude_patients:
            continue
        pid = nhc_mapping.get(nhc_str, nhc_str)
        embeddings[pid] = np.asarray(record["embeddings_raw"], dtype="float32")
        words[pid] = record["words"]
        nhc[pid] = nhc_str
        ids.append(pid)
        labels.append(int(record["label"]))

    return PatientData(
        ids=ids,
        labels=np.asarray(labels, dtype=int),
        embeddings=embeddings,
        words=words,
        nhc=nhc,
    )
