from __future__ import annotations

import numpy as np


def score_patient(
    patient_emb: np.ndarray,
    term_emb_dims: np.ndarray,
    centroid_full: np.ndarray,
    dims: np.ndarray,
    top_n: int = 50,
) -> float:

    emb_p = patient_emb

  
    norms = np.linalg.norm(emb_p, axis=1, keepdims=True) + 1e-8
    sims_c = (emb_p / norms) @ centroid_full
    if len(sims_c) > top_n:
        top_idx = np.argsort(sims_c)[-top_n:]
        emb_p = emb_p[top_idx]


    emb_p_proj = emb_p[:, dims]

    norms_p = np.linalg.norm(emb_p_proj, axis=1, keepdims=True) + 1e-8
    norms_t = np.linalg.norm(term_emb_dims, axis=1, keepdims=True) + 1e-8
    sim = (emb_p_proj / norms_p) @ (term_emb_dims / norms_t).T
    return float(sim.max(axis=0).mean())


def asthma_centroid(term_emb_raw: np.ndarray) -> np.ndarray:
    centroid = term_emb_raw.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8
    return centroid


def score_all_patients(
    patient_data,
    term_emb_raw: np.ndarray,
    dims: np.ndarray,
    top_n: int = 50,
) -> np.ndarray:

    centroid_full = asthma_centroid(term_emb_raw)
    term_emb_dims = term_emb_raw[:, dims]
    return np.array(
        [
            score_patient(
                patient_data.embeddings[pid],
                term_emb_dims,
                centroid_full,
                dims,
                top_n=top_n,
            )
            for pid in patient_data.ids
        ]
    )
