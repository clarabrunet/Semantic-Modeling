"""The C x S discriminative-dimension criterion."""

from __future__ import annotations

import numpy as np


def compute_CS(
    term_emb: np.ndarray,
    emb_icd_prop: np.ndarray,
    margen: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_dims = term_emb.shape[1]

   # cohesion
    var_asma = term_emb.var(axis=0)
    C = 1 - (var_asma - var_asma.min()) / (var_asma.max() - var_asma.min() + 1e-8)

    # Separation
    S = np.zeros(n_dims)
    for d in range(n_dims):
        vals_asma = term_emb[:, d]
        vals_icd = emb_icd_prop[:, d]
        mu_a = vals_asma.mean()
        std_a = vals_asma.std() + 1e-8
        min_r = mu_a - margen * std_a
        max_r = mu_a + margen * std_a
        mask_overlap = (vals_icd >= min_r) & (vals_icd <= max_r)
        if mask_overlap.sum() < 2:
            S[d] = 1.0
            continue
        vals_overlap = vals_icd[mask_overlap]
        mu_s = vals_overlap.mean()
        std_s = vals_overlap.std() + 1e-8
        std_pool = np.sqrt((std_a**2 + std_s**2) / 2)
        S[d] = np.abs(mu_a - mu_s) / std_pool

    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return C * S, C, S
