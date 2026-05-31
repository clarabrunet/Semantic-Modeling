"""Embedding propagation over the ICD-10 knowledge graph"""

from __future__ import annotations

import numpy as np


def propagate_embeddings(
    G,
    emb_matrix: np.ndarray,
    codes: list[str],
    steps: int = 2,
    alpha: float = 0.7,
    verbose: bool = False,
) -> np.ndarray:

    from scipy.sparse import csr_matrix, diags, lil_matrix

    n = len(codes)
    code_to_idx = {c: i for i, c in enumerate(codes)}

    A = lil_matrix((n, n), dtype=np.float32)
    for u, v in G.edges():
        i = code_to_idx.get(u)
        j = code_to_idx.get(v)
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0
    A = csr_matrix(A)

    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    A_norm = diags(1.0 / row_sums) @ A

    emb = emb_matrix.copy().astype(np.float32)
    for step in range(steps):
        emb = alpha * emb + (1 - alpha) * (A_norm @ emb)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = emb / norms
        if verbose:
            print(f"  Step {step + 1}/{steps} completed")
    return emb
