"""ICD-10 knowledge-graph construction"""

from __future__ import annotations

import numpy as np


def build_graph(
    df_icd10,
    emb_matrix: np.ndarray,
    codes: list[str],
    threshold: float,
    k_neighbors: int,
):

    import faiss
    import networkx as nx

    G = nx.Graph()
    for _, row in df_icd10.iterrows():
        G.add_node(row["code"], description=row["diag_descr"])

    # Hierarchical edges
    for code in G.nodes():
        parent = code.split(".")[0] if "." in code else None
        if parent and parent in G.nodes():
            G.add_edge(code, parent, type="hierarchy")
            n_hier += 1

    # Semantic edges
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix.astype("float32"))
    D, I = index.search(emb_matrix.astype("float32"), k_neighbors)
    mask = D > threshold
    rows_idx, cols_idx = np.where(mask)

    n_sem = 0
    for r, c in zip(rows_idx, cols_idx):
        i, j = int(r), int(I[r, c])
        if i != j:
            G.add_edge(codes[i], codes[j], type="semantic")
            n_sem += 1

    return G, n_hier, n_sem
