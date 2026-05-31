"""Tests for ICD-10 graph construction."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")
pytest.importorskip("networkx")

from asthma_e2.graph import build_graph


def test_hierarchical_and_semantic_edges():
    df = pd.DataFrame(
        {
            "code": ["J45", "J45.0", "J44"],
            "diag_descr": ["asthma", "predominantly allergic asthma", "copd"],
        }
    )
    codes = df["code"].tolist()
    # J45 and J45.0 nearly identical; J44 orthogonal.
    emb = np.array(
        [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]], dtype="float32"
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    G, n_hier, n_sem = build_graph(df, emb, codes, threshold=0.9, k_neighbors=3)

    # J45.0 -> J45 hierarchical edge.
    assert G.has_edge("J45.0", "J45")
    assert n_hier == 1
    # At least one semantic edge between the two similar asthma codes.
    assert n_sem >= 1
    assert G.number_of_nodes() == 3
