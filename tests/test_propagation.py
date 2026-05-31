"""Tests for graph embedding propagation."""

import numpy as np
import pytest

nx = pytest.importorskip("networkx")
pytest.importorskip("scipy")

from asthma_e2.propagation import propagate_embeddings


def _line_graph(codes):
    G = nx.Graph()
    G.add_nodes_from(codes)
    for a, b in zip(codes, codes[1:]):
        G.add_edge(a, b)
    return G


def test_zero_steps_is_identity():
    codes = ["a", "b", "c"]
    emb = np.eye(3, dtype="float32")
    out = propagate_embeddings(_line_graph(codes), emb, codes, steps=0)
    assert np.allclose(out, emb)


def test_output_is_l2_normalised():
    codes = ["a", "b", "c", "d"]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(4, 8)).astype("float32")
    out = propagate_embeddings(_line_graph(codes), emb, codes, steps=2, alpha=0.7)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert out.shape == emb.shape


def test_propagation_smooths_neighbours():
    # Two connected nodes with opposite embeddings become more similar.
    codes = ["a", "b"]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    G = nx.Graph()
    G.add_edge("a", "b")
    out = propagate_embeddings(G, emb, codes, steps=3, alpha=0.5)
    cos_before = float(emb[0] @ emb[1])
    cos_after = float(out[0] @ out[1])
    assert cos_after > cos_before
