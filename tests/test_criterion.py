"""Tests for the C x S discriminative criterion."""

import numpy as np

from asthma_e2.criterion import compute_CS


def test_shapes_and_ranges():
    rng = np.random.default_rng(0)
    term_emb = rng.normal(size=(19, 16)).astype("float32")
    emb_icd = rng.normal(size=(200, 16)).astype("float32")

    D, C, S = compute_CS(term_emb, emb_icd, margen=1.0)

    assert D.shape == C.shape == S.shape == (16,)
    # C and S are min-max normalised into [0, 1]; D is their product.
    assert C.min() >= 0.0 and C.max() <= 1.0 + 1e-6
    assert S.min() >= 0.0 and S.max() <= 1.0 + 1e-6
    assert np.allclose(D, C * S)


def test_cohesion_high_for_low_variance_dimension():
    # Dimension 0 has (near) zero variance across reference terms -> high C.
    term_emb = np.ones((10, 4), dtype="float32")
    term_emb[:, 1:] += np.random.default_rng(1).normal(size=(10, 3))
    emb_icd = np.random.default_rng(2).normal(size=(50, 4)).astype("float32")

    _, C, _ = compute_CS(term_emb, emb_icd)
    assert C[0] == C.max()


def test_separation_defaults_when_no_overlap():
   
    term_emb = np.zeros((5, 3), dtype="float32")
    emb_icd = np.full((20, 3), 100.0, dtype="float32")  
    D, C, S = compute_CS(term_emb, emb_icd, margen=1.0)
    assert np.all(np.isfinite(D))
