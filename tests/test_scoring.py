"""Tests for patient scoring and cross-validation."""

import numpy as np

from asthma_e2.data_loading import PatientData
from asthma_e2.evaluation import cv_evaluate
from asthma_e2.scoring import asthma_centroid, score_all_patients, score_patient


def _toy_term_emb(dim=8, m=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, dim)).astype("float32")


def test_centroid_is_unit_norm():
    term_emb = _toy_term_emb()
    c = asthma_centroid(term_emb)
    assert np.isclose(np.linalg.norm(c), 1.0, atol=1e-5)


def test_score_patient_is_bounded_cosine():
    term_emb = _toy_term_emb()
    dims = np.arange(term_emb.shape[1])
    centroid = asthma_centroid(term_emb)
    patient = np.random.default_rng(1).normal(size=(20, 8)).astype("float32")

    s = score_patient(patient, term_emb[:, dims], centroid, dims, top_n=5)
    assert -1.0 <= s <= 1.0


def _toy_patient_data(n_per_class=15, dim=8, seed=0):
    """Asthma patients embed near the reference terms; controls do not."""
    rng = np.random.default_rng(seed)
    term_emb = _toy_term_emb(dim=dim, seed=seed)
    centroid = asthma_centroid(term_emb)

    embeddings, words, nhc, ids, labels = {}, {}, {}, [], []
    for i in range(n_per_class):
        pid = f"pos_{i}"
        toks = centroid + 0.05 * rng.normal(size=(10, dim))
        embeddings[pid] = toks.astype("float32")
        words[pid] = [f"w{j}" for j in range(10)]
        nhc[pid] = pid
        ids.append(pid)
        labels.append(1)
    for i in range(n_per_class):
        pid = f"neg_{i}"
        toks = -centroid + 0.05 * rng.normal(size=(10, dim))
        embeddings[pid] = toks.astype("float32")
        words[pid] = [f"w{j}" for j in range(10)]
        nhc[pid] = pid
        ids.append(pid)
        labels.append(0)

    data = PatientData(
        ids=ids,
        labels=np.array(labels),
        embeddings=embeddings,
        words=words,
        nhc=nhc,
    )
    return data, term_emb


def test_score_all_patients_separates_classes():
    data, term_emb = _toy_patient_data()
    dims = np.arange(term_emb.shape[1])
    scores = score_all_patients(data, term_emb, dims, top_n=10)
    pos = scores[data.labels == 1].mean()
    neg = scores[data.labels == 0].mean()
    assert pos > neg


def test_cv_evaluate_runs_and_reports_metrics():
    data, term_emb = _toy_patient_data()
    dim = term_emb.shape[1]
    # Make every dimension discriminative so the subspace is non-empty.
    score_D = np.full(dim, 0.6)

    df = cv_evaluate(
        score_D, term_emb, data,
        umbrales=[0.2, 0.5], top_n=10, n_splits=3,
    )
    assert {"AUC", "F1", "Precision", "Recall", "Specificity"} <= set(df.columns)
    assert "mean" in df.index and "std" in df.index
    assert 0.0 <= df.loc["mean", "AUC"] <= 1.0
