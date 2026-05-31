"""Cross-validation  evaluation"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import CONFIG
from .scoring import score_all_patients


def _select_threshold(y_tr, s_tr, criterion: str) -> float:

    p_tr, r_tr, thr_tr = precision_recall_curve(y_tr, s_tr)
    if criterion == "f1":
        f1_tr = 2 * p_tr * r_tr / (p_tr + r_tr + 1e-8)
        if len(thr_tr) == 0:
            return float(s_tr.mean())
        return float(thr_tr[min(np.argmax(f1_tr[:-1]), len(thr_tr) - 1)])
    # recall = 1.0
    mask = r_tr[:-1] >= 1.0
    if mask.any():
        return float(thr_tr[mask][np.argmax(p_tr[:-1][mask])])
    return float(thr_tr.min())


def cv_evaluate(
    score_D: np.ndarray,
    term_emb_raw: np.ndarray,
    patient_data,
    *,
    umbrales=None,
    top_n: int = CONFIG.top_n_tokens,
    umbral_criterio: str = "f1",
    n_splits: int = CONFIG.cv_n_splits,
    random_state: int = CONFIG.random_state,
) -> pd.DataFrame:

    if umbrales is None:
        umbrales = CONFIG.umbral_range

    labels_all = patient_data.labels
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    scores_u: dict[float, np.ndarray] = {}
    for u in umbrales:
        dims_u = np.where(score_D > u)[0]
        if len(dims_u) < 2:
            continue
        scores_u[u] = score_all_patients(patient_data, term_emb_raw, dims_u, top_n)

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(
        skf.split(labels_all, labels_all), 1
    ):
        # Select best subspace threshold on the training fold
        best_score = -1.0
        best_u = next(iter(scores_u))
        for u, sc in scores_u.items():
            y_tr, s_tr = labels_all[tr_idx], sc[tr_idx]
            p_tr, r_tr, _ = precision_recall_curve(y_tr, s_tr)
            if umbral_criterio == "f1":
                f1_tr = 2 * p_tr * r_tr / (p_tr + r_tr + 1e-8)
                val = f1_tr[:-1].max() if len(f1_tr) > 1 else 0.0
            else:
                val = roc_auc_score(y_tr, s_tr)
            if val > best_score:
                best_score = val
                best_u = u

        sc = scores_u[best_u]
        y_tr, y_te = labels_all[tr_idx], labels_all[te_idx]
        s_tr, s_te = sc[tr_idx], sc[te_idx]
        threshold = _select_threshold(y_tr, s_tr, umbral_criterio)
        preds = (s_te >= threshold).astype(int)

        rows.append(
            {
                "fold": fold,
                "best_tau": best_u,
                "n_dims": int(len(np.where(score_D > best_u)[0])),
                "AUC": roc_auc_score(y_te, s_te),
                "AUPR": average_precision_score(y_te, s_te),
                "Precision": precision_score(y_te, preds, zero_division=0),
                "Recall": recall_score(y_te, preds, zero_division=0),
                "F1": f1_score(y_te, preds, zero_division=0),
                "Specificity": recall_score(
                    y_te, preds, pos_label=0, zero_division=0
                ),
            }
        )

    df = pd.DataFrame(rows).set_index("fold")
    mean_r = df.mean(numeric_only=True).to_frame().T
    std_r = df.std(numeric_only=True).to_frame().T
    mean_r.index = ["mean"]
    std_r.index = ["std"]
    return pd.concat([df, mean_r, std_r])
