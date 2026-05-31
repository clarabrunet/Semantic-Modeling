"""Run"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asthma_e2.config import CONFIG, PATHS, REFERENCE_TERMS  # noqa: E402
from asthma_e2.criterion import compute_CS  # noqa: E402
from asthma_e2.data_loading import (  # noqa: E402
    load_icd10_catalogue,
    load_icd10_embeddings,
    load_nhc_mapping,
    load_patient_data,
)
from asthma_e2.embeddings import embed_batch  # noqa: E402
from asthma_e2.evaluation import cv_evaluate  # noqa: E402
from asthma_e2.graph import build_graph  # noqa: E402
from asthma_e2.propagation import propagate_embeddings  # noqa: E402


def main() -> None:
    print("1/6  Loading input artefacts...")
    df_icd10 = load_icd10_catalogue()
    codes, emb_icd10 = load_icd10_embeddings()
    nhc_mapping = load_nhc_mapping()
    patients = load_patient_data(nhc_mapping=nhc_mapping)
    n_pos = int(patients.labels.sum())
    print(f"     ICD-10 codes: {len(codes):,} | patients: {len(patients.ids)} "
          f"(asthma={n_pos}, non-asthma={len(patients.ids) - n_pos})")

    print("2/6  Embedding reference terms...")
    term_emb = embed_batch(list(REFERENCE_TERMS))

    print("3/6  Building ICD-10 knowledge graph...")
    G, n_hier, n_sem = build_graph(
        df_icd10, emb_icd10, codes,
        CONFIG.graph_threshold, CONFIG.graph_k_neighbors,
    )
    print(f"     nodes={G.number_of_nodes():,} | "
          f"hierarchical={n_hier:,} | semantic={n_sem:,}")

    print("4/6  Propagating embeddings over the graph...")
    emb_prop = propagate_embeddings(
        G, emb_icd10, codes,
        CONFIG.propagation_steps, CONFIG.propagation_alpha, verbose=True,
    )

    print("5/6  Computing C x S discriminative criterion...")
    score_D, C, S = compute_CS(term_emb, emb_prop, margen=CONFIG.margin_gamma)
    print(f"     dims D>0.20: {(score_D > 0.20).sum()} | "
          f"dims D>0.50: {(score_D > 0.50).sum()}")

    print("6/6  Cross-validating the reference pipeline...")
    df_ref = cv_evaluate(score_D, term_emb, patients, umbral_criterio="f1")

    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = PATHS.tables_dir / "table_2_reference_pipeline.csv"
    df_ref.to_csv(out_csv)

    print("\n== Reference pipeline (5-fold CV) ==")
    cols = ["best_tau", "n_dims", "AUC", "AUPR",
            "Precision", "Recall", "F1", "Specificity"]
    print(df_ref[cols].round(4).to_string())
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
