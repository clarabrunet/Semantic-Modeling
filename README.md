# Semantic-Modeling

## Overview

**Semantic-Modeling** detects asthma in clinical narratives by reasoning in the
_semantic space_ of bio-bsc-ehr-es model.
It combines a biomedical knowledge graph (ICD-10), **graph
embedding propagation**, and a discriminative-subspace criterion to turn a
handful of reference terms into a robust, interpretable patient-level score.

This repository contains the complete code for **Strategy 2 ** and a
notebook that reproduces the results and ablation studies of
the thesis.

```
reference terms ─┐
                 ├─► C × S criterion ─► discriminative subspace ─┐
ICD-10 graph ─► propagation ─────────┘                          ├─► patient
                                                                 │   score ─► τ
patient tokens ─► top-N prefiltration ─► subspace projection ───┘
```

## Repository layout

```
Semantic-Modeling/
├── src/asthma_e2/            # the importable package
│   ├── config.py             # hyperparameters, paths, reference terms
│   ├── embeddings.py         # transformer [CLS] embeddings
│   ├── data_loading.py       # ICD-10 catalogue, embeddings, patient data
│   ├── graph.py              # build_graph (hierarchical + semantic edges)
│   ├── propagation.py        # propagate_embeddings
│   ├── criterion.py          # compute_CS (C × S discriminative score)
│   ├── scoring.py            # score_patient
│   └── evaluation.py         # cv_evaluate (stratified k-fold CV)
├── scripts/
│   ├── extract_icd10_embeddings.py     # offline: embed ICD-10 descriptions
│   ├── extract_patient_embeddings.py   # offline: embed patient tokens
│   └── run_pipeline.py                 # end-to-end reference pipeline
├── notebooks/
│   └── results.ipynb
├── tests/                    # unit tests on synthetic data (no clinical data)
├── data/                     # input artefacts — empty, see data/README.md
└── results/                  # generated figures/ and tables/
```

## Installation

```bash
python -m venv .venv

pip install -e ".[dev]"
```

## Data

The clinical data used in the thesis are **confidential** and are **not**
included in this repository


## License

[MIT](LICENSE) — **source code only**. The clinical data are confidential and
are not covered by this license.
