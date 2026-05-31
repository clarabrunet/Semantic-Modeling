# `data/` — input artefacts (not distributed)

The clinical data used in the thesis are **confidential** and are **not** part
of this repository. This folder is intentionally empty (everything except this
file is git-ignored). To run the pipeline, place the following artefacts here
(or point the `ASTHMA_E2_DATA` environment variable at another folder).

| File | Produced by | Description |
|------|-------------|-------------|
| `dic_diagnostic.csv` | provided | ICD-10 catalogue. Latin-1, quoted line-by-line. Must contain `code`, `diag_descr`, `catalog`. Only `catalog == 12` rows are used. |
| `emb_icd10_raw.pkl` | `scripts/extract_icd10_embeddings.py` | Pickle `{"codes": list[str], "emb_matrix": np.ndarray (n_codes, 768)}` — `[CLS]` embeddings of every ICD-10 description. |
| `word_embeddings_cls.pkl` | `scripts/extract_patient_embeddings.py` | Pickle `{nhc: {"words": list[str], "embeddings_raw": np.ndarray (n_tokens, 768), "label": int}}` — per-patient token embeddings. |
| `nhc_mapping.csv` | provided | Maps clinical history number to dataset folder id. Must contain `NHC`, `id`, `DODx` (diagnosis date). Duplicate NHCs are resolved by earliest `DODx`. |

## Expected formats

### `emb_icd10_raw.pkl`
```python
{
    "codes":      ["J45", "J45.0", "J44", ...],          # len = n_codes
    "emb_matrix": np.ndarray,  # shape (n_codes, 768), float32, L2-normalised
}
```

### `word_embeddings_cls.pkl`
```python
{
    "1234567": {                      # key = NHC (string or int)
        "words":          ["tos", "disnea", ...],         # n_tokens
        "embeddings_raw": np.ndarray,  # (n_tokens, 768), float32
        "label":          1,           # 1 = asthma, 0 = non-asthma
    },
    ...
}
```

## Regenerating the embeddings

With the raw clinical text and the ICD-10 catalogue available:

```bash
python scripts/extract_icd10_embeddings.py
python scripts/extract_patient_embeddings.py --texts-dir /path/to/clinical_texts
```

Both scripts use the Spanish clinical encoder
`PlanTL-GOB-ES/bsc-bio-ehr-es` and write their output into this folder.
