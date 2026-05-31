"""Pre-compute [CLS] embeddings for every ICD-10 code description"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asthma_e2.config import CONFIG, PATHS  # noqa: E402
from asthma_e2.data_loading import load_icd10_catalogue  # noqa: E402
from asthma_e2.embeddings import embed_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=PATHS.icd10_catalogue)
    parser.add_argument("--out", type=Path, default=PATHS.icd10_embeddings)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    args = parser.parse_args()

    print(f"Loading ICD-10 catalogue: {args.catalogue}")
    df = load_icd10_catalogue(args.catalogue)
    codes = df["code"].tolist()
    descriptions = df["diag_descr"].astype(str).tolist()
    print(f"  {len(codes):,} codes to embed")

    embeddings = []
    for start in range(0, len(descriptions), args.batch_size):
        batch = descriptions[start : start + args.batch_size]
        embeddings.append(embed_batch(batch))
        print(f"  embedded {min(start + args.batch_size, len(descriptions)):,}/"
              f"{len(descriptions):,}", end="\r")
    emb_matrix = np.vstack(embeddings).astype("float32")
    print(f"\n  embedding matrix: {emb_matrix.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"codes": codes, "emb_matrix": emb_matrix}, f)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
