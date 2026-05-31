"""Pre-compute per-patient token [CLS] embeddings from raw clinical text"""


from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asthma_e2.config import CONFIG, PATHS  # noqa: E402
from asthma_e2.embeddings import embed_batch  # noqa: E402

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}")
ASTHMA_DIRS = {"asthma", "asma"}


def read_patient_words(patient_dir: Path) -> list[str]:
    """Return the unique lowercase word tokens of a patient's notes."""
    text = []
    for txt in patient_dir.glob("*.txt"):
        try:
            text.append(txt.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    joined = " ".join(text).lower()
    # Preserve order while removing duplicates.
    seen: dict[str, None] = {}
    for w in WORD_RE.findall(joined):
        seen.setdefault(w, None)
    return list(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts-dir", type=Path, required=True,
                        help="Root with <label>/<NHC>/*.txt clinical notes")
    parser.add_argument("--out", type=Path, default=PATHS.patient_embeddings)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    args = parser.parse_args()

    out: dict[str, dict] = {}
    label_dirs = [d for d in args.texts_dir.iterdir() if d.is_dir()]
    for label_dir in label_dirs:
        label = 1 if label_dir.name.lower() in ASTHMA_DIRS else 0
        patients = [d for d in label_dir.iterdir() if d.is_dir()]
        print(f"[{label_dir.name}] {len(patients)} patients (label={label})")
        for patient_dir in patients:
            words = read_patient_words(patient_dir)
            if not words:
                continue
            embeddings = []
            for start in range(0, len(words), args.batch_size):
                embeddings.append(embed_batch(words[start : start + args.batch_size]))
            emb = np.vstack(embeddings).astype("float32")
            out[patient_dir.name] = {
                "words": words,
                "embeddings_raw": emb,
                "label": label,
            }
            print(f"  {patient_dir.name}: {emb.shape[0]} tokens", end="\r")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {len(out)} patients -> {args.out}")


if __name__ == "__main__":
    main()
