
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import CONFIG


@lru_cache(maxsize=2)
def load_model(model_name: str = CONFIG.model_name, device: str | None = None):

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def embed_batch(
    texts: list[str],
    *,
    model_name: str = CONFIG.model_name,
    max_length: int = CONFIG.max_length,
    device: str | None = None,
) -> np.ndarray:

    import torch

    tokenizer, model, device = load_model(model_name, device)
    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu().numpy().astype("float32")
