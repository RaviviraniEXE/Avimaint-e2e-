"""Domain FastText utilities for the recurrent baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_PATH = Path("outputs/embeddings/domain_ft.model")


def train_fasttext(sentences: list[list[str]], path: str | Path = DEFAULT_PATH, dim: int = 100):
    from gensim.models import FastText

    model = FastText(
        sentences=sentences, vector_size=dim, window=5, min_count=1,
        workers=1, sg=1, seed=42, epochs=15,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target))
    return model


def load_matrix(vocab: dict[str, int], dim: int = 100, path: str | Path = DEFAULT_PATH):
    target = Path(path)
    if not target.exists():
        return None
    from gensim.models import FastText

    model = FastText.load(str(target))
    matrix = np.random.default_rng(42).normal(0.0, 0.05, (len(vocab), dim)).astype("float32")
    if "<PAD>" in vocab:
        matrix[vocab["<PAD>"]] = 0.0
    for token, index in vocab.items():
        if token in model.wv:
            vector = model.wv[token]
            matrix[index, : min(dim, len(vector))] = vector[:dim]
    return matrix

