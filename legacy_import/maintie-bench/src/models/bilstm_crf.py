"""BiLSTM-CRF NER baseline with optional FastText and character features.

The original repository referenced this module but did not commit it.  This
implementation keeps the public API used by scripts/05_train_eval.py and uses
only the frozen train/dev split for fitting and early stopping.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random

import numpy as np


PAD, UNK = "<PAD>", "<UNK>"


def build_char_vocab(records: list[dict]) -> dict[str, int]:
    chars = sorted({c for row in records for token in row["tokens"] for c in token})
    return {token: index for index, token in enumerate([PAD, UNK, *chars])}


class _Network:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn
        from torchcrf import CRF

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                vocab_size, tag_count, word_dim, hidden_dim, dropout, char_vocab_size, char_dim, pretrained = args
                self.word = nn.Embedding(vocab_size, word_dim, padding_idx=0)
                if pretrained is not None:
                    matrix = __import__("torch").as_tensor(pretrained, dtype=__import__("torch").float32)
                    if matrix.shape == self.word.weight.shape:
                        self.word.weight.data.copy_(matrix)
                self.use_char = char_vocab_size > 0
                if self.use_char:
                    self.char = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
                    self.char_conv = nn.Conv1d(char_dim, char_dim, kernel_size=3, padding=1)
                in_dim = word_dim + (char_dim if self.use_char else 0)
                self.lstm = nn.LSTM(in_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(dropout)
                self.out = nn.Linear(hidden_dim, tag_count)
                self.crf = CRF(tag_count, batch_first=True)

            def emissions(self, words, chars=None):
                pieces = [self.word(words)]
                if self.use_char:
                    # chars: [batch, words, chars]
                    bsz, nwords, nchars = chars.shape
                    encoded = self.char(chars.reshape(bsz * nwords, nchars)).transpose(1, 2)
                    encoded = self.char_conv(encoded).relu().amax(dim=2).reshape(bsz, nwords, -1)
                    pieces.append(encoded)
                packed, _ = self.lstm(self.dropout(__import__("torch").cat(pieces, dim=-1)))
                return self.out(self.dropout(packed))

            def loss(self, words, tags, mask, chars=None):
                return -self.crf(self.emissions(words, chars), tags, mask=mask, reduction="mean")

            def decode(self, words, mask, chars=None):
                return self.crf.decode(self.emissions(words, chars), mask=mask)

        return Network()


class BiLSTMCRF:
    def __init__(self, vocab: dict[str, int], tags: list[str], pretrained=None,
                 char_vocab: dict[str, int] | None = None, seed: int = 42,
                 word_dim: int = 100, hidden_dim: int = 192, char_dim: int = 32,
                 dropout: float = 0.4, lr: float = 1e-3):
        import torch

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.vocab, self.tags = vocab, tags
        self.tag_to_id = {tag: i for i, tag in enumerate(tags)}
        self.char_vocab = char_vocab or {}
        self.params = dict(word_dim=word_dim, hidden_dim=hidden_dim, char_dim=char_dim,
                           dropout=dropout, lr=lr, seed=seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _Network(len(vocab), len(tags), word_dim, hidden_dim, dropout,
                            len(self.char_vocab), char_dim, pretrained).to(self.device)

    @staticmethod
    def build_vocab(records: list[dict], min_count: int = 1) -> dict[str, int]:
        counts = Counter(token.lower() for row in records for token in row["tokens"])
        words = sorted(word for word, count in counts.items() if count >= min_count)
        return {word: i for i, word in enumerate([PAD, UNK, *words])}

    def _tensorize(self, rows: list[dict], labelled: bool = True):
        import torch

        max_words = max(len(row["tokens"]) for row in rows)
        max_chars = max(1, max(len(token) for row in rows for token in row["tokens"]))
        words = torch.zeros((len(rows), max_words), dtype=torch.long)
        mask = torch.zeros((len(rows), max_words), dtype=torch.bool)
        tags = torch.zeros((len(rows), max_words), dtype=torch.long)
        chars = torch.zeros((len(rows), max_words, max_chars), dtype=torch.long)
        for i, row in enumerate(rows):
            for j, token in enumerate(row["tokens"]):
                words[i, j] = self.vocab.get(token.lower(), self.vocab[UNK]); mask[i, j] = True
                if labelled:
                    tags[i, j] = self.tag_to_id.get(row["bio"][j], self.tag_to_id["O"])
                if self.char_vocab:
                    for k, char in enumerate(token[:max_chars]):
                        chars[i, j, k] = self.char_vocab.get(char, self.char_vocab[UNK])
        return words.to(self.device), tags.to(self.device), mask.to(self.device), chars.to(self.device)

    def fit(self, records: list[dict], dev: list[dict] | None = None,
            epochs: int = 60, patience: int = 8, batch_size: int = 16):
        import torch
        from tqdm.auto import tqdm

        if not records:
            raise ValueError("BiLSTM-CRF training requires records")
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"], weight_decay=1e-5)
        best_loss, best_state, stale = float("inf"), None, 0
        rng = random.Random(self.params["seed"])
        total_batches = (len(records) + batch_size - 1) // batch_size
        print(f"[Tier2 NER] device={self.device} | train_docs={len(records)} | batches/epoch={total_batches}")
        for epoch in range(1, epochs + 1):
            order = list(range(len(records))); rng.shuffle(order); self.net.train()
            running = 0.0
            bar = tqdm(range(0, len(order), batch_size), total=total_batches,
                       desc=f"Tier2 NER epoch {epoch:02d}/{epochs}", unit="batch",
                       dynamic_ncols=True, leave=True)
            for step, start in enumerate(bar, 1):
                batch = [records[i] for i in order[start:start + batch_size]]
                words, tags, mask, chars = self._tensorize(batch)
                optimizer.zero_grad(); loss = self.net.loss(words, tags, mask, chars)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0); optimizer.step()
                running += float(loss.item())
                bar.set_postfix(train_loss=f"{running/step:.4f}")
            check = dev or records
            self.net.eval()
            with torch.no_grad():
                w, t, m, c = self._tensorize(check)
                dev_loss = float(self.net.loss(w, t, m, c).item())
            improved = dev_loss < best_loss - 1e-5
            if improved:
                best_loss, best_state, stale = dev_loss, deepcopy(self.net.state_dict()), 0
            else:
                stale += 1
            print(f"[Tier2 NER] epoch={epoch:02d} dev_loss={dev_loss:.6f} best={best_loss:.6f} patience={stale}/{patience}")
            if stale >= patience:
                print(f"[Tier2 NER] early stopping at epoch {epoch}; restoring best checkpoint")
                break
        if best_state is not None: self.net.load_state_dict(best_state)
        return self

    def predict(self, records: list[dict]) -> list[list[str]]:
        import torch

        if not records: return []
        self.net.eval()
        print(f"[Tier2 NER] predicting {len(records)} documents ...")
        with torch.no_grad():
            words, _, mask, chars = self._tensorize(records, labelled=False)
            decoded = self.net.decode(words, mask, chars)
        print(f"[Tier2 NER] prediction complete: {len(decoded)}/{len(records)}")
        return [[self.tags[index] for index in row] for row in decoded]

    def save(self, path: str):
        import torch

        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "vocab": self.vocab, "tags": self.tags,
                    "char_vocab": self.char_vocab, "params": self.params}, target)

    @classmethod
    def tuned(cls, vocab, tags, train, dev, grid=None, pretrained=None, char_vocab=None):
        grid = grid or {"lr": [1e-3], "dropout": [0.4]}
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities

        best_model, best = None, {"micro_f1": -1.0}
        gold = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], r["bio"]), "relations": []} for r in dev]
        for lr in grid.get("lr", [1e-3]):
            for dropout in grid.get("dropout", [0.4]):
                model = cls(vocab, tags, pretrained=pretrained, char_vocab=char_vocab,
                            lr=float(lr), dropout=float(dropout)).fit(train, dev=dev)
                pred = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], p), "relations": []}
                        for r, p in zip(dev, model.predict(dev))]
                score = entity_scores(gold, pred)["micro_f1"]
                if score > best["micro_f1"]:
                    best_model, best = model, {"lr": float(lr), "dropout": float(dropout), "micro_f1": score}
        return best_model, best
