"""BiLSTM-CRF NER baseline with optional FastText and character features.

Uses only the frozen train/dev split for fitting and early stopping.  Training
progress is observable in real time but the optimisation protocol is unchanged:
early stopping is based on DEV loss and the best DEV-loss checkpoint is restored.
DEV entity F1 is shown as a diagnostic each epoch.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import random

import numpy as np

from src.progress import EpochProgress, trace_event

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
        self.training_summary: dict = {}
        self.training_history: list[dict] = []

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

    def _dev_entity_f1(self, dev: list[dict] | None) -> float | None:
        if not dev:
            return None
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities
        pred_tags = self.predict(dev)
        gold = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], r["bio"]), "relations": []}
                for r in dev]
        pred = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], p), "relations": []}
                for r, p in zip(dev, pred_tags)]
        return float(entity_scores(gold, pred)["micro_f1"])

    def fit(self, records: list[dict], dev: list[dict] | None = None,
            epochs: int = 60, patience: int = 8, batch_size: int = 16):
        import torch

        if not records:
            raise ValueError("BiLSTM-CRF training requires records")
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"], weight_decay=1e-5)
        best_loss, best_state, stale = float("inf"), None, 0
        best_epoch, stopped_epoch = 0, 0
        rng = random.Random(self.params["seed"])
        monitor = EpochProgress("BiLSTM-CRF NER", epochs=epochs, patience=patience)
        trace_event("model_fit_context", model="BiLSTM-CRF NER", device=str(self.device),
                    records=len(records), dev_records=len(dev or []), batch_size=batch_size,
                    lr=self.params["lr"], dropout=self.params["dropout"],
                    early_stopping_metric="dev_loss", patience=patience)

        for epoch in range(1, epochs + 1):
            order = list(range(len(records))); rng.shuffle(order); self.net.train()
            total_batches = (len(order) + batch_size - 1) // batch_size
            monitor.start_epoch(epoch, total_batches)
            running_loss = 0.0; batches = 0
            for start in range(0, len(order), batch_size):
                batch = [records[i] for i in order[start:start + batch_size]]
                words, tags, mask, chars = self._tensorize(batch)
                optimizer.zero_grad(); loss = self.net.loss(words, tags, mask, chars)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0); optimizer.step()
                batches += 1; running_loss += float(loss.item())
                monitor.batch(epoch, batches, total_batches, running_loss / batches)

            check = dev or records
            self.net.eval()
            with torch.no_grad():
                w, t, m, c = self._tensorize(check)
                dev_loss = float(self.net.loss(w, t, m, c).item())
            dev_f1 = self._dev_entity_f1(dev)
            train_loss = running_loss / max(1, batches)
            improved = dev_loss < best_loss - 1e-5
            if improved:
                best_loss, best_state, stale, best_epoch = dev_loss, deepcopy(self.net.state_dict()), 0, epoch
            else:
                stale += 1
            self.training_history.append({
                "epoch": epoch, "train_loss": round(train_loss, 8), "dev_loss": round(dev_loss, 8),
                "dev_f1": None if dev_f1 is None else round(dev_f1, 8), "stale": stale,
                "improved": bool(improved),
            })
            monitor.finish_epoch(epoch, train_loss, dev_loss, dev_f1, best_loss, stale, improved, best_epoch)
            stopped_epoch = epoch
            if stale >= patience:
                monitor.early_stop(epoch, best_epoch, best_loss)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.training_summary = {
            "model": "BiLSTM-CRF NER", "device": str(self.device), "max_epochs": int(epochs),
            "stopped_epoch": int(stopped_epoch), "best_epoch": int(best_epoch),
            "best_dev_loss": round(float(best_loss), 8), "patience": int(patience),
            "early_stopped": bool(stopped_epoch < epochs), "early_stopping_metric": "dev_loss",
            "batch_size": int(batch_size), "params": dict(self.params), "history": list(self.training_history),
        }
        monitor.close(stopped_epoch, best_epoch, best_loss)
        return self

    def predict(self, records: list[dict]) -> list[list[str]]:
        import torch

        if not records: return []
        self.net.eval()
        with torch.no_grad():
            words, _, mask, chars = self._tensorize(records, labelled=False)
            decoded = self.net.decode(words, mask, chars)
        return [[self.tags[index] for index in row] for row in decoded]

    def save(self, path: str):
        import torch

        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "vocab": self.vocab, "tags": self.tags,
                    "char_vocab": self.char_vocab, "params": self.params,
                    "training_summary": self.training_summary}, target)

    @classmethod
    def tuned(cls, vocab, tags, train, dev, grid=None, pretrained=None, char_vocab=None):
        grid = grid or {"lr": [1e-3], "dropout": [0.4]}
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities

        configs = [(float(lr), float(dropout)) for lr in grid.get("lr", [1e-3])
                   for dropout in grid.get("dropout", [0.4])]
        best_model, best = None, {"micro_f1": -1.0, "history": []}
        gold = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], r["bio"]), "relations": []}
                for r in dev]
        for idx, (lr, dropout) in enumerate(configs, 1):
            print(f"\n[BiLSTM tuning] config {idx}/{len(configs)} | lr={lr:g} dropout={dropout:g}", flush=True)
            model = cls(vocab, tags, pretrained=pretrained, char_vocab=char_vocab,
                        lr=lr, dropout=dropout).fit(train, dev=dev)
            pred = [{"tokens": r["tokens"], "entities": bio_to_entities(r["tokens"], p), "relations": []}
                    for r, p in zip(dev, model.predict(dev))]
            score = float(entity_scores(gold, pred)["micro_f1"])
            item = {"lr": lr, "dropout": dropout, "micro_f1": score,
                    "training_summary": model.training_summary}
            best["history"].append(item)
            is_best = score > best["micro_f1"]
            print(f"[BiLSTM tuning] DEV entity micro-F1={score:.4f}{' | BEST' if is_best else ''}", flush=True)
            if is_best:
                if best_model is not None:
                    try: best_model.net.to("cpu")
                    except Exception: pass
                best_model = model
                best.update({"lr": lr, "dropout": dropout, "micro_f1": score,
                             "training_summary": model.training_summary})
            else:
                try: model.net.to("cpu")
                except Exception: pass
                del model
                if __import__("torch").cuda.is_available(): __import__("torch").cuda.empty_cache()
        return best_model, best
