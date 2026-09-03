"""System C — ByT5 character-level transformer normalizer (inference wrapper).

Heavy imports (torch/transformers) are deferred so the rule pipeline stays
lightweight. Train with scripts/05_train_byt5.py.
"""
from __future__ import annotations

import re
from typing import Optional

from src.normalization.base import NormalizationResult, Normalizer


class ByT5Normalizer(Normalizer):
    name = "C_byt5"

    def __init__(self, model_dir: str, cfg_byt5: Optional[dict] = None, device: Optional[str] = None):
        self.model_dir = model_dir
        self.cfg = cfg_byt5 or {}
        self._device = device
        self._model = None
        self._tokenizer = None

    def _lazy(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as e:  # pragma: no cover
            raise ImportError("System C needs torch + transformers; train with 05_train_byt5.py") from e
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def normalize(self, text: str) -> NormalizationResult:
        raw = text or ""
        self._lazy()
        enc = self._tokenizer(raw, return_tensors="pt", truncation=True,
                              max_length=self.cfg.get("max_source_length", 160)).to(self._device)
        with self._torch.no_grad():
            out = self._model.generate(**enc, max_length=self.cfg.get("max_target_length", 160), num_beams=4)
        norm = self._tokenizer.decode(out[0], skip_special_tokens=True)
        norm = re.sub(r"\s+", " ", norm).strip().lower()
        return NormalizationResult(raw=raw, normalized=norm, alignment=[], stats={"model": 1})

