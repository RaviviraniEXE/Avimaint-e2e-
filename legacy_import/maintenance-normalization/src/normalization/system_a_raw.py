"""System A — Raw baseline (control): unicode + whitespace + optional lowercase."""
from __future__ import annotations

import re

from src.normalization.base import NormalizationResult, Normalizer
from src.utils.text import clean_unicode


class RawNormalizer(Normalizer):
    name = "A_raw"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def normalize(self, text: str) -> NormalizationResult:
        raw = text or ""
        out = re.sub(r"\s+", " ", clean_unicode(raw)).strip()
        if self.lowercase:
            out = out.lower()
        return NormalizationResult(raw=raw, normalized=out, alignment=[], stats={})

