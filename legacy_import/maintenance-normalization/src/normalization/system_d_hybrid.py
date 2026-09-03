"""System D — Hybrid: rules (B) first, then ByT5 (C) to repair the long tail.

Alignment comes from the rule stage (ByT5 rewrites are not aligned). Degrades
gracefully to System B output if ByT5 is unavailable.
"""
from __future__ import annotations

from typing import Optional

from src.normalization.base import NormalizationResult, Normalizer
from src.normalization.system_b_rules import RuleBasedNormalizer
from src.normalization.system_c_byt5 import ByT5Normalizer


class HybridNormalizer(Normalizer):
    name = "D_hybrid"

    def __init__(self, rules: RuleBasedNormalizer, byt5: Optional[ByT5Normalizer] = None):
        self.rules = rules
        self.byt5 = byt5

    def normalize(self, text: str) -> NormalizationResult:
        r = self.rules.normalize(text)
        if self.byt5 is None:
            r.stats = {**r.stats, "hybrid_stage": 1}
            return r
        try:
            b = self.byt5.normalize(r.normalized)
        except ImportError:
            r.stats = {**r.stats, "hybrid_stage": 1}
            return r
        return NormalizationResult(raw=text or "", normalized=b.normalized,
                                   alignment=r.alignment, stats={**r.stats, "hybrid_stage": 2})

