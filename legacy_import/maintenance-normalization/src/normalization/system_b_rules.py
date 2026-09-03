"""System B — Rule-based normalizer built on Amin's expert resources.

Pipeline (per combined PROBLEM+ACTION record):
    1. multiword misspelling fixes   (AIR BOX -> AIRBOX)    [phrase-level]
    2. tokenize (alignment preserved)
    3. per token:
         symbols   #->number, &->and, "->inches, *->degrees, @->at ...
         misspell  single-token corrections (ENGIEN->engine)
         keep      Amin's unexpanded list (CHT, EGT, PSI, RPM, FOD)
         abbrev    Amin's expansions (CYL->cylinder, R/H->right-hand)   [ambiguous skipped]
         numbers   kept as digits (or spelled out in numbers:words mode)
    4. lowercase + tidy; drop parentheses (Amin removes brackets)

Deliberately conservative: ambiguous abbreviations and risky misspellings are
skipped (see the dictionary build report). It cannot reconstruct truncated
words — that ceiling is exactly what System C/D are meant to close.
"""
from __future__ import annotations

import re
from typing import Optional

from src.data.dictionary import Lexicon, build_lexicon
from src.data.load import load_config
from src.normalization.base import NormalizationResult, Normalizer
from src.utils.numbers import spell
from src.utils.text import Token, render, tokenize

_NUMREF_RE = re.compile(r"\#\s*(\d+)([A-Za-z]?)")


class RuleBasedNormalizer(Normalizer):
    name = "B_rules"

    def __init__(self, lex: Lexicon, cfg: dict):
        self.lex = lex
        n = cfg.get("normalizer", {})
        self.fix_misspell = n.get("fix_misspellings", True)
        self.expand_abbrev = n.get("expand_abbreviations", True)
        self.expand_symbols = n.get("expand_symbols", True)
        self.keep_unexpanded = n.get("keep_unexpanded", True)
        self.lowercase = n.get("lowercase", True)
        self.emit_alignment = n.get("emit_alignment", True)
        self.numbers = cfg.get("numbers", "digits")
        # precompile multiword misspelling patterns (longest first)
        self._multi = [(re.compile(rf"\b{re.escape(src)}\b", re.I), cor)
                       for src, cor in self.lex.misspell_multi]

    @classmethod
    def from_config(cls, path: str = "config/config.yaml") -> "RuleBasedNormalizer":
        cfg = load_config(path)
        lex, _ = build_lexicon(cfg)
        return cls(lex, cfg)

    def _num_out(self, digits: str) -> str:
        return spell(digits) if self.numbers == "words" else digits

    def _token(self, tok: Token, stats: dict) -> None:
        low = tok.text.lower()

        if tok.kind == "ws":
            tok.set_norm([tok.text], "ws")
            return

        if tok.kind == "symbol":
            if self.expand_symbols and tok.text in self.lex.symbols:
                tok.set_norm(self.lex.symbols[tok.text], "symbol")
                stats["symbol"] = stats.get("symbol", 0) + 1
            elif tok.text in "()":
                tok.set_norm([], "drop_bracket")          # Amin removes brackets
            else:
                tok.set_norm([tok.text], "keep")
            return

        if tok.kind == "numref":
            m = _NUMREF_RE.match(tok.text.replace(" ", ""))
            if m and self.expand_symbols:
                num = self._num_out(m.group(1)) + (m.group(2) if m.group(2) else "")
                tok.set_norm([self.lex.symbols.get("#", "number"), num], "numref")
                stats["symbol"] = stats.get("symbol", 0) + 1
                return
            tok.set_norm([tok.text], "keep")
            return

        if tok.kind == "num":
            tok.set_norm([self._num_out(tok.text)], "num")
            return

        if tok.kind in ("word", "slash"):
            # 1) single-token misspelling correction
            surface = low
            if self.fix_misspell and surface in self.lex.misspellings:
                surface = self.lex.misspellings[surface]
                stats["misspell"] = stats.get("misspell", 0) + 1
            # 2) protected keep-as-is
            if self.keep_unexpanded and surface in self.lex.keep:
                tok.set_norm([surface], "keep")
                return
            # 3) abbreviation expansion (single-expansion only; ambiguous skipped)
            if self.expand_abbrev and surface in self.lex.abbrev:
                tok.set_norm(self.lex.abbrev[surface], "abbrev")
                stats["abbrev"] = stats.get("abbrev", 0) + 1
                return
            tok.set_norm([surface if self.lowercase else tok.text], "keep")
            return

        # punctuation
        if tok.text in "()":
            tok.set_norm([], "drop_bracket")
        else:
            tok.set_norm([tok.text], "keep")

    def normalize(self, text: str) -> NormalizationResult:
        raw = text or ""
        pre = raw
        for pat, cor in self._multi:                       # phrase-level misspellings
            pre = pat.sub(cor, pre)
        tokens = tokenize(pre)
        stats: dict = {}
        for tok in tokens:
            self._token(tok, stats)
        normalized = render(tokens)
        if self.lowercase:
            normalized = normalized.lower()

        alignment = []
        if self.emit_alignment:
            for tok in tokens:
                if tok.kind == "ws":
                    continue
                alignment.append((tok.text, (tok.start, tok.end), tok.normalized, tok.rule))

        stats["n_tokens"] = sum(1 for t in tokens if t.kind != "ws")
        stats["n_expansions"] = sum(stats.get(k, 0) for k in ("symbol", "abbrev", "misspell"))
        return NormalizationResult(raw=raw, normalized=normalized, alignment=alignment, stats=stats)

