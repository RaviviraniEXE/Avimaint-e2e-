"""Tokenization and low-level text helpers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    text: str
    start: int
    end: int
    kind: str
    norm: Optional[List[str]] = None
    rule: Optional[str] = None

    def set_norm(self, value, rule: str) -> None:
        if isinstance(value, str):
            value = value.split()
        self.norm = list(value)
        self.rule = rule

    @property
    def normalized(self) -> List[str]:
        return self.norm if self.norm is not None else [self.text]


# Order matters: slash abbreviations and #-number refs are captured as units.
_TOKEN_RE = re.compile(
    r"""
    (?P<wslash>[wW]/[oO]?(?![A-Za-z]))                  |  # w/  w/o
    (?P<slash>[A-Za-z]{1,4}/[A-Za-z]{1,4})              |  # R/H L/H A/C I/B C/W T/O
    (?P<numref>\#\s?\d+[A-Za-z]?)                        |  # #2  #4A
    (?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)                  |  # engine  CK'ED
    (?P<num>\d+(?:[.,/]\d+)?)                            |  # 1200  30.5  1/8
    (?P<sym>[&@+*=>"'#])                                 |  # symbols Amin expands
    (?P<ws>\s+)                                          |
    (?P<punct>[^\sA-Za-z0-9])
    """,
    re.VERBOSE,
)

_KIND = {"wslash": "slash", "slash": "slash", "numref": "numref", "word": "word",
         "num": "num", "sym": "symbol", "ws": "ws", "punct": "punct"}


def clean_unicode(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for a, b in {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                 " ": " ", "﻿": "", "°": "*"}.items():
        text = text.replace(a, b)
    return text


def tokenize(text: str) -> List[Token]:
    text = clean_unicode(text)
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(Token(m.group(), m.start(), m.end(), _KIND.get(m.lastgroup, "punct")))
    return tokens


def render(tokens: List[Token]) -> str:
    parts = []
    for tok in tokens:
        if tok.kind == "ws":
            continue
        for sub in tok.normalized:
            if sub:
                parts.append(sub)
    out = " ".join(parts)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def simple_tokens(text: str) -> List[str]:
    text = clean_unicode(text).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)

