"""Matched System-D preprocessing and conservative deployment guards.

The optional operational branch must use the same representation family as
the paired SpERT checkpoint: expert rules first, followed by the locked ByT5
checkpoint. The frozen TRUE-RAW RQ4/RQ5 branch never imports this module.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_SYMBOL_KEYS = {
    '(")': '"', '(#)': '#', '(&)': '&', "(')": "'", '(*)': '*',
    '(@)': '@', '(+)': '+', '(=)': '=', '(>)': '>', '(1/2)': '1/2',
}
_MISSPELL_BLOCKLIST = {"off", "that", "affect", "sam", "same", "no", "on", "in", "by"}
_TOKEN_RE = re.compile(
    r"(?P<wslash>[wW]/[oO]?(?![A-Za-z]))|"
    r"(?P<slash>[A-Za-z]{1,4}/[A-Za-z]{1,4})|"
    r"(?P<numref>\#\s?\d+[A-Za-z]?)|"
    r"(?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)|"
    r"(?P<num>\d+(?:[.,/]\d+)?)|"
    r"(?P<symbol>[&@+*=>\"'#])|"
    r"(?P<ws>\s+)|(?P<punct>[^\sA-Za-z0-9])"
)
_NUMERIC = re.compile(r"(?<![A-Za-z])#?(\d+(?:[.,/]\d+)?)(?:[A-Za-z])?(?![A-Za-z])")
_PART = re.compile(
    r"\b(?:P/?N|S/?N|PART\s+NO\.?|SERIAL\s+NO\.?)\s*[:#-]?\s*([A-Z0-9._/-]+)", re.I
)
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million",
}
_ANCHORS = {
    "aircraft", "alternator", "assembly", "battery", "carb", "carburetor",
    "compressor", "compression", "cylinder", "engine", "exhaust", "fault",
    "gasket", "generator", "governor", "intake", "injector", "landing", "leak",
    "leaking", "magneto", "manifold", "oil", "piston", "pressure", "propeller",
    "rpm", "starter", "tachometer", "temperature", "throttle", "valve", "vibration",
}
_DIRECTIONS = {
    "left": re.compile(r"\b(?:l/?h|left(?:-hand|\s+hand)?)\b", re.I),
    "right": re.compile(r"\b(?:r/?h|right(?:-hand|\s+hand)?)\b", re.I),
    "inboard": re.compile(r"\b(?:i/?b|inboard)\b", re.I),
    "outboard": re.compile(r"\b(?:o/?b|outboard)\b", re.I),
}


def clean_unicode(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    for before, after in {
        "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "\u00a0": " ", "\ufeff": "", "°": "*",
    }.items():
        value = value.replace(before, after)
    return value


@dataclass(frozen=True)
class RuleResult:
    raw: str
    normalized: str
    expansions: int


class ExpertRuleNormalizer:
    """Dependency-free port of the authoritative System-B rule stage."""

    def __init__(self, resource_dir: Path):
        self.resource_dir = Path(resource_dir).resolve()
        self.keep = self._load_keep()
        self.abbrev, self.symbols = self._load_abbreviations()
        self.misspell, self.multi = self._load_misspellings()

    @property
    def files(self) -> dict[str, Path]:
        return {
            "abbreviations": self.resource_dir / "abbreviations.csv",
            "misspellings": self.resource_dir / "misspellings.csv",
            "unexpanded": self.resource_dir / "unexpanded.csv",
        }

    def _rows(self, name: str):
        path = self.resource_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required normalization resource missing: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)

    def _load_keep(self):
        return {
            str(row.get("keep", "")).strip().lower()
            for row in self._rows("unexpanded.csv")
            if str(row.get("keep", "")).strip()
        }

    def _load_abbreviations(self):
        grouped: dict[str, set[str]] = {}
        symbols: dict[str, str] = {}
        for row in self._rows("abbreviations.csv"):
            key = str(row.get("abbrev", "")).strip()
            expansion = str(row.get("expansion", "")).strip().lower()
            if not key or not expansion:
                continue
            if key in _SYMBOL_KEYS:
                symbols[_SYMBOL_KEYS[key]] = expansion
            else:
                grouped.setdefault(key.lower(), set()).add(expansion)
        return ({key: next(iter(values)) for key, values in grouped.items()
                 if len(values) == 1 and key not in self.keep}, symbols)

    def _load_misspellings(self):
        single: dict[str, str] = {}
        multi: list[tuple[str, str]] = []
        for row in self._rows("misspellings.csv"):
            source = str(row.get("misspelling", "")).strip()
            correction = re.sub(r"\s*\(.*\)\s*", "", str(row.get("correction", ""))).strip()
            low = source.lower()
            if not source or not correction or len(low) < 3 or low in _MISSPELL_BLOCKLIST:
                continue
            if " " in source:
                multi.append((low, correction.lower()))
            else:
                single[low] = correction.lower()
        multi.sort(key=lambda pair: -len(pair[0]))
        return single, multi

    def normalize(self, text: str) -> RuleResult:
        raw = clean_unicode(text)
        prepared = raw
        for source, correction in self.multi:
            prepared = re.sub(rf"\b{re.escape(source)}\b", correction, prepared, flags=re.I)

        parts: list[str] = []
        expansions = 0
        for match in _TOKEN_RE.finditer(prepared):
            value = match.group(0)
            kind = match.lastgroup
            if kind == "ws":
                continue
            if kind == "symbol":
                if value in self.symbols:
                    parts.extend(self.symbols[value].split())
                    expansions += 1
                elif value not in "()":
                    parts.append(value)
                continue
            if kind == "numref":
                compact = value.replace(" ", "")
                num_match = re.match(r"#(\d+)([A-Za-z]?)", compact)
                if num_match and "#" in self.symbols:
                    parts.extend([self.symbols["#"], num_match.group(1) + num_match.group(2)])
                    expansions += 1
                else:
                    parts.append(value.lower())
                continue
            if kind in {"word", "slash"}:
                surface = value.lower()
                if surface in self.misspell:
                    surface = self.misspell[surface]
                    expansions += 1
                if surface in self.keep:
                    parts.append(surface)
                elif surface in self.abbrev:
                    parts.extend(self.abbrev[surface].split())
                    expansions += 1
                else:
                    parts.append(surface)
                continue
            if kind == "punct" and value in "()":
                continue
            parts.append(value.lower())

        normalized = " ".join(p for p in parts if p)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip().lower()
        return RuleResult(raw=raw, normalized=normalized, expansions=expansions)


def _numeric_values(text: str) -> Counter[str]:
    return Counter(match.group(1).replace(",", ".").lower() for match in _NUMERIC.finditer(text or ""))


def _part_values(text: str) -> Counter[str]:
    return Counter(match.group(1).lower() for match in _PART.finditer(text or ""))


def _word_number_values(text: str) -> Counter[str]:
    words = re.findall(r"[a-z]+", str(text or "").lower())
    return Counter(word for word in words if word in _NUMBER_WORDS)


def _directions(text: str) -> set[str]:
    return {name for name, pattern in _DIRECTIONS.items() if pattern.search(text or "")}


def _anchors(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text or "").lower())) & _ANCHORS


def validate_hybrid_candidate(original: str, rule_text: str, candidate: str) -> tuple[bool, list[str]]:
    """Reject destructive or hallucinated ByT5 rewrites; rule output is fallback."""
    candidate = re.sub(r"\s+", " ", str(candidate or "")).strip().lower()
    problems: list[str] = []
    if not candidate:
        return False, ["ByT5 returned empty text."]
    expected_numbers = _numeric_values(original)
    got_numbers = _numeric_values(candidate)
    if got_numbers != expected_numbers:
        problems.append(f"numeric values changed (expected {dict(expected_numbers)}, got {dict(got_numbers)})")
    expected_words = _word_number_values(rule_text)
    got_words = _word_number_values(candidate)
    if got_words - expected_words:
        problems.append(f"new number words appeared: {dict(got_words - expected_words)}")
    if _part_values(candidate) != _part_values(original):
        problems.append("part/serial identifiers changed")
    if _directions(candidate) != _directions(rule_text):
        problems.append("directional/location anchors changed")
    missing_anchors = sorted(_anchors(rule_text) - _anchors(candidate))
    if missing_anchors:
        problems.append("maintenance anchors disappeared: " + ", ".join(missing_anchors))
    if len(candidate) > max(5000, len(rule_text) * 3 + 80):
        problems.append("rewrite was implausibly long")
    if len(candidate) < max(3, int(len(rule_text) * 0.35)):
        problems.append("rewrite was implausibly short")
    return not problems, problems
