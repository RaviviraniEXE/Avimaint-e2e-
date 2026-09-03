"""Build the normalization lexicon from Amin's expert lists, with auditing.

Three resources, each cleaned defensively:

* Abbreviations (100): abbrev -> expansion. Keys with MORE THAN ONE expansion
  (COMP=compression/compressor, INSP=inspected/inspection, IN=intake/inches,
  SEC=second/section) are genuinely ambiguous out of context, so they are
  skipped and logged (precision over recall). Symbol abbreviations ((#), (&),
  ...) are split out into a separate symbol map.

* Misspellings (114): misspelling -> correction. Sources that are valid English
  words or too short (OFF->OF, THAT->THAN, E->3) are dangerous to apply globally
  and are skipped and logged. Multi-word sources ("AIR BOX"->"AIRBOX") are kept
  and applied as phrases.

* Unexpanded (26): abbreviations Amin deliberately left as-is (CHT, EGT, PSI,
  RPM, FOD ...). These are protected from expansion.

Every decision is written to outputs/reports/dictionary_build_report.txt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.data.load import (
    load_abbreviations,
    load_config,
    load_misspellings,
    load_unexpanded,
)

# map the parenthesised symbol keys Amin uses to raw characters
_SYMBOL_KEYS = {"(\")": '"', "(#)": "#", "(&)": "&", "(')": "'", "(*)": "*",
                "(@)": "@", "(+)": "+", "(=)": "=", "(>)": ">", "(1/2)": "1/2"}


@dataclass
class Lexicon:
    abbrev: Dict[str, str] = field(default_factory=dict)         # single-expansion abbreviations
    symbols: Dict[str, str] = field(default_factory=dict)        # raw char -> word
    misspellings: Dict[str, str] = field(default_factory=dict)   # lower source -> correction (lower)
    misspell_multi: List[Tuple[str, str]] = field(default_factory=list)  # (phrase, correction) longest-first
    keep: set = field(default_factory=set)                       # unexpanded / protected
    ambiguous: set = field(default_factory=set)                  # abbrev keys skipped


@dataclass
class BuildReport:
    abbrev_ok: List[Tuple[str, str]] = field(default_factory=list)
    abbrev_ambiguous: List[Tuple[str, List[str]]] = field(default_factory=list)
    symbols: List[Tuple[str, str]] = field(default_factory=list)
    misspell_ok: int = 0
    misspell_multi: int = 0
    misspell_skipped: List[Tuple[str, str]] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        L = ["# Lexicon build report (Amin expert resources)", ""]
        L.append(f"abbreviations accepted : {len(self.abbrev_ok)}")
        L.append(f"abbreviations ambiguous(skipped): {len(self.abbrev_ambiguous)}")
        L.append(f"symbols                : {len(self.symbols)}")
        L.append(f"misspellings single    : {self.misspell_ok}")
        L.append(f"misspellings multiword : {self.misspell_multi}")
        L.append(f"misspellings skipped(real word/short): {len(self.misspell_skipped)}")
        L.append(f"keep-as-is             : {len(self.keep)}")
        L.append("")
        L.append("## symbols")
        L += [f"  {k!r} -> {v}" for k, v in self.symbols]
        L.append("\n## ambiguous abbreviations (skipped)")
        L += [f"  {k} -> {v}" for k, v in self.abbrev_ambiguous]
        L.append("\n## skipped misspellings (source is a real word or too short)")
        L += [f"  {k} -> {v}" for k, v in self.misspell_skipped]
        L.append("\n## keep-as-is (unexpanded)")
        L.append("  " + ", ".join(self.keep))
        L.append("\n## accepted abbreviations")
        L += [f"  {k} -> {v}" for k, v in self.abbrev_ok]
        return "\n".join(L)


def build_lexicon(cfg: dict) -> Tuple[Lexicon, BuildReport]:
    rep = BuildReport()
    keep = set(load_unexpanded(cfg))
    rep.keep = sorted(keep)

    # ---- abbreviations ----
    ab = load_abbreviations(cfg)
    by_key: Dict[str, List[str]] = {}
    symbols: Dict[str, str] = {}
    for _, r in ab.iterrows():
        k = r["abbrev"].strip()
        if k in _SYMBOL_KEYS:
            symbols[_SYMBOL_KEYS[k]] = r["expansion"]
            continue
        by_key.setdefault(k.lower(), []).append(r["expansion"])
    rep.symbols = sorted(symbols.items())

    abbrev: Dict[str, str] = {}
    ambiguous = set()
    for k, exps in by_key.items():
        uniq = sorted(set(exps))
        if k in keep:
            continue
        if len(uniq) > 1:
            ambiguous.add(k)
            rep.abbrev_ambiguous.append((k, uniq))
        else:
            abbrev[k] = uniq[0]
            rep.abbrev_ok.append((k, uniq[0]))

    # ---- misspellings ----
    mis = load_misspellings(cfg)
    min_len = cfg["dictionary"]["misspelling_min_len"]
    block = {w.lower() for w in cfg["dictionary"]["misspelling_blocklist"]}
    single: Dict[str, str] = {}
    multi: List[Tuple[str, str]] = []
    for _, r in mis.iterrows():
        src = r["misspelling"].strip()
        cor = r["correction"].strip()
        if not src or not cor:
            continue
        low = src.lower()
        if len(low) < min_len or low in block:
            rep.misspell_skipped.append((src, cor))
            continue
        if " " in src:
            multi.append((src.lower(), cor.lower()))
            rep.misspell_multi += 1
        else:
            single[low] = cor.lower()
            rep.misspell_ok += 1
    multi.sort(key=lambda x: -len(x[0]))

    lex = Lexicon(abbrev=abbrev, symbols=symbols, misspellings=single,
                  misspell_multi=multi, keep=keep, ambiguous=ambiguous)
    return lex, rep


if __name__ == "__main__":
    lex, rep = build_lexicon(load_config())
    print(rep.to_text()[:2000])

