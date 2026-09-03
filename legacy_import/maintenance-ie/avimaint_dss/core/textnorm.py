"""Per-field text normalization (abbreviations, misspellings, symbols).

This solves the "normalized data is combined problem+action" problem by
normalizing PROBLEM and ACTION *separately* with the same rules — so the
dashboard gets normalized text without ever needing the combined file.

Two sources of rules, in priority order:
  1. Your exact thesis lists, if present in the data folder:
       data/abbreviations.csv  (columns: abbrev,expansion)
       data/misspellings.csv   (columns: wrong,correct)
       data/keep.csv           (column:  word)      # words to leave untouched
     (System B from your normalization component.)
  2. A strong built-in aviation-maintenance fallback map (below).

Design choices match your normalization component:
  * number convention = DIGITS (best for IE) — digits are left as-is.
  * ambiguous abbreviations are NOT expanded (kept in _AMBIGUOUS skip set).
  * changes are token-level and order-preserving, so problem/action stay separate.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

# --- built-in aviation abbreviation map (whole-word, lowercase keys) -------- #
_ABBREV = {
    "l/h": "left", "lh": "left", "r/h": "right", "rh": "right",
    "a/c": "aircraft", "ac": "aircraft", "acft": "aircraft",
    "eng": "engine", "engs": "engines", "cyl": "cylinder", "cyls": "cylinders",
    "insp": "inspect", "inspd": "inspected", "inspt": "inspect",
    "r&r": "removed and replaced", "r/r": "removed and replaced",
    "rmvd": "removed", "rmv": "remove", "instl": "install", "instld": "installed",
    "instld": "installed", "repl": "replaced", "rplcd": "replaced", "rpld": "replaced",
    "c/w": "complied with", "ck": "check", "ckd": "checked", "cka": "check",
    "cks": "checks", "ops": "operational", "op": "operational",
    "lk": "leak", "lkg": "leaking", "temp": "temperature", "press": "pressure",
    "fwd": "forward", "aft": "aft", "gen": "generator", "alt": "alternator",
    "mag": "magneto", "mags": "magnetos", "carb": "carburetor",
    "gskt": "gasket", "gkt": "gasket", "gskt": "gasket", "gas": "gasket",
    "exh": "exhaust", "assy": "assembly", "asss": "assembly", "hdw": "hardware",
    "rpm": "rpm", "psi": "psi", "qty": "quantity", "qt": "quart",
    "sys": "system", "elec": "electrical", "hyd": "hydraulic",
    "lg": "landing gear", "ldg": "landing", "t/o": "takeoff", "b/u": "backup",
    "brg": "bearing", "rvt": "rivet", "scr": "screw", "wshr": "washer",
    "sat": "satisfactory", "disch": "discharge", "vac": "vacuum",
    "no": "number", "no.": "number", "nbr": "number", "pos": "position",
    "w/": "with", "w/o": "without", "b/w": "between", "thru": "through",
    "prop": "propeller", "govnr": "governor", "gov": "governor",
    "cont": "continue", "rswg": "reswaged", "torqd": "torqued",
    "sec": "secure", "secd": "secured", "reinst": "reinstall",
    "clnd": "cleaned", "cln": "clean", "adj": "adjust", "adjd": "adjusted",
    "found": "found", "flt": "flight", "gnd": "ground", "servd": "serviced",
    "svc": "service", "svcd": "serviced", "replcd": "replaced",
}
# deliberately NOT expanded (real ambiguity — matches your System B skips)
_AMBIGUOUS = {"comp", "in", "off", "of", "test", "run", "up"}

# --- built-in common misspellings ------------------------------------------ #
_MISSPELL = {
    "gasget": "gasket", "gaskit": "gasket", "casket": "gasket",
    "cylender": "cylinder", "cylnder": "cylinder", "cylinder": "cylinder",
    "leking": "leaking", "leeking": "leaking", "lekaing": "leaking",
    "replced": "replaced", "replasced": "replaced", "instaled": "installed",
    "inspet": "inspect", "inspeted": "inspected", "exaust": "exhaust",
    "thottle": "throttle", "throtle": "throttle", "magneto": "magneto",
    "compresion": "compression", "compresson": "compression",
    "loos": "loose", "brocken": "broken", "craked": "cracked", "crackd": "cracked",
    "worm": "worn", "seperated": "separated", "seperate": "separate",
    "recieved": "received", "peformed": "performed", "perfomed": "performed",
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9/&\.\-]*|[^\sa-z0-9]", re.IGNORECASE)
_WS = re.compile(r"\s+")


class Normalizer:
    def __init__(self, abbrev: dict | None = None, misspell: dict | None = None,
                 keep: set | None = None):
        self.abbrev = {**_ABBREV, **(abbrev or {})}
        self.misspell = {**_MISSPELL, **(misspell or {})}
        self.keep = keep or set()

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "Normalizer":
        d = Path(data_dir)
        abbrev, misspell, keep = {}, {}, set()

        def _read(name, a, b=None):
            p = d / name
            if not p.is_file():
                return None
            rows = {}
            with open(p, encoding="utf-8-sig", newline="") as fh:
                rd = csv.reader(fh)
                header = next(rd, None)
                for r in rd:
                    if len(r) >= 2 and r[0].strip():
                        rows[r[0].strip().lower()] = r[1].strip()
                    elif b is None and r and r[0].strip():
                        rows[r[0].strip().lower()] = None
            return rows

        ab = _read("abbreviations.csv", "abbrev", "expansion")
        ms = _read("misspellings.csv", "wrong", "correct")
        kp = _read("keep.csv", "word")
        if ab:
            abbrev = {k: v for k, v in ab.items() if v}
        if ms:
            misspell = {k: v for k, v in ms.items() if v}
        if kp:
            keep = set(kp.keys())
        return cls(abbrev=abbrev, misspell=misspell, keep=keep)

    def _map_token(self, tok: str) -> str:
        low = tok.lower()
        if low == "&":
            return "and"
        if low in self.keep or low in _AMBIGUOUS:
            return tok
        if low in self.misspell:
            return self.misspell[low]
        if low in self.abbrev:
            return self.abbrev[low]
        # strip trailing period abbreviations like "no."
        if low.endswith(".") and low[:-1] in self.abbrev:
            return self.abbrev[low[:-1]]
        return tok

    @lru_cache(maxsize=20000)
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        toks = _TOKEN.findall(text)
        out = [self._map_token(t) for t in toks]
        joined = " ".join(out)
        joined = re.sub(r"\s+([.,;:])", r"\1", joined)
        return _WS.sub(" ", joined).strip()


_DEFAULT = Normalizer()


def normalize_text(text: str) -> str:
    return _DEFAULT.normalize(text)

