"""Smoke-test the live dashboard query adapter without changing frozen outputs."""
from __future__ import annotations

import yaml
from pathlib import Path

from core.recommend import Recommender

ROOT = Path(__file__).resolve().parent

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
mode = cfg["extraction"].get("live_query_case_adapter", "none")

samples = [
    "#2 intake gasket leaking",
    "#3 rocker cover leaking and #4 intake gasket leaking",
    "#4 cylinder low compression",
    "left magneto excessive rpm drop during run up",
]

for text in samples:
    if mode == "ascii_uppercase":
        adapted = Recommender._ascii_uppercase(text)
    else:
        adapted = text
    assert len(adapted) == len(text), (text, adapted)
    print("USER :", text)
    print("SPERT:", adapted)
    print()

print("LIVE_QUERY_CASE_ADAPTER_OK", mode)
print("No frozen artifact was modified.")
