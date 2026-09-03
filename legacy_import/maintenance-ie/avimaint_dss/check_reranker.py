"""Fail-fast health check for the optional AviMaint cross-encoder reranker."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from core.reranker import CrossEncoderReranker

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        model = cfg["retrieval"].get("reranker_model") or ""

    if not model:
        print("RERANKER OFF: retrieval.reranker_model is empty in config.yaml")
        return 2

    print(f"Loading reranker: {model}")
    r = CrossEncoderReranker(model)
    if not r.available():
        print("RERANKER LOAD FAILED")
        print(r.last_error())
        print("First use of a Hugging Face model requires internet access to download/cache it.")
        return 3

    print(f"RERANKER READY: backend={r.backend()}")
    scores = r.rerank(
        "engine runs rough",
        [(0, "engine runs rough on run up", 0.9), (1, "intake gasket leaking", 0.4)],
    )
    print("sanity_scores=", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
