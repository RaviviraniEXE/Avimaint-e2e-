"""Small installation and environment diagnostic command."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys


EXPECTED_MODULES = {
    "avimaint-core": ["yaml", "pandas", "pyarrow", "pydantic"],
    "avimaint-normalization": ["torch", "transformers", "datasets", "jiwer"],
    "avimaint-ie-classical": ["sklearn", "sklearn_crfsuite", "seqeval"],
    "avimaint-ie-neural": ["torch", "transformers", "seqeval"],
    "avimaint-spert": ["torch", "transformers", "sklearn"],
    "avimaint-retrieval": ["sentence_transformers", "rank_bm25", "faiss"],
    "avimaint-dashboard": ["streamlit", "plotly", "altair"],
    "avimaint-dev": ["pytest", "ruff", "mypy"],
}


def doctor(environment: str) -> int:
    missing = [
        module
        for module in EXPECTED_MODULES.get(environment, [])
        if importlib.util.find_spec(module) is None
    ]
    print(f"Environment: {environment}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if missing:
        print(f"Missing modules: {', '.join(missing)}")
        return 1
    print("AviMaint package and expected modules are available.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--environment", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor(args.environment))


if __name__ == "__main__":
    main()

