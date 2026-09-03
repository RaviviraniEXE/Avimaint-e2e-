"""Input/output helpers for normalization artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".jsonl":
        return pd.read_json(source, lines=True)
    raise ValueError(f"Unsupported table format: {source}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(target, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif suffix == ".jsonl":
        frame.to_json(target, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {target}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
