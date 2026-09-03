"""Create an explicitly labeled rule-derived silver training set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from avimaint.normalization.io import read_table, require_columns, write_table
from avimaint.normalization.rules import load_replacements, normalize_rules


def make_silver(config: dict[str, Any]) -> Path:
    audit = read_table(config["outputs"]["pair_audit_csv"])
    require_columns(audit, ["record_id", "raw_text"], "pair audit")
    held_out_ids: set[str] = set()
    split_path = Path(config["outputs"]["frozen_split_path"])
    if split_path.exists():
        split = read_table(split_path)
        require_columns(split, ["record_id", "split"], "frozen split")
        held_out_ids = set(
            split.loc[split["split"].isin(["validation", "test"]), "record_id"].astype(str)
        )
    frame = audit[~audit["record_id"].astype(str).isin(held_out_ids)].copy()
    frame = frame[frame["raw_text"].fillna("").astype(str).str.len() > 0]
    replacements = load_replacements(config.get("rules", {}).get("dictionary_path"))
    frame["input_text"] = "normalize: " + frame["raw_text"].astype(str)
    frame["target_text"] = [normalize_rules(value, replacements) for value in frame["raw_text"]]
    frame["supervision"] = "silver_rules"
    output = Path(config["outputs"]["silver_train_parquet"])
    write_table(frame, output)
    return output
