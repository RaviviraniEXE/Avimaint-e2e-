"""Build the held-out expert-completion sensitivity population.

These targets are kept outside training/model selection because the raw record
does not always contain enough evidence to reconstruct the expert completion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/aviation/interim/normalization_manual_review.csv"
TARGET = ROOT / "data/aviation/processed/normalization_expert_sensitivity.parquet"
STATUSES = {
    "expert_completion_separate",
    "expert_truncation_separate",
    "expert_anonymization_separate",
}


def main() -> None:
    frame = pd.read_csv(SOURCE, dtype=str).fillna("")
    selected = frame[frame["review_status"].isin(STATUSES)].copy()
    if len(selected) != 1045:
        raise SystemExit(f"Expected 1,045 expert-sensitivity pairs; found {len(selected)}")
    selected["input_text"] = [
        f"normalize {field}: {text}"
        for field, text in zip(selected["field"], selected["raw_text"], strict=False)
    ]
    selected["target_text"] = selected["reference_text"].astype(str).str.strip()
    selected["split"] = "sensitivity"
    selected["evaluation_scope"] = "expert_completion_not_primary_gold"
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(TARGET, index=False)
    print(f"{len(selected)} expert-sensitivity pairs -> {TARGET}")


if __name__ == "__main__":
    main()
