"""Export a field-preserving normalization output for the dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", default="rules_then_byt5", choices=["raw", "rules", "byt5", "rules_then_byt5"]
    )
    args = parser.parse_args()
    source = Path("outputs/normalization/full_corpus") / f"{args.system}.csv"
    frame = pd.read_csv(source, dtype=str).fillna("")
    required = {
        "record_id",
        "problem_normalized",
        "action_normalized",
        "protected_tokens_preserved",
    }
    if not required.issubset(frame.columns):
        raise SystemExit(f"Missing columns: {sorted(required - set(frame.columns))}")
    unsafe = ~frame["protected_tokens_preserved"].astype(str).str.lower().eq("true")
    # Preserve provenance and safety: records with a protected-token violation
    # fall back field-wise to the raw source instead of being removed.
    problem = frame["problem_normalized"].copy()
    action = frame["action_normalized"].copy()
    problem.loc[unsafe] = frame.loc[unsafe, "problem_raw"]
    action.loc[unsafe] = frame.loc[unsafe, "action_raw"]
    output = pd.DataFrame(
        {
            "IDENT": frame["record_id"],
            "PROBLEM": problem,
            "ACTION": action,
            "NORMALIZATION_SYSTEM": args.system,
            "PROTECTED_FALLBACK": unsafe,
        }
    )
    filename = (
        "dashboard_dataset_D.csv"
        if args.system == "rules_then_byt5"
        else f"dashboard_dataset_{args.system}.csv"
    )
    target = Path("legacy_import/maintenance-ie/avimaint_dss/data") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(f"{len(output)} records -> {target}; protected fallbacks={int(unsafe.sum())}")


if __name__ == "__main__":
    main()
