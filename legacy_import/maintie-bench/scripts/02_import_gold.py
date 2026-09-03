"""Step 2 — import a corrected Label Studio export into the gold corpus.

  python scripts/02_import_gold.py --export mycorrected.json --name pilot

Converts LS export -> internal JSONL in outputs/gold/<name>.jsonl and appends a
per-round row (gold size + per-class entity support) to the tracking report.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from collections import Counter

import pandas as pd

from src.data.corpus import annotated_idents
from src.data.labelstudio import from_export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Label Studio export JSON")
    ap.add_argument("--name", required=True, help="name for this gold batch (e.g. pilot, round1)")
    args = ap.parse_args()

    recs = from_export(args.export)
    os.makedirs("outputs/gold", exist_ok=True)
    out = f"outputs/gold/{args.name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Imported {len(recs)} records -> {out}")

    # tracking: total gold + per-class support after this import
    total = len(annotated_idents())
    ent = Counter(e["type"] for r in recs for e in r["entities"])
    rel = Counter(rr["type"] for r in recs for rr in r["relations"])
    row = {"batch": args.name, "records_in_batch": len(recs), "gold_total": total,
           **{f"ent_{k}": v for k, v in ent.items()},
           **{f"rel_{k}": v for k, v in rel.items()}}
    os.makedirs("outputs/reports", exist_ok=True)
    path = "outputs/reports/annotation_tracking.csv"
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"gold total now = {total}. Per-class support this batch: {dict(ent)}")
    print(f"tracking -> {path}")


if __name__ == "__main__":
    main()
