"""Step 6b — import official SpERT test predictions into the reporting format.

After training the official SpERT (see SPERT_SETUP.md), it writes a predictions
file (predictions_test.json). This converts it into outputs/reports/spert_test.json
— the {tokens, entities, relations} list that 09_report.py --spert consumes — and
checks it aligns with the frozen test split.

  python scripts/06b_import_spert_preds.py outputs/spert/predictions_test.json
  python scripts/09_report.py --tiers 1 2 3 --spert outputs/reports/spert_test.json --run-id gold1400_spert
"""
import _bootstrap  # noqa: F401
import json
import os
import sys

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.schema import load_schema


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: 06b_import_spert_preds.py <spert_predictions.json>")
    src = sys.argv[1]
    preds = json.load(open(src, encoding="utf-8"))

    # frozen test split, in the same order export_spert wrote it
    schema = load_schema()
    gold = load_gold("outputs/gold/*.jsonl")
    tr, dv, te = (assign(gold) if load_splits()
                  else grouped_split(gold, seed=schema["annotation"]["seed"]))
    if len(preds) != len(te):
        print(f"[warn] {len(preds)} predictions vs {len(te)} test docs — check you trained "
              f"valid_path = outputs/spert/test.json and didn't shuffle.")

    out = []
    for p in preds:
        out.append({
            "tokens": p["tokens"],
            "entities": [{"type": e["type"], "start": e["start"], "end": e["end"]}
                         for e in p.get("entities", [])],
            "relations": [{"type": r["type"], "head": r["head"], "tail": r["tail"]}
                          for r in p.get("relations", [])],
        })
    os.makedirs("outputs/reports", exist_ok=True)
    dst = "outputs/reports/spert_test.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(out)} SpERT test predictions -> {dst}")
    print("now: python scripts/09_report.py --tiers 1 2 3 --spert outputs/reports/spert_test.json --run-id <id>")


if __name__ == "__main__":
    main()
