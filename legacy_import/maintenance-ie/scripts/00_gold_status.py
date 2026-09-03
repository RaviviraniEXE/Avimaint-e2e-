"""Show the combined gold corpus (all outputs/gold/*.jsonl unioned) and optionally
merge it into one file. Every training step already trains on this union — this
script just lets you SEE it and track progress toward the target.

  python scripts/00_gold_status.py            # print status
  python scripts/00_gold_status.py --merge     # also write outputs/gold_all.jsonl
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import os
from collections import Counter

from src.data.gold import load_gold
from src.data.split import load_splits
from src.schema import entity_types, load_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    schema = load_schema()

    files = sorted(glob.glob("outputs/gold/*.jsonl"))
    if not files:
        print("No gold yet in outputs/gold/. Annotate the pilot and import it first.")
        return
    gold = load_gold("outputs/gold/*.jsonl")

    print(f"GOLD CORPUS — {len(gold)} records across {len(files)} batch files")
    for f in files:
        n = sum(1 for _ in open(f))
        print(f"  {os.path.basename(f):24} {n}")

    ent = Counter(e["type"] for r in gold for e in r.get("entities", []))
    rel = Counter(rr["type"] for r in gold for rr in r.get("relations", []))
    n_rel = sum(len(r.get("relations", [])) for r in gold)

    print("\nEntity support (spans):")
    for e in entity_types(schema):
        flag = "  <-- LOW" if ent.get(e, 0) < 50 else ""
        print(f"  {e:14} {ent.get(e,0)}{flag}")
    print(f"\nRelations: {n_rel} instances across {len(rel)} types")
    for r, c in rel.most_common():
        print(f"  {r:26} {c}")

    sp = load_splits()
    if sp:
        print(f"\nFrozen split: train={len(sp['train'])} dev={len(sp['dev'])} test={len(sp['test'])}")
    else:
        print("\nNo frozen split yet (run 03_freeze_test.py once you have ~800 random gold).")

    if args.merge:
        out = "outputs/gold_all.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for r in gold:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nMerged -> {out} ({len(gold)} records)")


if __name__ == "__main__":
    main()
