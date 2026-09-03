"""Step 3 — freeze the random test/dev split (do this ONCE, early).

Run after the first random rounds reach annotation.freeze_after gold records.
Assigns whole duplicate groups to a random test/dev set that active learning will
never touch, keeping F1 honest. Writes outputs/splits.json.

  python scripts/03_freeze_test.py
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from src.data.gold import load_gold
from src.data.split import freeze
from src.schema import load_schema


def load_files(paths):
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-files",
        nargs="+",
        default=["outputs/gold/pilot.jsonl", "outputs/gold/round1.jsonl"],
        help="Only these unbiased batches may populate test/dev",
    )
    args = parser.parse_args()
    schema = load_schema()
    ann = schema["annotation"]
    random_gold = load_files(args.random_files)
    required = ann["split"]["test"] + ann["split"]["dev"]
    if len(random_gold) < required:
        raise SystemExit(f"Only {len(random_gold)} random records; at least {required} are needed "
                         "to populate the declared test/dev sizes.")
    if len(random_gold) != ann["freeze_after"]:
        raise SystemExit(
            f"Random pool has {len(random_gold)} records; expected exactly {ann['freeze_after']} "
            "from pilot + round1. Rare-enriched batches must never enter dev/test."
        )
    sp = freeze(random_gold, ann["split"]["test"], ann["split"]["dev"], ann["seed"])
    # After the representative split is fixed, every rare/active-learning batch
    # is added to training only.
    gold = load_gold("outputs/gold/*.jsonl")
    sp = freeze(gold, ann["split"]["test"], ann["split"]["dev"], ann["seed"])
    print(f"Frozen split -> outputs/splits.json")
    print(f"  test={len(sp['test'])}  dev={len(sp['dev'])}  train={len(sp['train'])}")
    print("Active-learning rounds from now on grow TRAIN only; test/dev stay fixed.")


if __name__ == "__main__":
    main()
