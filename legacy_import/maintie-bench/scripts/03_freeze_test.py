"""Step 3 — freeze the random test/dev split (do this ONCE, early).

Run after the first random rounds reach annotation.freeze_after gold records.
Assigns whole duplicate groups to a random test/dev set that active learning will
never touch, keeping F1 honest. Writes outputs/splits.json.

  python scripts/03_freeze_test.py
"""
import _bootstrap  # noqa: F401
from src.data.gold import load_gold
from src.data.split import freeze
from src.schema import load_schema


def main():
    schema = load_schema()
    ann = schema["annotation"]
    gold = load_gold("outputs/gold/*.jsonl")
    if len(gold) < ann["freeze_after"]:
        print(f"Only {len(gold)} gold records; freeze_after={ann['freeze_after']}. "
              "Annotate more random rounds first (do NOT freeze on too little).")
        return
    sp = freeze(gold, ann["split"]["test"], ann["split"]["dev"], ann["seed"])
    print(f"Frozen split -> outputs/splits.json")
    print(f"  test={len(sp['test'])}  dev={len(sp['dev'])}  train={len(sp['train'])}")
    print("Active-learning rounds from now on grow TRAIN only; test/dev stay fixed.")


if __name__ == "__main__":
    main()

