"""Step 6 — export gold to SpERT format using the frozen split.

  python scripts/06_export_spert.py
Then train the official SpERT (github.com/lavis-nlp/spert) on outputs/spert/.
"""
import _bootstrap  # noqa: F401
from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.models.spert_export import export_spert, write_config, write_types
from src.schema import load_schema


def main():
    schema = load_schema()
    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No gold found.")
    if load_splits():
        tr, dv, te = assign(gold)
    else:
        tr, dv, te = grouped_split(gold, seed=schema["annotation"]["seed"])
    for name, part in [("train", tr), ("dev", dv), ("test", te)]:
        print(name, len(part), "->", export_spert(part, "outputs/spert", name))
    print("types  ->", write_types("outputs/spert"))
    print("config ->", write_config("outputs/spert"))
    print("\nNext: see SPERT_SETUP.md — train the official SpERT with this config,")
    print("then: python scripts/06b_import_spert_preds.py outputs/spert/predictions_test.json")


if __name__ == "__main__":
    main()

