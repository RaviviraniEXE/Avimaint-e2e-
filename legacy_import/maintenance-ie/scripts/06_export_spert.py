"""Step 6 — export gold to SpERT format using the frozen split.

  python scripts/06_export_spert.py
Then train the official SpERT (github.com/lavis-nlp/spert) on outputs/spert/.
"""
import _bootstrap  # noqa: F401
import argparse
from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.models.spert_export import export_spert, write_config, write_types
from src.schema import load_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-glob", default="outputs/gold/*.jsonl")
    parser.add_argument("--schema-path", default="config/schema.yaml")
    parser.add_argument("--output-dir", default="outputs/spert")
    args = parser.parse_args()
    schema = load_schema(args.schema_path)
    gold = load_gold(args.gold_glob)
    if not gold:
        raise SystemExit("No gold found.")
    if load_splits():
        tr, dv, te = assign(gold)
    else:
        tr, dv, te = grouped_split(gold, seed=schema["annotation"]["seed"])
    for name, part in [("train", tr), ("dev", dv), ("test", te)]:
        print(name, len(part), "->", export_spert(part, args.output_dir, name))
    print("types  ->", write_types(args.output_dir, args.schema_path))
    print("config ->", write_config(args.output_dir))
    print("\nNext: see SPERT_SETUP.md — train the official SpERT with this config,")
    print(
        "then: python scripts/06b_import_spert_preds.py "
        f"{args.output_dir}/predictions_test.json"
    )


if __name__ == "__main__":
    main()
