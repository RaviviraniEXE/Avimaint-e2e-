"""Step 7 — the novel experiment: does normalization improve IE F1?

Trains/evaluates the CRF on the SAME gold under each normalization variant
(A/B/C/D) and compares entity F1 on the frozen test split. Provide, per variant,
the gold with spans projected onto that variant's tokens under
outputs/gold_variants/<VARIANT>/*.jsonl (annotate once on normalized text, project
via the normalization component's alignment files).

  python scripts/07_normalization_ie.py
"""
import _bootstrap  # noqa: F401
import glob
import json
import os

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import entity_scores
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.schema import load_schema


def main():
    schema = load_schema()
    seed = schema["annotation"]["seed"]
    base = "outputs/gold_variants"
    variants = sorted(d for d in glob.glob(f"{base}/*") if os.path.isdir(d))
    if not variants:
        raise SystemExit(f"No variant gold in {base}/<VARIANT>/*.jsonl (see docstring).")
    rows = []
    for v in variants:
        recs = load_gold(f"{v}/*.jsonl")
        if not recs:
            continue
        tr, dv, te = assign(recs) if load_splits() else grouped_split(recs, seed=seed)
        crf = CRFTagger().fit(tr)
        pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b)}
                for d, b in zip(te, crf.predict(te))]
        gold = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], d["bio"])} for d in te]
        es = entity_scores(gold, pred)
        rows.append({"variant": os.path.basename(v), "ent_micro_f1": es["micro_f1"],
                     "ent_macro_f1": es["macro_f1"], "n_test": len(te)})
        print(f"{os.path.basename(v):12} entity micro-F1={es['micro_f1']} macro-F1={es['macro_f1']}")
    os.makedirs("outputs/reports", exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/reports/normalization_ie.csv", index=False)
    print("\nSaved -> outputs/reports/normalization_ie.csv  (answers the thesis' core question)")


if __name__ == "__main__":
    main()
