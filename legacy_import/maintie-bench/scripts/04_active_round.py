"""Step 4 — run one annotation round: train on current gold, pre-label the next
batch, export it for Label Studio correction. No overlap with existing gold or
the frozen test/dev.

Modes (auto-selected, or forced with --mode):
  random  : next N random pool records (rounds 1-2, before/after freeze)
  active  : active-learning selection — rank pool by rare-class presence +
            model uncertainty (round 3+, rare enrichment). TRAIN-pool only.

  python scripts/04_active_round.py --name round1 --n 500 --mode random
  python scripts/04_active_round.py --name rare1  --n 400 --mode active
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

from src.data.active_learning import rank_pool
from src.data.corpus import annotated_idents, load
from src.data.gold import load_gold
from src.data.labelstudio import to_tasks
from src.data.preannotate import preannotate
from src.data.sampling import random_sample
from src.data.split import load_splits
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.models.relation_logreg import RelationClassifier
from src.schema import load_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", choices=["random", "active", "auto"], default="auto")
    args = ap.parse_args()
    schema = load_schema()
    ann = schema["annotation"]

    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No gold yet — annotate the pilot and import it first.")
    splits = load_splits()
    mode = args.mode
    if mode == "auto":
        mode = "active" if splits else "random"

    # training set = TRAIN portion if frozen, else all gold
    if splits:
        train_ids = set(splits["train"])
        train_gold = [r for r in gold if r["ident"] in train_ids] or gold
    else:
        train_gold = gold
    print(f"[{args.name}] mode={mode}  training CRF on {len(train_gold)} gold records...")
    crf = CRFTagger().fit(train_gold)
    rel = None
    if any(r.get("relations") for r in train_gold):
        try:
            rel = RelationClassifier(schema).fit(train_gold)
        except Exception as e:
            print("  (relation pre-fill skipped:", e, ")")

    # candidate pool: unique, not annotated, not in test/dev
    _, pool, nmap, _ = load()
    exclude = annotated_idents()
    if splits:
        exclude |= set(splits["test"]) | set(splits["dev"])
    pool = pool[~pool["IDENT"].isin(exclude)]

    if mode == "random":
        sel = random_sample(pool, args.n, ann["seed"])
        recs = []
        for _, r in sel.iterrows():
            toks = preannotate(r["text"])["tokens"]
            bio = crf.predict_bio(toks)
            recs.append({"ident": r["IDENT"], "tokens": toks, "bio": bio,
                         "entities": bio_to_entities(toks, bio), "relations": [],
                         "exact_group_id": r["exact_group_id"], "stratum": args.name})
    else:  # active learning
        cand = [{"ident": r["IDENT"], "tokens": preannotate(r["text"])["tokens"],
                 "exact_group_id": r["exact_group_id"]} for _, r in pool.iterrows()]
        ranked = rank_pool(crf, cand, schema)[:args.n]
        for r in ranked:
            r["stratum"] = args.name
        recs = ranked
        n_rare = sum(1 for r in recs if r.get("n_rare", 0) > 0)
        print(f"  selected {len(recs)} records; {n_rare} contain a predicted rare class")

    # relation pre-fill
    if rel is not None:
        for r, rp in zip(recs, rel.predict([{"tokens": r["tokens"], "entities": r["entities"]} for r in recs])):
            r["relations"] = rp

    os.makedirs("outputs/rounds", exist_ok=True)
    path = f"outputs/rounds/{args.name}_tasks.json"
    json.dump(to_tasks(recs), open(path, "w"), ensure_ascii=False, indent=1)
    print(f"Wrote {len(recs)} pre-labeled records -> {path}")
    print("Import into Label Studio, correct, export, then: 02_import_gold.py --name", args.name)


if __name__ == "__main__":
    main()

