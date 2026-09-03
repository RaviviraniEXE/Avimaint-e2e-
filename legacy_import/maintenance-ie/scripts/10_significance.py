"""Step 10 — statistical rigor: paired-bootstrap significance between tiers, plus
optional multi-seed mean±std for the neural tiers.

Trains the requested tiers on the frozen split, then for every pair of models runs
a paired bootstrap on the test set: is the F1 gap real or within noise? With
--seeds N > 1 it also retrains the neural tiers N times (different seeds) and
reports mean±std, so you can state results as e.g. 0.826 ± 0.004.

  python scripts/10_significance.py --tiers 1 2 3
  python scripts/10_significance.py --tiers 2 3 --seeds 3

Outputs:
  outputs/reports/tables/significance_pairs.csv   (A vs B: diff, 95% CI, p, significant?)
  outputs/reports/tables/seed_variance.csv        (mean±std per tier, if --seeds>1)
"""
import _bootstrap  # noqa: F401
import argparse
import itertools
import os
import statistics

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import entity_scores, relation_scores, paired_bootstrap
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.models.relation_logreg import RelationClassifier
from src.schema import bio_tags, entity_types, load_schema, relation_types


def _docs(recs):
    return [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], d["bio"]),
             "relations": d.get("relations", []), "bio": d["bio"]} for d in recs]


def _tier1(tr, te, schema, tune, dv):
    crf = CRFTagger.tuned(tr, dv)[0] if tune else CRFTagger().fit(tr)
    pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
            for d, b in zip(te, crf.predict(te))]
    if any(r.get("relations") for r in tr):
        rc = RelationClassifier.tuned(schema, tr, dv)[0] if tune else RelationClassifier(schema).fit(tr)
        for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
            p["relations"] = rp
    return pred


def _tier2(tr, dv, te, schema, seed):
    from src.models.bilstm_crf import BiLSTMCRF, build_char_vocab
    from src.models.embeddings import load_matrix
    from src.models.relation_bilstm import NeuralRelationClassifier
    vocab = BiLSTMCRF.build_vocab(tr)
    tagger = BiLSTMCRF(vocab, bio_tags(schema), pretrained=load_matrix(vocab, dim=100),
                       char_vocab=build_char_vocab(tr), seed=seed).fit(tr, dev=dv, epochs=60, patience=8)
    pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
            for d, b in zip(te, tagger.predict(te))]
    rc = NeuralRelationClassifier(schema, vocab, entity_types(schema), relation_types(schema),
                                  seed=seed).fit(tr, dev=dv, epochs=40, patience=6)
    for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
        p["relations"] = rp
    return pred


def _tier3(tr, dv, te, schema, seed):
    from src.models.transformer_ie import TransformerNER, TransformerRE
    cfg = schema.get("models", {}).get("transformer", {})
    enc = cfg.get("encoder", "distilbert-base-uncased"); ml = cfg.get("max_len", 128)
    nc, rc_ = cfg.get("ner", {}), cfg.get("re", {})
    ner = TransformerNER(bio_tags(schema), model_name=enc, max_len=ml, seed=seed).fit(
        tr, dev=dv, epochs=nc.get("epochs", 10), lr=float(nc.get("lr", 3e-5)),
        batch_size=nc.get("batch_size", 16), patience=nc.get("patience", 3))
    pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
            for d, b in zip(te, ner.predict(te))]
    re = TransformerRE(schema, entity_types(schema), relation_types(schema), model_name=enc, max_len=ml, seed=seed).fit(
        tr, dev=dv, epochs=rc_.get("epochs", 10), lr=float(rc_.get("lr", 3e-5)),
        batch_size=rc_.get("batch_size", 8), patience=rc_.get("patience", 3))
    for p, rp in zip(pred, re.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
        p["relations"] = rp
    return pred


NAMES = {"1": "Tier1_CRF_LogReg", "2": "Tier2_BiLSTM_Neural", "3": "Tier3_Transformer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="legacy count; --seeds 3 uses the thesis seeds 42,123,2026")
    ap.add_argument("--seed-list", nargs="+", type=int,
                    help="explicit neural seeds; thesis contract: 42 123 2026")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    schema = load_schema()
    os.makedirs("outputs/reports/tables", exist_ok=True)

    gold = load_gold("outputs/gold/*.jsonl")
    if not load_splits():
        raise SystemExit("Frozen split required for thesis significance runs. Run 02_freeze_split.bat first.")
    tr, dv, te = assign(gold)
    print(f"FROZEN split: train={len(tr)} dev={len(dv)} test={len(te)}")
    te_docs = _docs(te)

    # ---- single canonical run per tier (seed 42) -> predictions for significance ----
    preds = {}
    for t in args.tiers:
        print(f"training {NAMES[t]} ...")
        if t == "1":
            preds[NAMES[t]] = _tier1(tr, te, schema, args.tune, dv)
        elif t == "2":
            preds[NAMES[t]] = _tier2(tr, dv, te, schema, 42)
        else:
            preds[NAMES[t]] = _tier3(tr, dv, te, schema, 42)

    # ---- pairwise significance ----
    rows = []
    for a, b in itertools.combinations(preds, 2):
        for kind in ("entity", "relation"):
            r = paired_bootstrap(te_docs, preds[a], preds[b], kind, n=args.bootstrap)
            rows.append({"model_A": a, "model_B": b, "task": kind, **r})
            verdict = "A>B (sig)" if (r["significant"] and r["diff"] > 0) else \
                      "B>A (sig)" if r["significant"] else "tie (n.s.)"
            print(f"  {kind:8} {a} vs {b}: diff={r['diff']:+.4f} "
                  f"[{r['diff_lo']:+.4f},{r['diff_hi']:+.4f}] p={r['p_value']}  -> {verdict}")
    pd.DataFrame(rows).to_csv("outputs/reports/tables/significance_pairs.csv", index=False)

    # ---- multi-seed mean/std for neural tiers ----
    # Thesis contract is explicit, not "42+s": 42, 123, 2026.
    if args.seed_list:
        seed_list = list(dict.fromkeys(args.seed_list))
    elif args.seeds == 3:
        seed_list = [42, 123, 2026]
    elif args.seeds > 1:
        # Backward-compatible fallback for exploratory runs only.
        seed_list = [42 + s for s in range(args.seeds)]
        print(f"[warn] exploratory generated seeds={seed_list}; thesis-final runs should use --seed-list 42 123 2026")
    else:
        seed_list = [42]

    if len(seed_list) > 1:
        print(f"multi-seed contract: {seed_list}")
        vrows = []
        for t in args.tiers:
            if t == "1":
                continue                      # CRF/LogReg are deterministic
            ent, rel = [], []
            for seed in seed_list:
                p = _tier2(tr, dv, te, schema, seed) if t == "2" else _tier3(tr, dv, te, schema, seed)
                ent.append(entity_scores(te_docs, p)["micro_f1"])
                rel.append(relation_scores(te_docs, p)["micro_f1"])
                print(f"  {NAMES[t]} seed {seed}: entity={ent[-1]:.4f} relation={rel[-1]:.4f}")
            vrows.append({"model": NAMES[t], "seeds": ",".join(map(str, seed_list)),
                          "n_seeds": len(seed_list),
                          "entity_mean": round(statistics.mean(ent), 4),
                          "entity_std": round(statistics.pstdev(ent), 4),
                          "relation_mean": round(statistics.mean(rel), 4),
                          "relation_std": round(statistics.pstdev(rel), 4)})
        pd.DataFrame(vrows).to_csv("outputs/reports/tables/seed_variance.csv", index=False)
        print("seed variance -> outputs/reports/tables/seed_variance.csv")

    print("\nSaved -> outputs/reports/tables/significance_pairs.csv")


if __name__ == "__main__":
    main()
