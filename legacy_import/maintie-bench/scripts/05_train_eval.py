"""Step 5 — train the tiers on TRAIN and evaluate on the FROZEN TEST split.

  python scripts/05_train_eval.py --tiers 1        # CRF + LogReg
  python scripts/05_train_eval.py --tiers 1 2      # + BiLSTM-CRF + neural RE (needs torch)

Reports entity & relation micro/macro/per-class F1 on the frozen random test set
(never enriched by active learning).

IMPORTANT FOR MAINTIE:
  Evaluation uses the ORIGINAL full MaintIE entity spans and relations, not the
  flattened BIO proxy. BIO tiers are therefore evaluated under their true
  representational ceiling on nested/overlapping spans and remain directly
  comparable to span-based SpERT.

Results are written to stable run-specific and timestamped files; the legacy
ie_results.json alias is retained only for compatibility.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import entity_scores, relation_scores
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.models.relation_logreg import RelationClassifier
from src.schema import bio_tags, entity_types, load_schema, relation_types


def _goldd(recs):
    # MaintIE contains overlapping/nested spans.  Use the original full span gold
    # for evaluation; BIO is only the input/output representation of BIO models.
    return [{"tokens": d["tokens"],
             "entities": d.get("entities") or bio_to_entities(d["tokens"], d["bio"]),
             "relations": d.get("relations", [])} for d in recs]


def _safe_run_id(value: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_results_safely(payload, run_id, predictions, split_counts):
    os.makedirs("outputs/reports/result_history", exist_ok=True)
    safe = _safe_run_id(run_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generic = "outputs/reports/ie_results.json"

    # Preserve whatever the generic compatibility alias currently contains.
    if os.path.exists(generic):
        archive = f"outputs/reports/result_history/generic_before__{safe}__{stamp}.json"
        shutil.copy2(generic, archive)

    stable = f"outputs/reports/ie_results__{safe}.json"
    hist = f"outputs/reports/result_history/ie_results__{safe}__{stamp}.json"
    _write_json(stable, payload)
    _write_json(hist, payload)
    _write_json(generic, payload)  # legacy alias only

    pred_dir = f"outputs/predictions/{safe}"
    for name, pred in predictions.items():
        _write_json(f"{pred_dir}/{name}_test.json", pred)

    manifest = {
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": stamp,
        "evaluation_gold": "original_full_spans_and_relations",
        "bio_policy": "earliest_then_longest_nonoverlap_for_BIO_models_only",
        "test_policy": "frozen_test_after_dev_only_selection",
        "split": split_counts,
        "results_stable": stable,
        "results_history": hist,
        "predictions_dir": pred_dir,
        "generic_alias": generic,
    }
    _write_json(f"outputs/reports/ie_results__{safe}_manifest.json", manifest)
    _write_json(f"outputs/reports/result_history/ie_results__{safe}__{stamp}_manifest.json", manifest)
    return stable, hist, pred_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["1"], choices=["1", "2", "3"])
    ap.add_argument("--tune", action="store_true", help="grid-search CRF c1/c2 on dev (recommended for final numbers)")
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()
    schema = load_schema()

    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No gold found.")
    if load_splits():
        tr, dv, te = assign(gold)
        print(f"using FROZEN split: train={len(tr)} dev={len(dv)} test={len(te)}")
    else:
        tr, dv, te = grouped_split(gold, seed=schema["annotation"]["seed"])
        print(f"[warn] no frozen split — using grouped split train={len(tr)} test={len(te)}. "
              "Run 03_freeze_test.py for the honest protocol.")
    if not te:
        raise SystemExit("Empty test set.")
    goldd = _goldd(te)
    rows = []
    predictions = {}

    tune = schema.get("models", {}).get("tuning", {})
    do_tune = bool(args.tune and dv)

    if "1" in args.tiers:
        crf = (CRFTagger.tuned(tr, dv, [(a, b) for a in tune.get("crf", {}).get("c1", [0.1])
                                        for b in tune.get("crf", {}).get("c2", [0.1])])[0]
               if do_tune else CRFTagger().fit(tr))
        pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
                for d, b in zip(te, crf.predict(te))]
        es = entity_scores(goldd, pred)
        if any(r.get("relations") for r in tr):
            rc = (RelationClassifier.tuned(schema, tr, dv, tune.get("logreg", {}).get("C", [1.0]))[0]
                  if do_tune else RelationClassifier(schema).fit(tr))
        else:
            rc = None
        if rc:
            for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
        rs = relation_scores(goldd, pred)
        rows.append(("Tier1_CRF_LogReg", es, rs))
        predictions["Tier1_CRF_LogReg"] = pred
        print(f"[Tier1] entity micro-F1={es['micro_f1']} macro-F1={es['macro_f1']} | relation micro-F1={rs['micro_f1']}")
        print("  entity per-class F1:", {k: v[2] for k, v in es["per_class"].items()})

    if "2" in args.tiers:
        try:
            from src.models.bilstm_crf import BiLSTMCRF, build_char_vocab
            from src.models.embeddings import load_matrix
            from src.models.relation_bilstm import NeuralRelationClassifier
            vocab = BiLSTMCRF.build_vocab(tr)
            char_vocab = build_char_vocab(tr)                 # char-CNN features
            pretrained = load_matrix(vocab, dim=100)          # domain embeddings if present
            print(f"[Tier2] char features ON | pretrained embeddings: "
                  f"{'yes' if pretrained is not None else 'no (run 08_make_embeddings.py)'}")
            if do_tune:
                tagger = BiLSTMCRF.tuned(vocab, bio_tags(schema), tr, dv, grid=tune.get("bilstm"),
                                         pretrained=pretrained, char_vocab=char_vocab)[0]
            else:
                tagger = BiLSTMCRF(vocab, bio_tags(schema), pretrained=pretrained,
                                   char_vocab=char_vocab).fit(tr, dev=dv, epochs=60, patience=8)
            pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
                    for d, b in zip(te, tagger.predict(te))]
            es = entity_scores(goldd, pred)
            rc = NeuralRelationClassifier(schema, vocab, entity_types(schema),
                                          relation_types(schema)).fit(tr, dev=dv, epochs=40, patience=6)
            for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
            rs = relation_scores(goldd, pred)
            rows.append(("Tier2_BiLSTM_Neural", es, rs))
            predictions["Tier2_BiLSTM_Neural"] = pred
            print(f"[Tier2] entity micro-F1={es['micro_f1']} | relation micro-F1={rs['micro_f1']}")
        except ImportError as e:
            print(f"[Tier2] skipped (pip install torch pytorch-crf): {e}")

    if "3" in args.tiers:
        try:
            from src.models.transformer_ie import TransformerNER, TransformerRE
            cfg = schema.get("models", {}).get("transformer", {})
            enc = cfg.get("encoder", "distilbert-base-uncased")
            nc, rcf = cfg.get("ner", {}), cfg.get("re", {})
            print(f"[Tier3] transformer encoder = {enc}")
            if do_tune:
                ner = TransformerNER.tuned(bio_tags(schema), tr, dv, model_name=enc,
                                           max_len=cfg.get("max_len", 128),
                                           lrs=tune.get("transformer", {}).get("lr", [3e-5]),
                                           epochs=nc.get("epochs", 10), batch_size=nc.get("batch_size", 16),
                                           patience=nc.get("patience", 3))[0]
            else:
                ner = TransformerNER(bio_tags(schema), model_name=enc, max_len=cfg.get("max_len", 128)).fit(
                    tr, dev=dv, epochs=nc.get("epochs", 10), lr=float(nc.get("lr", 3e-5)),
                    batch_size=nc.get("batch_size", 16), patience=nc.get("patience", 3))
            pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
                    for d, b in zip(te, ner.predict(te))]
            es = entity_scores(goldd, pred)
            re = TransformerRE(schema, entity_types(schema), relation_types(schema), model_name=enc,
                               max_len=cfg.get("max_len", 128)).fit(
                tr, dev=dv, epochs=rcf.get("epochs", 10), lr=float(rcf.get("lr", 3e-5)),
                batch_size=rcf.get("batch_size", 8), patience=rcf.get("patience", 3))
            for p, rp in zip(pred, re.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
            rs = relation_scores(goldd, pred)
            rows.append(("Tier3_Transformer", es, rs))
            predictions["Tier3_Transformer"] = pred
            print(f"[Tier3] entity micro-F1={es['micro_f1']} | relation micro-F1={rs['micro_f1']}")
        except ImportError as e:
            print(f"[Tier3] skipped (pip install transformers): {e}")

    os.makedirs("outputs/reports", exist_ok=True)
    payload = {n: {"entity": es, "relation": rs} for n, es, rs in rows}
    stable, hist, pred_dir = _save_results_safely(
        payload, args.run_id, predictions,
        {"train": len(tr), "dev": len(dv), "test": len(te)})
    log = pd.DataFrame([{"run_id": args.run_id, "tier": n, "n_test": len(te),
                         "ent_micro_f1": es["micro_f1"], "ent_macro_f1": es["macro_f1"],
                         "rel_micro_f1": rs["micro_f1"], "rel_macro_f1": rs["macro_f1"]}
                        for n, es, rs in rows])
    p = "outputs/reports/ie_results_log.csv"
    if os.path.exists(p):
        log = pd.concat([pd.read_csv(p), log], ignore_index=True)
    log.to_csv(p, index=False)
    print(f"\nStable results -> {stable}")
    print(f"History snapshot -> {hist}")
    print(f"Frozen-test predictions -> {pred_dir}")
    print("Compatibility alias -> outputs/reports/ie_results.json")
    print("Log -> outputs/reports/ie_results_log.csv")


if __name__ == "__main__":
    main()

