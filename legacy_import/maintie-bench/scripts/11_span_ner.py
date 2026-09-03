"""Step 11 — BIO vs span-based NER ablation (nesting-capable heads).

Trains, on the frozen split, four NER models with the SAME two encoders but two
different OUTPUT representations:

  BiLSTM      : BIO (BiLSTM-CRF)          vs  span (BiLSTMSpanNER)
  Transformer : BIO (token-classification) vs span (TransformerSpanNER)

and reports, on the test set:
  * entity micro-F1 and macro-F1
  * NESTED-entity recall — recall restricted to gold entities that overlap another
    gold entity. BIO cannot represent overlaps, so its nested recall is ~0; the
    span heads recover them. This is the core finding.

  python scripts/11_span_ner.py                 # both encoders
  python scripts/11_span_ner.py --encoders transformer
  python scripts/11_span_ner.py --tune          # (transformer lr grid, optional)

Writes: outputs/reports/tables/span_ner_ablation.csv
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import entity_scores
from src.schema import bio_tags, entity_types, load_schema


def nested_gold(docs):
    """Set of (doc_idx, start, end, type) for gold entities that overlap another."""
    out = set()
    for di, d in enumerate(docs):
        es = d["entities"]
        for a in range(len(es)):
            for b in range(len(es)):
                if a == b:
                    continue
                if es[a]["start"] < es[b]["end"] and es[b]["start"] < es[a]["end"]:
                    out.add((di, es[a]["start"], es[a]["end"], es[a]["type"]))
                    break
    return out


def pred_set(preds):
    s = set()
    for di, ents in enumerate(preds):
        for e in ents:
            s.add((di, e["start"], e["end"], e["type"]))
    return s


def nested_recall(te, preds):
    g = nested_gold(te)
    if not g:
        return 0.0, 0
    p = pred_set(preds)
    return round(len(g & p) / len(g), 4), len(g)


def _score(name, te, preds, rows, ng):
    gold = [{"tokens": d["tokens"], "entities": d["entities"], "relations": []} for d in te]
    prd = [{"tokens": d["tokens"], "entities": e, "relations": []} for d, e in zip(te, preds)]
    es = entity_scores(gold, prd)
    nr, n_nested = nested_recall(te, preds)
    rows.append({"model": name, "entity_micro_f1": es["micro_f1"], "entity_macro_f1": es["macro_f1"],
                 "nested_entity_recall": nr, "n_nested_gold": n_nested})
    print(f"  {name:26} micro-F1={es['micro_f1']:.4f}  macro-F1={es['macro_f1']:.4f}  "
          f"nested-recall={nr:.4f} (of {n_nested})")


def _load_saved_bio(path, te):
    if not path or not os.path.exists(path):
        return None
    docs = json.load(open(path, encoding="utf-8"))
    if len(docs) != len(te):
        raise RuntimeError(f"saved BIO predictions {path}: {len(docs)} docs, expected {len(te)}")
    for i, (g, p) in enumerate(zip(te, docs)):
        if p.get("tokens") != g.get("tokens"):
            raise RuntimeError(f"saved BIO predictions are not aligned at TEST row {i}: {path}")
    return [d.get("entities", []) for d in docs]


def _save_span_predictions(name, te, preds):
    os.makedirs("outputs/predictions/span_ablation", exist_ok=True)
    out = [{"tokens": d["tokens"], "entities": e, "relations": []} for d, e in zip(te, preds)]
    path = f"outputs/predictions/span_ablation/{name}_test.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["bilstm", "transformer"],
                    choices=["bilstm", "transformer"])
    ap.add_argument("--max-span", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--reuse-bio-dir", default=None,
                    help="directory containing saved baseline TEST predictions; avoids retraining BIO baselines")
    ap.add_argument("--require-reuse-bio", action="store_true",
                    help="fail instead of retraining BIO baselines when saved predictions are unavailable")
    args = ap.parse_args()
    schema = load_schema()
    ents = entity_types(schema)
    os.makedirs("outputs/reports/tables", exist_ok=True)

    gold = load_gold("outputs/gold/*.jsonl")
    tr, dv, te = (assign(gold) if load_splits() else grouped_split(gold, seed=schema["annotation"]["seed"]))
    print(f"train={len(tr)} dev={len(dv)} test={len(te)} | nested gold in test: {len(nested_gold(te))}")
    rows = []

    if "bilstm" in args.encoders:
        _bootstrap.banner("BiLSTM  ·  BIO vs SPAN")
        from src.models.bilstm_crf import BiLSTMCRF, build_char_vocab
        from src.models.crf_ner import bio_to_entities
        from src.models.span_ner import BiLSTMSpanNER
        try:
            from src.models.embeddings import load_matrix
            vocab = BiLSTMCRF.build_vocab(tr)
            pre = load_matrix(vocab, dim=100)
        except Exception:
            vocab = BiLSTMCRF.build_vocab(tr); pre = None
        # BIO: reuse the already-trained Tier2 baseline when available.
        saved = (os.path.join(args.reuse_bio_dir, "Tier2_BiLSTM_Neural_test.json")
                 if args.reuse_bio_dir else None)
        preds_bio = _load_saved_bio(saved, te)
        if preds_bio is None:
            if args.require_reuse_bio:
                raise SystemExit(f"Missing saved Tier2 BIO predictions: {saved}. Run 02_train_baselines.bat first.")
            bio = BiLSTMCRF(vocab, bio_tags(schema), pretrained=pre,
                            char_vocab=build_char_vocab(tr)).fit(tr, dev=dv, epochs=60, patience=8)
            preds_bio = [bio_to_entities(d["tokens"], b) for d, b in zip(te, bio.predict(te))]
        else:
            print(f"  [reuse] BiLSTM BIO predictions <- {saved}")
        _score("BiLSTM_BIO", te, preds_bio, rows, ents)
        # span: this is the actual ablation model and therefore must be trained.
        span = BiLSTMSpanNER(vocab, ents, max_span=args.max_span, pretrained=pre).fit(
            tr, dev=dv, epochs=args.epochs, lr=1e-3, batch_size=8, patience=6)
        span_pred = span.predict(te)
        _score("BiLSTM_Span", te, span_pred, rows, ents)
        print("  [saved]", _save_span_predictions("BiLSTM_Span", te, span_pred))

    if "transformer" in args.encoders:
        _bootstrap.banner("Transformer  ·  BIO vs SPAN")
        from src.models.crf_ner import bio_to_entities
        from src.models.span_ner import TransformerSpanNER
        from src.models.transformer_ie import TransformerNER
        cfg = schema.get("models", {}).get("transformer", {})
        enc = cfg.get("encoder", "distilbert-base-uncased"); ml = cfg.get("max_len", 128)
        nc = cfg.get("ner", {})
        # BIO: reuse the already-trained Tier3 baseline when available.
        saved = (os.path.join(args.reuse_bio_dir, "Tier3_Transformer_test.json")
                 if args.reuse_bio_dir else None)
        preds_bio = _load_saved_bio(saved, te)
        if preds_bio is None:
            if args.require_reuse_bio:
                raise SystemExit(f"Missing saved Tier3 BIO predictions: {saved}. Run 02_train_baselines.bat first.")
            bio = TransformerNER(bio_tags(schema), model_name=enc, max_len=ml).fit(
                tr, dev=dv, epochs=nc.get("epochs", 10), lr=float(nc.get("lr", 3e-5)),
                batch_size=nc.get("batch_size", 16), patience=nc.get("patience", 3))
            preds_bio = [bio_to_entities(d["tokens"], b) for d, b in zip(te, bio.predict(te))]
        else:
            print(f"  [reuse] Transformer BIO predictions <- {saved}")
        _score("Transformer_BIO", te, preds_bio, rows, ents)
        # span: this is the actual ablation model and therefore must be trained.
        span = TransformerSpanNER(ents, model_name=enc, max_len=ml, max_span=args.max_span).fit(
            tr, dev=dv, epochs=args.epochs, lr=float(nc.get("lr", 3e-5)),
            batch_size=nc.get("batch_size", 8), patience=4)
        span_pred = span.predict(te)
        _score("Transformer_Span", te, span_pred, rows, ents)
        print("  [saved]", _save_span_predictions("Transformer_Span", te, span_pred))

    df = pd.DataFrame(rows)
    df.to_csv("outputs/reports/tables/span_ner_ablation.csv", index=False)
    _bootstrap.banner("SPAN-NER ABLATION  ·  BIO cannot nest; span heads can")
    print(df.to_string(index=False))
    print("\nSaved -> outputs/reports/tables/span_ner_ablation.csv")


if __name__ == "__main__":
    main()
