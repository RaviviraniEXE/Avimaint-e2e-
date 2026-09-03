"""Step 9 — full evaluation report: per-class tables (with support), confusion
matrices, model-comparison figures, and a learning curve. Trains the tiers on the
frozen split and saves everything under outputs/reports/ for the thesis.

  python scripts/09_report.py --tiers 1 2         # CRF + BiLSTM
  python scripts/09_report.py --tiers 1 2 --spert outputs/reports/spert_test.json

Outputs:
  outputs/reports/tables/*.csv        per-class P/R/F1/support, overall, corpus support
  outputs/reports/figures/*.png       overall, per-class, confusion, support, learning curve
  outputs/reports/metrics_full.json   every number in one file
  outputs/reports/f1_trajectory.csv   appended each run -> feeds the learning curve
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.models.relation_logreg import RelationClassifier
from src.evaluate import bootstrap_ci, entity_scores, relation_scores
from src.reporting import (confusion, ensure_dirs, fig_confusion, fig_learning_curve,
                           fig_overall, fig_per_class, fig_support, fig_training_curve,
                           memorization_baseline, per_class_table, FIGDIR, TABDIR)
from src.schema import bio_tags, entity_types, load_schema, relation_types
from collections import Counter

N_BOOT = 1000   # bootstrap resamples for confidence intervals (set via --bootstrap)



def _safe_run_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _write_protected_metrics(payload: dict, run_id: str) -> tuple[str, str, str]:
    """Keep report metrics versioned while preserving the legacy compatibility path."""
    reports = Path("outputs/reports")
    history = reports / "result_history"
    history.mkdir(parents=True, exist_ok=True)
    generic = reports / "metrics_full.json"
    if generic.exists():
        old_stamp = datetime.fromtimestamp(generic.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = history / f"metrics_full__previous__{old_stamp}__preoverwrite.json"
        if not archive.exists():
            shutil.copy2(generic, archive)
            print(f"[result safety] archived previous metrics_full -> {archive}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_slug = _safe_run_slug(run_id)
    versioned = history / f"metrics_full__{run_slug}__{stamp}.json"
    run_alias = reports / f"metrics_full__{run_slug}.json"
    text = json.dumps(payload, indent=2, default=str) + "\n"
    versioned.write_text(text, encoding="utf-8")
    run_alias.write_text(text, encoding="utf-8")
    generic.write_text(text, encoding="utf-8")
    print(f"[result safety] metrics versioned -> {versioned}")
    print(f"[result safety] metrics stable alias -> {run_alias}")
    return str(versioned), str(run_alias), str(generic)


def _docs(recs):
    return [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], d["bio"]),
             "relations": d.get("relations", []), "bio": d["bio"]} for d in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["1", "2"], choices=["1", "2", "3"])
    ap.add_argument("--spert", help="optional SpERT test predictions JSON (Tier 3)")
    ap.add_argument("--external", nargs="+", default=[],
                    help="external model predictions as NAME=path.json (repeatable), e.g. "
                         "REBEL=outputs/reports/rebel_test.json — each is scored on the SAME "
                         "frozen test set and joins the comparison table/figure")
    ap.add_argument("--tune", action="store_true", help="grid-search each model's hyperparameters on dev")
    ap.add_argument("--train-batches", nargs="+",
                    help="restrict TRAINING to these gold batch names (e.g. pilot) — the frozen "
                         "test/dev stay fixed, so stages are comparable for a learning curve")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for 95%% confidence intervals (0 to disable)")
    ap.add_argument("--save-models", action="store_true",
                    help="persist every trained tier model to outputs/models/ (use on the final run)")
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()
    global N_BOOT
    N_BOOT = args.bootstrap
    schema = load_schema()
    ensure_dirs()
    version = MODELDIR = None
    if args.save_models:                         # versioned: never overwrite a prior train
        from datetime import datetime
        version = f"{args.run_id}__{datetime.now():%Y%m%d-%H%M%S}"
        MODELDIR = f"outputs/models/{version}"
        os.makedirs(MODELDIR, exist_ok=True)
        print(f"[save] model version = {version}")

    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No gold found.")
    if load_splits():
        tr, dv, te = assign(gold)
        split_note = "frozen"
    else:
        tr, dv, te = grouped_split(gold, seed=schema["annotation"]["seed"])
        split_note = "grouped (no freeze)"
    if args.train_batches:                       # learning-curve stage: subset TRAIN only
        allowed = set()
        for name in args.train_batches:
            path = f"outputs/gold/{name}.jsonl"
            if not os.path.exists(path):
                raise SystemExit(f"--train-batches: {path} not found")
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    allowed.add(str(json.loads(line)["ident"]))
        n0 = len(tr)
        tr = [r for r in tr if str(r["ident"]) in allowed]
        print(f"train restricted to {args.train_batches}: {n0} -> {len(tr)} "
              f"(test/dev stay frozen — comparable across stages)")
    print(f"split={split_note}  train={len(tr)} dev={len(dv)} test={len(te)}")
    te_docs = _docs(te)
    gold_bio = [d["bio"] for d in te_docs]
    etypes = entity_types(schema)

    metrics, ent_tables, rel_tables, confusions, diagnostics = {}, {}, {}, {}, {}
    tune = schema.get("models", {}).get("tuning", {})
    do_tune = bool(args.tune and dv)
    selection_meta = {}

    # ---- Tier 1: CRF + LogReg ----
    if "1" in args.tiers:
        _bootstrap.banner("TIER 1 / 4  ·  CRF + LogReg  (baseline, fast)")
        if do_tune:
            crf, crf_selection = CRFTagger.tuned(
                tr, dv, [(a, b) for a in tune.get("crf", {}).get("c1", [0.1])
                         for b in tune.get("crf", {}).get("c2", [0.1])]
            )
            selection_meta["tier1_crf"] = crf_selection
            _bootstrap.step(
                f"selected CRF c1={crf_selection['c1']} c2={crf_selection['c2']} "
                f"on DEV entity F1={crf_selection['micro_f1']:.4f}"
            )
        else:
            crf = CRFTagger().fit(tr)
        pred_bio = crf.predict(te)
        pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b),
                 "relations": []} for d, b in zip(te, pred_bio)]
        if any(r.get("relations") for r in tr):
            if do_tune:
                rc, rc_selection = RelationClassifier.tuned(
                    schema, tr, dv, tune.get("logreg", {}).get("C", [1.0])
                )
                selection_meta["tier1_logreg"] = rc_selection
                _bootstrap.step(
                    f"selected LogReg C={rc_selection['C']} on DEV gold-entity relation "
                    f"F1={rc_selection['micro_f1']:.4f}"
                )
            else:
                rc = RelationClassifier(schema).fit(tr)
        else:
            rc = None
        if rc:
            for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
        name = "Tier1_CRF_LogReg"
        _collect(name, te_docs, pred, gold_bio, pred_bio, etypes,
                 metrics, ent_tables, rel_tables, confusions)
        if rc:
            gold_inputs = [{"tokens": d["tokens"], "entities": d["entities"]} for d in te]
            gold_rels = rc.predict(gold_inputs)
            gold_rel_pred = [
                {"tokens": d["tokens"], "entities": d["entities"], "relations": rp}
                for d, rp in zip(te, gold_rels)
            ]
            gold_rel_docs = [
                {"tokens": d["tokens"], "entities": d["entities"],
                 "relations": d.get("relations", [])} for d in te
            ]
            metrics[name]["relation_gold_entities"] = relation_scores(gold_rel_docs, gold_rel_pred)
            _bootstrap.step(
                f"gold-entity RE diagnostic F1={metrics[name]['relation_gold_entities']['micro_f1']:.4f}"
            )
        if args.save_models:
            crf.save(f"{MODELDIR}/tier1_crf.pkl")
            if rc:
                rc.save(f"{MODELDIR}/tier1_logreg.pkl")
            print(f"  [saved] {MODELDIR}/tier1_crf.pkl (+ logreg)")
        # validity diagnostics (is the model really learning, or memorising?)
        mb = memorization_baseline(tr, te)
        tr_pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b), "relations": []}
                   for d, b in zip(tr, crf.predict(tr))]
        tr_gold = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], d["bio"]), "relations": []}
                   for d in tr]
        f_train = entity_scores(tr_gold, tr_pred)["micro_f1"]
        f_test = metrics[name]["entity"]["micro_f1"]
        diagnostics = {"memorization_baseline_f1": mb["memorization_f1"],
                       "test_oov_rate": mb["test_oov_rate"],
                       "tier1_train_f1": round(f_train, 4), "tier1_test_f1": round(f_test, 4),
                       "generalization_gap": round(f_train - f_test, 4),
                       "context_gain_over_memorization": round(f_test - mb["memorization_f1"], 4)}
        print(f"  [diagnostics] memorization-F1={mb['memorization_f1']} "
              f"(model +{diagnostics['context_gain_over_memorization']}) | "
              f"train={f_train:.4f}/test={f_test:.4f} gap={diagnostics['generalization_gap']} | "
              f"test-OOV={mb['test_oov_rate']}")

    # ---- Tier 2: BiLSTM-CRF + neural RE ----
    if "2" in args.tiers:
        _bootstrap.banner("TIER 2 / 4  ·  BiLSTM-CRF + Neural RE  (training, ~5 min)")
        try:
            from src.models.bilstm_crf import BiLSTMCRF, build_char_vocab
            from src.models.embeddings import load_matrix
            from src.models.relation_bilstm import NeuralRelationClassifier
            vocab = BiLSTMCRF.build_vocab(tr)
            pretrained, cv = load_matrix(vocab, dim=100), build_char_vocab(tr)
            if do_tune:
                tagger = BiLSTMCRF.tuned(vocab, bio_tags(schema), tr, dv, grid=tune.get("bilstm"),
                                         pretrained=pretrained, char_vocab=cv)[0]
            else:
                tagger = BiLSTMCRF(vocab, bio_tags(schema), pretrained=pretrained,
                                   char_vocab=cv).fit(tr, dev=dv, epochs=60, patience=8)
            pred_bio = tagger.predict(te)
            pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b),
                     "relations": []} for d, b in zip(te, pred_bio)]
            rc = NeuralRelationClassifier(schema, vocab, entity_types(schema),
                                          relation_types(schema)).fit(tr, dev=dv, epochs=40, patience=6)
            for p, rp in zip(pred, rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
            name = "Tier2_BiLSTM_Neural"
            _collect(name, te_docs, pred, gold_bio, pred_bio, etypes,
                     metrics, ent_tables, rel_tables, confusions)
            fig_training_curve(getattr(tagger, "history", None),
                               f"{FIGDIR}/training_curve_Tier2_BiLSTM_NER.png", "Tier 2 BiLSTM-CRF · NER")
            fig_training_curve(getattr(rc, "history", None),
                               f"{FIGDIR}/training_curve_Tier2_Neural_RE.png", "Tier 2 Neural · RE")
            if args.save_models:
                tagger.save(f"{MODELDIR}/tier2_bilstm.pt")
                rc.save(f"{MODELDIR}/tier2_neural_re.pt")
                print(f"  [saved] {MODELDIR}/tier2_bilstm.pt (+ neural_re)")
        except ImportError as e:
            print(f"[Tier2] skipped (pip install torch pytorch-crf gensim): {e}")

    # ---- Tier 3: Transformer encoder (BERT NER + span-pooling RE) ----
    if "3" in args.tiers:
        _bootstrap.banner("TIER 3 / 4  ·  Transformer (DistilBERT)  (training, GPU)")
        try:
            from src.models.transformer_ie import TransformerNER, TransformerRE
            cfg = schema.get("models", {}).get("transformer", {})
            enc = cfg.get("encoder", "distilbert-base-uncased")
            nc, rcf = cfg.get("ner", {}), cfg.get("re", {})
            _bootstrap.step(f"encoder = {enc}")
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
            pred_bio = ner.predict(te)
            pred = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], b),
                     "relations": []} for d, b in zip(te, pred_bio)]
            re = TransformerRE(schema, entity_types(schema), relation_types(schema), model_name=enc,
                               max_len=cfg.get("max_len", 128)).fit(
                tr, dev=dv, epochs=rcf.get("epochs", 10), lr=float(rcf.get("lr", 3e-5)),
                batch_size=rcf.get("batch_size", 8), patience=rcf.get("patience", 3))
            for p, rp in zip(pred, re.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred])):
                p["relations"] = rp
            _collect("Tier3_Transformer", te_docs, pred, gold_bio, pred_bio, etypes,
                     metrics, ent_tables, rel_tables, confusions)
            fig_training_curve(getattr(ner, "history", None),
                               f"{FIGDIR}/training_curve_Tier3_Transformer_NER.png", "Tier 3 Transformer · NER")
            fig_training_curve(getattr(re, "history", None),
                               f"{FIGDIR}/training_curve_Tier3_Transformer_RE.png", "Tier 3 Transformer · RE")
            if args.save_models:
                ner.save(f"{MODELDIR}/tier3_transformer_ner")
                re.save(f"{MODELDIR}/tier3_transformer_re")
                print(f"  [saved] {MODELDIR}/tier3_transformer_ner (+ _re)")
        except ImportError as e:
            print(f"[Tier3] skipped (pip install transformers): {e}")

    # ---- External models (SpERT, REBEL, ...): score imported predictions on the
    #      SAME frozen test set so every model lands in one comparison table/figure ----
    externals = []
    if args.spert:
        externals.append(("Tier3b_SpERT", args.spert))
    for item in args.external:                        # NAME=path.json
        nm, _, pth = item.partition("=")
        externals.append((nm or os.path.basename(pth), pth))

    for name, path in externals:
        if not path or not os.path.exists(path):
            print(f"[external] {name}: file not found ({path}) — skipped")
            continue
        _bootstrap.banner(f"EXTERNAL  ·  {name}  (imported predictions, scored here)")
        sp = json.load(open(path, encoding="utf-8"))   # [{tokens,entities,relations}]
        es = entity_scores(te_docs, sp); es["ci"] = bootstrap_ci(te_docs, sp, "entity", N_BOOT)
        rs = relation_scores(te_docs, sp); rs["ci"] = bootstrap_ci(te_docs, sp, "relation", N_BOOT)
        metrics[name] = {"entity": es, "relation": rs}
        ent_tables[name] = per_class_table(te_docs, sp, "entity")
        rel_tables[name] = per_class_table(te_docs, sp, "relation")
        _bootstrap.step(f"{name}: entity F1={es['micro_f1']}  relation F1={rs['micro_f1']}")

    # ---- write tables ----
    for name in ent_tables:
        ent_tables[name].to_csv(f"{TABDIR}/entity_per_class_{name}.csv", index=False)
        rel_tables[name].to_csv(f"{TABDIR}/relation_per_class_{name}.csv", index=False)
    def _ci(m, k):
        return m[k].get("ci", {"lo": None, "hi": None})
    overall = pd.DataFrame([{
        "model": n, "entity_micro_f1": m["entity"]["micro_f1"],
        "entity_ci_lo": _ci(m, "entity")["lo"], "entity_ci_hi": _ci(m, "entity")["hi"],
        "entity_macro_f1": m["entity"]["macro_f1"],
        "entity_micro_p": m["entity"]["micro_p"], "entity_micro_r": m["entity"]["micro_r"],
        "relation_micro_f1": m["relation"]["micro_f1"],
        "relation_ci_lo": _ci(m, "relation")["lo"], "relation_ci_hi": _ci(m, "relation")["hi"],
        "relation_macro_f1": m["relation"]["macro_f1"],
        "relation_gold_entity_micro_f1": m.get("relation_gold_entities", {}).get("micro_f1"),
    } for n, m in metrics.items()])
    overall.to_csv(f"{TABDIR}/overall_metrics.csv", index=False)
    if diagnostics:
        pd.DataFrame([diagnostics]).to_csv(f"{TABDIR}/validity_diagnostics.csv", index=False)

    # corpus support (whole gold, not just test)
    ent_sup = Counter(e["type"] for r in gold for e in r.get("entities", []))
    rel_sup = Counter(rr["type"] for r in gold for rr in r.get("relations", []))
    pd.DataFrame([{"class": k, "count": v} for k, v in ent_sup.most_common()]
                 ).to_csv(f"{TABDIR}/corpus_support_entities.csv", index=False)
    pd.DataFrame([{"class": k, "count": v} for k, v in rel_sup.most_common()]
                 ).to_csv(f"{TABDIR}/corpus_support_relations.csv", index=False)

    metrics_payload = {"split": split_note, "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
                       "selection": selection_meta, "metrics": metrics}
    metrics_versioned, metrics_run_alias, metrics_compatibility = _write_protected_metrics(
        metrics_payload, args.run_id
    )

    # ---- figures ----
    if metrics:
        fig_overall(metrics, f"{FIGDIR}/overall_comparison.png")
    if ent_tables:
        fig_per_class(ent_tables, "entity", f"{FIGDIR}/entity_per_class_f1.png")
    if rel_tables:
        fig_per_class(rel_tables, "relation", f"{FIGDIR}/relation_per_class_f1.png")
    for name, (labels, M) in confusions.items():
        fig_confusion(labels, M, f"Entity confusion — {name}", f"{FIGDIR}/confusion_{name}.png")
    fig_support(dict(ent_sup), "Entity support (full gold corpus)", f"{FIGDIR}/support_entities.png", "#0072B2")
    fig_support(dict(rel_sup), "Relation support (full gold corpus)", f"{FIGDIR}/support_relations.png", "#E69F00")

    # ---- learning-curve trajectory (append this run, then plot) ----
    traj_path = "outputs/reports/f1_trajectory.csv"
    rows = [{"run_id": args.run_id, "n_train": len(tr), "tier": n,
             "ent_micro_f1": m["entity"]["micro_f1"],
             "ent_micro_f1_ci_lo": _ci(m, "entity")["lo"], "ent_micro_f1_ci_hi": _ci(m, "entity")["hi"],
             "rel_micro_f1": m["relation"]["micro_f1"],
             "rel_micro_f1_ci_lo": _ci(m, "relation")["lo"], "rel_micro_f1_ci_hi": _ci(m, "relation")["hi"]}
            for n, m in metrics.items()]
    traj = pd.DataFrame(rows)
    if os.path.exists(traj_path):
        traj = pd.concat([pd.read_csv(traj_path), traj], ignore_index=True)
    traj = traj.drop_duplicates(subset=["n_train", "tier"], keep="last")
    traj.to_csv(traj_path, index=False)
    if traj["n_train"].nunique() >= 2:
        fig_learning_curve(traj, f"{FIGDIR}/learning_curve.png")
        print("learning curve updated (>=2 corpus sizes).")
    else:
        print("learning curve: needs >=2 rounds; will populate as you add gold.")

    # ---- versioned model card + registry (only when models were saved) ----
    if args.save_models and version:
        card = {"version": version, "run_id": args.run_id, "timestamp": version.split("__", 1)[1],
                "tuned": bool(args.tune), "encoder": schema.get("models", {}).get("transformer", {}).get("encoder"),
                "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
                "train_batches": args.train_batches or "all",
                "tiers": list(metrics),
                "dev_selected_hyperparameters": selection_meta,
                "metrics": {n: {"entity_micro_f1": m["entity"]["micro_f1"], "entity_ci": _ci(m, "entity"),
                                "relation_micro_f1": m["relation"]["micro_f1"], "relation_ci": _ci(m, "relation")}
                            for n, m in metrics.items()},
                "diagnostics": diagnostics}
        json.dump(card, open(f"{MODELDIR}/model_card.json", "w"), indent=2, default=str)
        reg_path = "outputs/models/registry.csv"
        reg = pd.DataFrame([{"version": version, "timestamp": version.split("__", 1)[1],
                             "run_id": args.run_id, "tuned": bool(args.tune), "n_train": len(tr),
                             "tier": n, "entity_micro_f1": m["entity"]["micro_f1"],
                             "relation_micro_f1": m["relation"]["micro_f1"]}
                            for n, m in metrics.items()])
        if os.path.exists(reg_path):
            reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
        reg.to_csv(reg_path, index=False)
        open("outputs/models/latest.txt", "w").write(version + "\n")
        print(f"[save] {len(metrics)} tier model(s) -> {MODELDIR}/  "
              f"(model_card.json written, registry.csv + latest.txt updated)")

    # ---- final terminal summary: the four-way comparison at a glance ----
    _bootstrap.banner("RESULTS  ·  all models on the frozen test set")
    print(f"  {'MODEL':<24}{'ENTITY F1':>12}{'RELATION F1':>14}{'RE F1 (GOLD ENT)':>18}")
    print(f"  {'-'*24}{'-'*12:>12}{'-'*14:>14}{'-'*18:>18}")
    for n, m in metrics.items():
        gold_re = m.get("relation_gold_entities", {}).get("micro_f1")
        gold_re_text = f"{gold_re:.4f}" if gold_re is not None else "-"
        print(f"  {n:<24}{m['entity']['micro_f1']:>12.4f}{m['relation']['micro_f1']:>14.4f}{gold_re_text:>18}")
    print(f"\nSaved tables -> {TABDIR}/  figures -> {FIGDIR}/")
    print(f"Metrics versioned -> {metrics_versioned}")
    print(f"Metrics stable alias -> {metrics_run_alias}")
    print(f"Metrics compatibility alias -> {metrics_compatibility}")


def _collect(name, gold_docs, pred, gold_bio, pred_bio, etypes,
             metrics, ent_tables, rel_tables, confusions):
    es = entity_scores(gold_docs, pred); es["ci"] = bootstrap_ci(gold_docs, pred, "entity", N_BOOT)
    rs = relation_scores(gold_docs, pred); rs["ci"] = bootstrap_ci(gold_docs, pred, "relation", N_BOOT)
    metrics[name] = {"entity": es, "relation": rs}
    ent_tables[name] = per_class_table(gold_docs, pred, "entity")
    rel_tables[name] = per_class_table(gold_docs, pred, "relation")
    confusions[name] = confusion(gold_bio, pred_bio, etypes)
    print(f"[{name}] entity F1={es['micro_f1']} [{es['ci']['lo']}–{es['ci']['hi']}] | "
          f"relation F1={rs['micro_f1']} [{rs['ci']['lo']}–{rs['ci']['hi']}]")


if __name__ == "__main__":
    main()
