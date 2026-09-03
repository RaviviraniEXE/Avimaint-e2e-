"""Step 5 - train IE tiers on TRAIN, select on DEV, evaluate on frozen TEST.

Examples:
  python scripts/05_train_eval.py --tiers 1 --tune --require-frozen-split
  python scripts/05_train_eval.py --tiers 1 2

The classical path now exposes live, truthful progress: which model is fitting,
which DEV configuration is active, completed/total configurations, elapsed time,
ETA based on completed fits, DEV F1, and best-so-far selection. Machine-readable
training traces and manifests are saved beside the normal result files.
"""
import _bootstrap  # noqa: F401
import argparse
import gc
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import entity_scores, relation_scores
from src.models.crf_ner import CRFTagger, bio_to_entities
from src.models.relation_logreg import RelationClassifier
from src.progress import trace_event
from src.schema import bio_tags, entity_types, load_schema, relation_types


def _goldd(recs):
    return [
        {
            "tokens": d["tokens"],
            "entities": bio_to_entities(d["tokens"], d["bio"]),
            "relations": d.get("relations", []),
        }
        for d in recs
    ]


def _relation_gold_docs(recs):
    """Gold relation docs using the original gold entity list/order.

    Relation indices in the annotation point into ``record['entities']``.  This
    representation is therefore the correct target for gold-entity RE scoring.
    """
    return [
        {
            "tokens": d["tokens"],
            "entities": d.get("entities", bio_to_entities(d["tokens"], d["bio"])),
            "relations": d.get("relations", []),
        }
        for d in recs
    ]


def _gold_entity_relation_score(model, recs):
    if model is None:
        return relation_scores(_relation_gold_docs(recs), _relation_gold_docs([]))
    inputs = [
        {
            "tokens": d["tokens"],
            "entities": d.get("entities", bio_to_entities(d["tokens"], d["bio"])),
        }
        for d in recs
    ]
    rels = model.predict(inputs)
    pred = [
        {"tokens": d["tokens"], "entities": inp["entities"], "relations": rp}
        for d, inp, rp in zip(recs, inputs, rels)
    ]
    return relation_scores(_relation_gold_docs(recs), pred)


def _schema_label(schema_path: str, schema: dict) -> str:
    return (
        f"{'CORE' if 'core' in schema_path.lower() else 'FULL'} "
        f"({len(entity_types(schema))} entities / {len(relation_types(schema))} relations)"
    )


def _safe_history(selection: dict | None) -> list[dict]:
    if not selection:
        return []
    return list(selection.get("history", []))



def _safe_run_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _protect_existing_generic(json_path: str, report_name: str) -> None:
    """Archive the current compatibility result before it can be overwritten."""
    src = Path(json_path)
    if not src.exists():
        return
    history_dir = Path("outputs/reports/result_history")
    history_dir.mkdir(parents=True, exist_ok=True)
    old_run = "unknown_previous_run"
    old_stamp = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest_manifest = Path(f"outputs/reports/{report_name}_latest_training_manifest.json")
    if latest_manifest.exists():
        try:
            old_manifest = json.loads(latest_manifest.read_text(encoding="utf-8"))
            old_run = _safe_run_slug(old_manifest.get("run_id", old_run))
            created = old_manifest.get("created_utc")
            if created:
                old_stamp = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            pass
    dst = history_dir / f"{report_name}__{old_run}__{old_stamp}__preoverwrite.json"
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"[result safety] archived previous generic result -> {dst}", flush=True)


def _write_result_copies(payload: dict, report_name: str, run_id: str, stamp: str) -> tuple[str, str, str]:
    """Write versioned + run-specific + legacy generic result files."""
    reports = Path("outputs/reports")
    history = reports / "result_history"
    history.mkdir(parents=True, exist_ok=True)
    run_slug = _safe_run_slug(run_id)
    versioned = history / f"{report_name}__{run_slug}__{stamp}.json"
    run_alias = reports / f"{report_name}__{run_slug}.json"
    generic = reports / f"{report_name}.json"
    text = json.dumps(payload, indent=2) + "\n"
    versioned.write_text(text, encoding="utf-8")
    run_alias.write_text(text, encoding="utf-8")
    generic.write_text(text, encoding="utf-8")
    print(f"[result safety] versioned result -> {versioned}", flush=True)
    print(f"[result safety] stable run alias -> {run_alias}", flush=True)
    print(f"[result safety] compatibility alias -> {generic}", flush=True)
    return str(versioned), str(run_alias), str(generic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["1"], choices=["1", "2", "3"])
    ap.add_argument(
        "--tune",
        action="store_true",
        help="grid-search hyperparameters on DEV (recommended for final numbers)",
    )
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--gold-glob", default="outputs/gold/*.jsonl")
    ap.add_argument("--schema-path", default="config/schema.yaml")
    ap.add_argument("--report-name", default="ie_results")
    ap.add_argument(
        "--require-frozen-split",
        action="store_true",
        help="fail instead of silently creating an ad-hoc split; use for thesis runs",
    )
    args = ap.parse_args()

    os.makedirs("outputs/reports/training_logs", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = Path(
        f"outputs/reports/training_logs/{args.report_name}__{args.run_id}__{stamp}.jsonl"
    )
    os.environ["AVIMAINT_TRAIN_TRACE"] = str(trace_path)
    run_started = time.perf_counter()

    schema = load_schema(args.schema_path)
    schema_label = _schema_label(args.schema_path, schema)
    gold = load_gold(args.gold_glob)
    if not gold:
        raise SystemExit(f"No gold found for: {args.gold_glob}")

    split = load_splits()
    if split:
        tr, dv, te = assign(gold)
        split_note = "FROZEN"
    else:
        if args.require_frozen_split:
            raise SystemExit(
                "Frozen split is required for this thesis run. Run scripts\\ie\\02_freeze_split.bat first."
            )
        tr, dv, te = grouped_split(gold, seed=schema["annotation"]["seed"])
        split_note = "GROUPED FALLBACK (NOT THESIS-FINAL)"

    if not te:
        raise SystemExit("Empty test set.")
    if args.tune and not dv:
        raise SystemExit("--tune requested but development split is empty.")

    tune = schema.get("models", {}).get("tuning", {})
    do_tune = bool(args.tune and dv)
    seed = int(schema.get("annotation", {}).get("seed", 42))

    _bootstrap.banner("AVIATION IE TRAINING - LIVE RUN CONTEXT")
    _bootstrap.step(f"run_id={args.run_id} | report={args.report_name}")
    _bootstrap.step(f"schema={schema_label} | file={args.schema_path}")
    _bootstrap.step(f"gold={args.gold_glob} | records={len(gold)}")
    _bootstrap.step(
        f"split={split_note}: TRAIN={len(tr)} DEV={len(dv)} TEST={len(te)} | seed={seed}"
    )
    _bootstrap.step("selection policy=DEV only; TEST is evaluated only after model selection")
    if args.tiers == ["1"]:
        _bootstrap.step("compute=CPU (CRF L-BFGS + balanced logistic regression)")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                _bootstrap.step(f"compute=CUDA GPU | {gpu_name} | torch={torch.__version__}")
            else:
                _bootstrap.step(f"compute=CPU | torch={torch.__version__} | CUDA unavailable")
        except Exception as exc:
            _bootstrap.step(f"compute=unknown (torch probe failed: {exc})")
    _bootstrap.step(f"live trace={trace_path}")

    trace_event(
        "run_start",
        run_id=args.run_id,
        report_name=args.report_name,
        tiers=args.tiers,
        tuned=do_tune,
        schema_path=args.schema_path,
        schema=schema_label,
        gold_glob=args.gold_glob,
        n_gold=len(gold),
        n_train=len(tr),
        n_dev=len(dv),
        n_test=len(te),
        split=split_note,
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
    )

    goldd = _goldd(te)
    rows: list[dict] = []
    selections: dict[str, dict] = {}

    if "1" in args.tiers:
        crf_grid = [
            (a, b)
            for a in tune.get("crf", {}).get("c1", [0.1])
            for b in tune.get("crf", {}).get("c2", [0.1])
        ]
        lr_grid = [float(v) for v in tune.get("logreg", {}).get("C", [1.0])]

        _bootstrap.banner("TIER 1A - CRF NER")
        _bootstrap.step(
            f"training={len(tr)} records | DEV={len(dv)} | "
            f"grid={len(crf_grid)} c1/c2 configurations"
        )
        _bootstrap.step("current objective=select highest strict DEV entity micro-F1")
        phase_started = time.perf_counter()
        if do_tune:
            crf, crf_selection = CRFTagger.tuned(tr, dv, crf_grid)
        else:
            crf = CRFTagger().fit(tr)
            crf_selection = {"c1": 0.1, "c2": 0.1, "micro_f1": None, "history": []}
        crf_seconds = time.perf_counter() - phase_started
        selections["crf"] = crf_selection
        _bootstrap.step(
            f"CRF selected: c1={crf_selection['c1']} c2={crf_selection['c2']} "
            f"DEV F1={crf_selection.get('micro_f1')} | {crf_seconds:.1f}s"
        )

        _bootstrap.banner("TIER 1A - FROZEN TEST NER EVALUATION")
        trace_event("test_evaluation_start", component="CRF_NER", n_test=len(te))
        pred_bio = crf.predict(te)
        pred = [
            {
                "tokens": d["tokens"],
                "entities": bio_to_entities(d["tokens"], b),
                "relations": [],
            }
            for d, b in zip(te, pred_bio)
        ]
        es = entity_scores(goldd, pred)
        _bootstrap.step(
            f"CRF TEST entity micro-F1={es['micro_f1']:.4f} | macro-F1={es['macro_f1']:.4f}"
        )

        _bootstrap.banner("TIER 1B - LOGISTIC-REGRESSION RELATION EXTRACTION")
        if any(r.get("relations") for r in tr):
            _bootstrap.step(
                f"training={len(tr)} records | candidate pairs constrained by schema | "
                f"class_weight=balanced | random_state=42"
            )
            _bootstrap.step(
                f"grid={len(lr_grid)} C configurations | selection=DEV gold-entity relation micro-F1"
            )
            phase_started = time.perf_counter()
            if do_tune:
                rc, rc_selection = RelationClassifier.tuned(schema, tr, dv, lr_grid)
            else:
                rc = RelationClassifier(schema).fit(tr)
                rc_selection = {
                    "C": 1.0,
                    "micro_f1": None,
                    "history": [],
                    "training_stats": getattr(rc, "fit_stats", {}),
                }
            rc_seconds = time.perf_counter() - phase_started
            selections["logreg"] = rc_selection
            stats = rc_selection.get("training_stats", getattr(rc, "fit_stats", {}))
            if stats:
                _bootstrap.step(
                    "RE candidates: "
                    f"{stats.get('candidate_pairs')} total = "
                    f"{stats.get('positive_pairs')} positive + {stats.get('negative_pairs')} negative; "
                    f"policy={stats.get('negative_policy')}"
                )
            _bootstrap.step(
                f"LogReg selected: C={rc_selection['C']} DEV F1={rc_selection.get('micro_f1')} "
                f"| {rc_seconds:.1f}s"
            )
        else:
            rc = None
            rc_selection = {}
            _bootstrap.step("No training relations found; relation model skipped.")

        _bootstrap.banner("TIER 1B - FROZEN TEST RELATION EVALUATION")
        trace_event("test_evaluation_start", component="LogReg_RE", n_test=len(te))
        if rc:
            # Strict end-to-end RE: relations are predicted over CRF-predicted entities.
            end_to_end_inputs = [
                {"tokens": p["tokens"], "entities": p["entities"]} for p in pred
            ]
            for p, rp in zip(pred, rc.predict(end_to_end_inputs)):
                p["relations"] = rp
        rs = relation_scores(goldd, pred)

        # Gold-entity RE separates relation-classification error from upstream NER error.
        if rc:
            gold_inputs = [
                {"tokens": d["tokens"], "entities": d["entities"]} for d in te
            ]
            gold_rels = rc.predict(gold_inputs)
            gold_rel_pred = [
                {"tokens": d["tokens"], "entities": d["entities"], "relations": rp}
                for d, rp in zip(te, gold_rels)
            ]
            rs_gold_entities = relation_scores(_relation_gold_docs(te), gold_rel_pred)
        else:
            rs_gold_entities = {
                "micro_p": 0.0,
                "micro_r": 0.0,
                "micro_f1": 0.0,
                "macro_f1": 0.0,
                "per_class": {},
                "support": 0,
            }

        _bootstrap.step(
            f"LogReg TEST strict end-to-end relation micro-F1={rs['micro_f1']:.4f}"
        )
        _bootstrap.step(
            f"LogReg TEST gold-entity relation micro-F1={rs_gold_entities['micro_f1']:.4f} "
            "(diagnostic upper-bound view)"
        )
        print(
            "  entity per-class F1:",
            {k: v[2] for k, v in es["per_class"].items()},
            flush=True,
        )
        rows.append(
            {
                "name": "Tier1_CRF_LogReg",
                "entity": es,
                "relation": rs,
                "relation_gold_entities": rs_gold_entities,
            }
        )

    if "2" in args.tiers:
        try:
            import torch
            from src.models.bilstm_crf import BiLSTMCRF, build_char_vocab
            from src.models.embeddings import load_matrix
            from src.models.relation_bilstm import NeuralRelationClassifier

            _bootstrap.banner("TIER 2A - BiLSTM-CRF NER")
            vocab = BiLSTMCRF.build_vocab(tr)
            char_vocab = build_char_vocab(tr)
            pretrained = load_matrix(vocab, dim=100)
            _bootstrap.step(
                f"device={'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
                f"TRAIN={len(tr)} DEV={len(dv)} TEST={len(te)}"
            )
            _bootstrap.step(
                f"features=word embeddings + char-CNN | pretrained FastText={'ON' if pretrained is not None else 'OFF'}"
            )
            _bootstrap.step(
                "early stopping=DEV loss | default max_epochs=60 | patience=8 | "
                "epoch progress shows train loss, DEV loss, DEV F1 and ETA"
            )
            if do_tune:
                grid = tune.get("bilstm") or {"lr": [1e-3], "dropout": [0.4]}
                n_cfg = len(grid.get("lr", [1e-3])) * len(grid.get("dropout", [0.4]))
                _bootstrap.step(f"DEV hyperparameter selection={n_cfg} BiLSTM configurations")
                tagger, bilstm_selection = BiLSTMCRF.tuned(
                    vocab,
                    bio_tags(schema),
                    tr,
                    dv,
                    grid=grid,
                    pretrained=pretrained,
                    char_vocab=char_vocab,
                )
                selections["bilstm"] = bilstm_selection
            else:
                tagger = BiLSTMCRF(
                    vocab, bio_tags(schema), pretrained=pretrained, char_vocab=char_vocab
                ).fit(tr, dev=dv, epochs=60, patience=8)
                selections["bilstm"] = {
                    "params": dict(tagger.params),
                    "training_summary": getattr(tagger, "training_summary", {}),
                    "selection": "fixed predeclared configuration; early stopping on DEV loss",
                }

            _bootstrap.banner("TIER 2A - FROZEN TEST NER EVALUATION")
            pred = [
                {
                    "tokens": d["tokens"],
                    "entities": bio_to_entities(d["tokens"], b),
                    "relations": [],
                }
                for d, b in zip(te, tagger.predict(te))
            ]
            es = entity_scores(goldd, pred)
            _bootstrap.step(f"BiLSTM TEST entity micro-F1={es['micro_f1']:.4f} | macro-F1={es['macro_f1']:.4f}")

            _bootstrap.banner("TIER 2B - NEURAL RELATION EXTRACTION")
            _bootstrap.step(
                "gold entities are used inside TRAIN/DEV RE fitting; strict TEST score uses BiLSTM-predicted entities"
            )
            _bootstrap.step("early stopping=DEV loss | max_epochs=40 | patience=6 | class-weighted loss")
            rc = NeuralRelationClassifier(
                schema, vocab, entity_types(schema), relation_types(schema)
            ).fit(tr, dev=dv, epochs=40, patience=6)
            selections["bilstm_re"] = {
                "params": dict(getattr(rc, "params", {})),
                "training_summary": getattr(rc, "training_summary", {}),
            }
            for p, rp in zip(
                pred,
                rc.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred]),
            ):
                p["relations"] = rp
            rs = relation_scores(goldd, pred)
            gold_inputs = [{"tokens": d["tokens"], "entities": d["entities"]} for d in te]
            gold_rels = rc.predict(gold_inputs)
            rs_gold_entities = relation_scores(
                _relation_gold_docs(te),
                [
                    {"tokens": d["tokens"], "entities": d["entities"], "relations": rp}
                    for d, rp in zip(te, gold_rels)
                ],
            )
            rows.append(
                {
                    "name": "Tier2_BiLSTM_Neural",
                    "entity": es,
                    "relation": rs,
                    "relation_gold_entities": rs_gold_entities,
                }
            )
            _bootstrap.step(
                f"Tier2 TEST entity F1={es['micro_f1']:.4f} | strict relation F1={rs['micro_f1']:.4f} "
                f"| gold-entity relation F1={rs_gold_entities['micro_f1']:.4f}"
            )
            # Free neural Tier-2 models before loading transformer checkpoints on small GPUs.
            try:
                tagger.net.to("cpu"); rc.net.to("cpu")
            except Exception:
                pass
            del tagger, rc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _bootstrap.step("released Tier-2 GPU memory before Tier-3")
        except ImportError as e:
            print(f"[Tier2] skipped (pip install torch pytorch-crf): {e}", flush=True)

    if "3" in args.tiers:
        try:
            import torch
            from src.models.transformer_ie import TransformerNER, TransformerRE

            cfg = schema.get("models", {}).get("transformer", {})
            enc = cfg.get("encoder", "distilbert-base-uncased")
            nc, rcf = cfg.get("ner", {}), cfg.get("re", {})
            _bootstrap.banner("TIER 3A - TRANSFORMER NER")
            _bootstrap.step(
                f"encoder={enc} | device={'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
                f"max_len={cfg.get('max_len', 128)}"
            )
            _bootstrap.step(
                f"early stopping=DEV loss | max_epochs={nc.get('epochs', 10)} | patience={nc.get('patience', 3)} | "
                f"batch={nc.get('batch_size', 16)}"
            )
            if do_tune:
                lr_grid = tune.get("transformer", {}).get("lr", [3e-5])
                _bootstrap.step(f"DEV learning-rate selection={len(lr_grid)} configurations: {lr_grid}")
                ner, transformer_selection = TransformerNER.tuned(
                    bio_tags(schema),
                    tr,
                    dv,
                    model_name=enc,
                    max_len=cfg.get("max_len", 128),
                    lrs=lr_grid,
                    epochs=nc.get("epochs", 10),
                    batch_size=nc.get("batch_size", 16),
                    patience=nc.get("patience", 3),
                )
                selections["transformer_ner"] = transformer_selection
            else:
                ner = TransformerNER(
                    bio_tags(schema), model_name=enc, max_len=cfg.get("max_len", 128)
                ).fit(
                    tr,
                    dev=dv,
                    epochs=nc.get("epochs", 10),
                    lr=float(nc.get("lr", 3e-5)),
                    batch_size=nc.get("batch_size", 16),
                    patience=nc.get("patience", 3),
                )
                selections["transformer_ner"] = {
                    "lr": float(nc.get("lr", 3e-5)),
                    "training_summary": getattr(ner, "training_summary", {}),
                    "selection": "fixed predeclared configuration; early stopping on DEV loss",
                }

            _bootstrap.banner("TIER 3A - FROZEN TEST NER EVALUATION")
            pred = [
                {
                    "tokens": d["tokens"],
                    "entities": bio_to_entities(d["tokens"], b),
                    "relations": [],
                }
                for d, b in zip(te, ner.predict(te))
            ]
            es = entity_scores(goldd, pred)
            _bootstrap.step(f"Transformer TEST entity micro-F1={es['micro_f1']:.4f} | macro-F1={es['macro_f1']:.4f}")
            # NER predictions are now materialized; release the NER transformer before
            # allocating the separate RE transformer (important for 4-GB GPUs).
            try:
                ner.model.to("cpu")
            except Exception:
                pass
            del ner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _bootstrap.step("released Transformer NER GPU memory before Transformer RE")

            _bootstrap.banner("TIER 3B - TRANSFORMER RELATION EXTRACTION")
            _bootstrap.step(
                f"encoder={enc} | early stopping=DEV loss | max_epochs={rcf.get('epochs', 10)} | "
                f"patience={rcf.get('patience', 3)} | batch={rcf.get('batch_size', 8)}"
            )
            re = TransformerRE(
                schema,
                entity_types(schema),
                relation_types(schema),
                model_name=enc,
                max_len=cfg.get("max_len", 128),
            ).fit(
                tr,
                dev=dv,
                epochs=rcf.get("epochs", 10),
                lr=float(rcf.get("lr", 3e-5)),
                batch_size=rcf.get("batch_size", 8),
                patience=rcf.get("patience", 3),
            )
            selections["transformer_re"] = {
                "lr": float(rcf.get("lr", 3e-5)),
                "training_summary": getattr(re, "training_summary", {}),
            }
            for p, rp in zip(
                pred,
                re.predict([{"tokens": p["tokens"], "entities": p["entities"]} for p in pred]),
            ):
                p["relations"] = rp
            rs = relation_scores(goldd, pred)
            gold_inputs = [{"tokens": d["tokens"], "entities": d["entities"]} for d in te]
            gold_rels = re.predict(gold_inputs)
            rs_gold_entities = relation_scores(
                _relation_gold_docs(te),
                [
                    {"tokens": d["tokens"], "entities": d["entities"], "relations": rp}
                    for d, rp in zip(te, gold_rels)
                ],
            )
            rows.append(
                {
                    "name": "Tier3_Transformer",
                    "entity": es,
                    "relation": rs,
                    "relation_gold_entities": rs_gold_entities,
                }
            )
            _bootstrap.step(
                f"Tier3 TEST entity F1={es['micro_f1']:.4f} | strict relation F1={rs['micro_f1']:.4f} "
                f"| gold-entity relation F1={rs_gold_entities['micro_f1']:.4f}"
            )
            try:
                re.model.to("cpu")
            except Exception:
                pass
            del re
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError as e:
            print(f"[Tier3] skipped (pip install transformers): {e}", flush=True)

    os.makedirs("outputs/reports", exist_ok=True)
    json_path = f"outputs/reports/{args.report_name}.json"
    _protect_existing_generic(json_path, args.report_name)
    result_payload = {
        row["name"]: {
            "entity": row["entity"],
            "relation": row["relation"],
            "relation_gold_entities": row["relation_gold_entities"],
        }
        for row in rows
    }
    versioned_result, run_alias_result, compatibility_result = _write_result_copies(
        result_payload, args.report_name, args.run_id, stamp
    )

    log = pd.DataFrame(
        [
            {
                "run_id": args.run_id,
                "tier": row["name"],
                "n_test": len(te),
                "ent_micro_f1": row["entity"]["micro_f1"],
                "ent_macro_f1": row["entity"]["macro_f1"],
                "rel_micro_f1": row["relation"]["micro_f1"],
                "rel_macro_f1": row["relation"]["macro_f1"],
                "rel_gold_entity_micro_f1": row["relation_gold_entities"]["micro_f1"],
            }
            for row in rows
        ]
    )
    log_path = f"outputs/reports/{args.report_name}_log.csv"
    if os.path.exists(log_path):
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log.to_csv(log_path, index=False)

    elapsed = time.perf_counter() - run_started
    manifest = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "schema_path": args.schema_path,
        "schema": schema_label,
        "gold_glob": args.gold_glob,
        "split": split_note,
        "seed": seed,
        "n_gold": len(gold),
        "n_train": len(tr),
        "n_dev": len(dv),
        "n_test": len(te),
        "test_policy": "DEV-only model selection; frozen TEST evaluated after selection",
        "tiers": args.tiers,
        "tuned": do_tune,
        "selections": selections,
        "results_file": versioned_result,
        "results_run_alias": run_alias_result,
        "results_compatibility_alias": compatibility_result,
        "log_file": log_path,
        "trace_file": str(trace_path),
        "elapsed_seconds": round(elapsed, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    manifest_path = Path(
        f"outputs/reports/training_logs/{args.report_name}__{args.run_id}__{stamp}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    latest_manifest = Path(f"outputs/reports/{args.report_name}_latest_training_manifest.json")
    latest_manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    run_manifest_alias = Path(
        f"outputs/reports/{args.report_name}__{_safe_run_slug(args.run_id)}_latest_training_manifest.json"
    )
    run_manifest_alias.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    trace_event("run_complete", elapsed_seconds=round(elapsed, 3), results_file=versioned_result)

    _bootstrap.banner("IE RUN COMPLETE")
    for row in rows:
        _bootstrap.step(
            f"{row['name']}: entity F1={row['entity']['micro_f1']:.4f} | "
            f"strict relation F1={row['relation']['micro_f1']:.4f} | "
            f"gold-entity relation F1={row['relation_gold_entities']['micro_f1']:.4f}"
        )
    _bootstrap.step(f"results versioned={versioned_result}")
    _bootstrap.step(f"results stable alias={run_alias_result}")
    _bootstrap.step(f"results compatibility alias={compatibility_result}")
    _bootstrap.step(f"append-only metrics log={log_path}")
    _bootstrap.step(f"training manifest={manifest_path}")
    _bootstrap.step(f"live trace={trace_path}")
    _bootstrap.step(f"total elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
