"""Build the final full-schema IE comparison from EXISTING results only.

This script never fits a model. It recovers already-computed Tier 1/2/3 result
artifacts, scores imported SpERT predictions on the frozen TEST set, and writes a
combined comparison without touching the trained checkpoints.

It also repairs the historical generic-result overwrite problem where
``outputs/reports/ie_results.json`` was reused by classical and neural runs.
Whenever enough evidence still exists (for example ``metrics_full.json`` from
Tier 1 plus the current neural ``ie_results.json``), stable recovered aliases are
written automatically.
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.data.gold import grouped_split, load_gold
from src.data.split import assign, load_splits
from src.evaluate import bootstrap_ci, entity_scores, relation_scores
from src.models.crf_ner import bio_to_entities
from src.schema import load_schema

EXPECTED = (
    "Tier1_CRF_LogReg",
    "Tier2_BiLSTM_Neural",
    "Tier3_Transformer",
)


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_metrics(payload):
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("metrics"), dict):
        payload = payload["metrics"]
    return {
        key: value
        for key, value in payload.items()
        if key in EXPECTED and isinstance(value, dict)
    }


def _candidate_paths():
    # Stable aliases created by the overwrite-protection patch are preferred.
    ordered = [
        "outputs/reports/ie_results__aviation_tier1.json",
        "outputs/reports/ie_results__aviation_neural.json",
    ]
    # Versioned snapshots created by the patch.
    ordered += sorted(
        glob.glob("outputs/reports/result_history/ie_results__*.json"),
        key=os.path.getmtime,
        reverse=True,
    )
    # Recovery locations suggested during the thesis workflow.
    ordered += [
        "outputs/reports/ie_results_full_classical.json",
        "outputs/reports/ie_results_full_neural.json",
        "outputs/reports/ie_results_full_classical_recovered.json",
        "outputs/reports/ie_results_full_neural_recovered.json",
        "outputs/reports/archive_full_classical/metrics_full.json",
    ]
    # Legacy generic files. metrics_full.json is especially useful for recovering
    # the Tier-1 file that was overwritten by the later neural compact run.
    ordered += [
        "outputs/reports/ie_results.json",
        "outputs/reports/metrics_full.json",
    ]
    # De-duplicate without changing priority.
    seen, result = set(), []
    for path in ordered:
        if path not in seen and os.path.exists(path):
            seen.add(path)
            result.append(path)
    return result


def _recover_existing_metrics():
    found, provenance = {}, {}
    for path in _candidate_paths():
        metrics = _extract_metrics(_load_json(path))
        for name in EXPECTED:
            if name not in found and name in metrics:
                found[name] = metrics[name]
                provenance[name] = path.replace("\\", "/")
        if all(name in found for name in EXPECTED):
            break
    return found, provenance


def _gold_docs(records):
    return [
        {
            "tokens": d["tokens"],
            "entities": bio_to_entities(d["tokens"], d["bio"]),
            "relations": d.get("relations", []),
        }
        for d in records
    ]


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_recovered_aliases(metrics, provenance):
    reports = Path("outputs/reports")
    if "Tier1_CRF_LogReg" in metrics:
        _write_json(
            reports / "ie_results_full_classical_recovered.json",
            {"Tier1_CRF_LogReg": metrics["Tier1_CRF_LogReg"]},
        )
    neural = {k: metrics[k] for k in ("Tier2_BiLSTM_Neural", "Tier3_Transformer") if k in metrics}
    if neural:
        _write_json(reports / "ie_results_full_neural_recovered.json", neural)
    _write_json(
        reports / "ie_results_recovery_provenance.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sources": provenance,
            "note": "Recovered from existing artifacts only; no model was trained.",
        },
    )


def _per_class_rows(metric, support):
    rows = []
    for label, values in metric.get("per_class", {}).items():
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            continue
        rows.append(
            {
                "class": label,
                "precision": values[0],
                "recall": values[1],
                "f1": values[2],
                "support": support.get(label, 0),
            }
        )
    return rows


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spert", default="outputs/reports/spert_test.json")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap resamples for SpERT only; existing model metrics are reused")
    parser.add_argument("--run-id", default="aviation_existing_all_tiers")
    args = parser.parse_args()

    _bootstrap.banner("REPORT EXISTING FULL-SCHEMA IE RESULTS - NO TRAINING")
    print("  This command will not fit CRF, BiLSTM, DistilBERT, LogReg, or SpERT.")

    schema = load_schema()
    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No full-schema gold found at outputs/gold/*.jsonl")
    tr, dv, te = (assign(gold) if load_splits()
                  else grouped_split(gold, seed=schema["annotation"]["seed"]))
    if not load_splits():
        raise SystemExit("Frozen split missing. Refusing report-only thesis evaluation.")
    print(f"  frozen split: train={len(tr)} dev={len(dv)} test={len(te)}")

    metrics, provenance = _recover_existing_metrics()
    missing = [name for name in EXPECTED if name not in metrics]
    if missing:
        print("\nExisting result discovery could not recover:")
        for name in missing:
            print(f"  - {name}")
        print("\nNo training was started. Available result candidates were:")
        for path in _candidate_paths():
            print(f"  - {path}")
        raise SystemExit(4)

    _write_recovered_aliases(metrics, provenance)
    print("\n[RECOVERED EXISTING RESULTS]")
    for name in EXPECTED:
        print(f"  {name:<24} <- {provenance[name]}")

    if not os.path.exists(args.spert):
        raise SystemExit(f"SpERT imported predictions not found: {args.spert}")
    spert = _load_json(args.spert)
    if not isinstance(spert, list):
        raise SystemExit(f"SpERT file is not a prediction list: {args.spert}")
    if len(spert) != len(te):
        raise SystemExit(f"SpERT prediction count mismatch: {len(spert)} != frozen TEST {len(te)}")

    gold_test = _gold_docs(te)
    es = entity_scores(gold_test, spert)
    rs = relation_scores(gold_test, spert)
    if args.bootstrap > 0:
        es["ci"] = bootstrap_ci(gold_test, spert, "entity", args.bootstrap)
        rs["ci"] = bootstrap_ci(gold_test, spert, "relation", args.bootstrap)
    metrics["Tier3b_SpERT"] = {"entity": es, "relation": rs}
    provenance["Tier3b_SpERT"] = args.spert.replace("\\", "/")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "existing-results-only",
        "training_performed": False,
        "schema": "FULL (9 entities / 11 relations)",
        "split": "FROZEN",
        "n_train": len(tr),
        "n_dev": len(dv),
        "n_test": len(te),
        "source_provenance": provenance,
        "metrics": metrics,
    }
    history = Path("outputs/reports/result_history")
    history.mkdir(parents=True, exist_ok=True)
    versioned = history / f"combined_existing__{args.run_id}__{stamp}.json"
    stable = Path("outputs/reports/ie_results_combined_existing.json")
    _write_json(versioned, payload)
    _write_json(stable, payload)

    ent_support = Counter(e["type"] for d in gold_test for e in d.get("entities", []))
    rel_support = Counter(r["type"] for d in gold_test for r in d.get("relations", []))
    overall = []
    table_dir = Path("outputs/reports/tables_existing")
    for name, model in metrics.items():
        ent = model.get("entity", {})
        rel = model.get("relation", {})
        gold_rel = model.get("relation_gold_entities", {})
        overall.append({
            "model": name,
            "entity_micro_p": ent.get("micro_p"),
            "entity_micro_r": ent.get("micro_r"),
            "entity_micro_f1": ent.get("micro_f1"),
            "entity_macro_f1": ent.get("macro_f1"),
            "relation_micro_p": rel.get("micro_p"),
            "relation_micro_r": rel.get("micro_r"),
            "relation_micro_f1": rel.get("micro_f1"),
            "relation_macro_f1": rel.get("macro_f1"),
            "relation_gold_entity_micro_f1": gold_rel.get("micro_f1"),
            "source": provenance.get(name),
        })
        _write_csv(
            table_dir / f"entity_per_class_{name}.csv",
            _per_class_rows(ent, ent_support),
            ["class", "precision", "recall", "f1", "support"],
        )
        _write_csv(
            table_dir / f"relation_per_class_{name}.csv",
            _per_class_rows(rel, rel_support),
            ["class", "precision", "recall", "f1", "support"],
        )
    _write_csv(
        table_dir / "overall_metrics.csv",
        overall,
        ["model", "entity_micro_p", "entity_micro_r", "entity_micro_f1", "entity_macro_f1",
         "relation_micro_p", "relation_micro_r", "relation_micro_f1", "relation_macro_f1",
         "relation_gold_entity_micro_f1", "source"],
    )

    _bootstrap.banner("EXISTING-RESULT FOUR-MODEL COMPARISON")
    print(f"  {'MODEL':<24}{'ENTITY F1':>12}{'RELATION F1':>14}{'REL MACRO':>12}")
    print(f"  {'-'*24}{'-'*12:>12}{'-'*14:>14}{'-'*12:>12}")
    for name in (*EXPECTED, "Tier3b_SpERT"):
        m = metrics[name]
        print(f"  {name:<24}{m['entity']['micro_f1']:>12.4f}{m['relation']['micro_f1']:>14.4f}{m['relation']['macro_f1']:>12.4f}")
    print("\nNO TRAINING PERFORMED.")
    print(f"Stable combined result : {stable}")
    print(f"Versioned snapshot     : {versioned}")
    print(f"Tables                 : {table_dir}")
    print("Significance/repeated-seed training is intentionally NOT run by this command.")


if __name__ == "__main__":
    main()
