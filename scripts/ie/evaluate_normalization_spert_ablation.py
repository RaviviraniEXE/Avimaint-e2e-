"""Evaluate five normalization variants with the same frozen FULL SpERT model.

Strict relation correctness requires:
- relation type exact match
- head entity type/start/end exact match
- tail entity type/start/end exact match

The script performs a raw-parity gate against the already-frozen full-schema
aviation SpERT result. If raw does not reproduce the frozen baseline closely,
the experiment is marked invalid and exits non-zero.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
ABL = IE / "outputs" / "normalization_spert_ablation"
PREPARED = ABL / "prepared"
PRED_DIR = ABL / "predictions"
LOG_DIR = ABL / "logs"
REPORT_DIR = IE / "outputs" / "reports" / "normalization_spert_ablation"

SYSTEMS = ["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"]

# Authoritative frozen FULL 9x11 aviation SpERT point estimates.
# Used only as a parity/safety gate, never for model selection.
EXPECTED_RAW = {
    "entity_micro_f1": 0.9520,
    "entity_macro_f1": 0.9063,
    "strict_relation_micro_f1": 0.8537,
    "relation_macro_f1": 0.7898,
}
RAW_MICRO_TOL = 0.0010
RAW_MACRO_TOL = 0.0020


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def entity_key(e):
    return (str(e["type"]), int(e["start"]), int(e["end"]))


def relation_key(doc, r):
    entities = doc.get("entities", [])
    head = r.get("head")
    tail = r.get("tail")
    if isinstance(head, dict):
        h = entity_key(head)
    else:
        h = entity_key(entities[int(head)])
    if isinstance(tail, dict):
        t = entity_key(tail)
    else:
        t = entity_key(entities[int(tail)])
    return (str(r["type"]), h, t)


def doc_entity_counter(doc, doc_index):
    return Counter((doc_index,) + entity_key(e) for e in doc.get("entities", []))


def doc_relation_counter(doc, doc_index, diagnostics):
    out = Counter()
    for r in doc.get("relations", []):
        try:
            key = relation_key(doc, r)
        except Exception:
            diagnostics["invalid_relation_endpoints"] += 1
            continue
        out[(doc_index,) + key] += 1
    return out


def counter_counts(gold, pred):
    inter = gold & pred
    tp = sum(inter.values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    return tp, fp, fn


def evaluate(gold_docs, pred_docs, indices=None):
    if len(gold_docs) != len(pred_docs):
        raise ValueError(
            f"gold/prediction length mismatch: {len(gold_docs)} vs {len(pred_docs)}"
        )
    if indices is None:
        indices = list(range(len(gold_docs)))

    diag = {"invalid_relation_endpoints": 0}
    gold_e = Counter()
    pred_e = Counter()
    gold_r = Counter()
    pred_r = Counter()

    entity_labels = set()
    relation_labels = set()

    for i in indices:
        g = gold_docs[i]
        p = pred_docs[i]
        gold_e += doc_entity_counter(g, i)
        pred_e += doc_entity_counter(p, i)
        gold_r += doc_relation_counter(g, i, diag)
        pred_r += doc_relation_counter(p, i, diag)
        entity_labels.update(e["type"] for e in g.get("entities", []))
        relation_labels.update(r["type"] for r in g.get("relations", []))

    etp, efp, efn = counter_counts(gold_e, pred_e)
    ep, er, ef1 = f1_from_counts(etp, efp, efn)
    rtp, rfp, rfn = counter_counts(gold_r, pred_r)
    rp, rr, rf1 = f1_from_counts(rtp, rfp, rfn)

    entity_rows = []
    entity_f1s = []
    for label in sorted(entity_labels):
        g = Counter({k: v for k, v in gold_e.items() if k[1] == label})
        p = Counter({k: v for k, v in pred_e.items() if k[1] == label})
        tp, fp, fn = counter_counts(g, p)
        prec, rec, f1 = f1_from_counts(tp, fp, fn)
        support = sum(g.values())
        entity_rows.append(
            {
                "class": label,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        entity_f1s.append(f1)

    relation_rows = []
    relation_f1s = []
    for label in sorted(relation_labels):
        # relation counter key = (doc_index, relation_type, head_entity_key, tail_entity_key)
        g = Counter({k: v for k, v in gold_r.items() if k[1] == label})
        p = Counter({k: v for k, v in pred_r.items() if k[1] == label})
        tp, fp, fn = counter_counts(g, p)
        prec, rec, f1 = f1_from_counts(tp, fp, fn)
        support = sum(g.values())
        relation_rows.append(
            {
                "class": label,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        relation_f1s.append(f1)

    return {
        "n_records": len(indices),
        "entity_precision": ep,
        "entity_recall": er,
        "entity_micro_f1": ef1,
        "entity_macro_f1": (
            sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0
        ),
        "strict_relation_precision": rp,
        "strict_relation_recall": rr,
        "strict_relation_micro_f1": rf1,
        "relation_macro_f1": (
            sum(relation_f1s) / len(relation_f1s) if relation_f1s else 0.0
        ),
        "gold_entity_support": sum(gold_e.values()),
        "gold_relation_support": sum(gold_r.values()),
        "pred_entity_count": sum(pred_e.values()),
        "pred_relation_count": sum(pred_r.values()),
        "entity_per_class": entity_rows,
        "relation_per_class": relation_rows,
        "diagnostics": diag,
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def projection_test_summary(test_ids):
    qc_path = IE / "outputs" / "gold_variants" / "projection_qc.csv"
    if not qc_path.exists():
        return {}
    wanted = set(test_ids)
    rows = defaultdict(list)
    with qc_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row["record_id"]) in wanted:
                rows[row["system"]].append(row)
    out = {}
    for system, vals in rows.items():
        out[system] = {
            "mean_record_entity_coverage": sum(
                float(r["entity_coverage"]) for r in vals
            ) / len(vals),
            "dropped_entities_test": sum(
                int(float(r["dropped_entities"])) for r in vals
            ),
            "dropped_relations_test": sum(
                int(float(r["dropped_relations"])) for r in vals
            ),
        }
    return out


def parse_model_paths():
    found = {}
    rx = re.compile(r"model\s*=\s*(.+?final_model[^\r\n]*)", re.I)
    for system in SYSTEMS:
        path = LOG_DIR / f"{system}.log"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = [m.group(1).strip() for m in rx.finditer(text)]
        if matches:
            found[system] = matches[-1]
    return found


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gold = {}
    pred = {}
    for system in SYSTEMS:
        gpath = PREPARED / f"{system}_test.json"
        ppath = PRED_DIR / f"{system}_predictions.json"
        if not gpath.exists():
            raise SystemExit(f"Missing prepared gold: {gpath}")
        if not ppath.exists():
            raise SystemExit(f"Missing SpERT predictions: {ppath}")
        gold[system] = load_json(gpath)
        pred[system] = load_json(ppath)
        if len(gold[system]) != 225 or len(pred[system]) != 225:
            raise SystemExit(
                f"{system}: expected 225 gold and 225 predictions, found "
                f"{len(gold[system])}/{len(pred[system])}"
            )

    raw_ids = [
        str(d.get("orig_id", d.get("ident", i)))
        for i, d in enumerate(gold["raw"])
    ]
    id_to_index = {rid: i for i, rid in enumerate(raw_ids)}
    common_path = ABL / "common_full_projection_ids.json"
    common_ids = load_json(common_path) if common_path.exists() else raw_ids
    common_indices = [id_to_index[rid] for rid in common_ids if rid in id_to_index]

    projection = projection_test_summary(raw_ids)
    all_metrics = {}
    common_metrics = {}
    overall_rows = []
    common_rows = []
    entity_rows = []
    relation_rows = []

    for system in SYSTEMS:
        m = evaluate(gold[system], pred[system])
        c = evaluate(gold[system], pred[system], common_indices)
        all_metrics[system] = m
        common_metrics[system] = c

        p = projection.get(system, {})
        overall_rows.append(
            {
                "variant": system,
                "entity_micro_f1": m["entity_micro_f1"],
                "entity_macro_f1": m["entity_macro_f1"],
                "strict_relation_micro_f1": m["strict_relation_micro_f1"],
                "relation_macro_f1": m["relation_macro_f1"],
                "n_test": m["n_records"],
                "gold_entities": m["gold_entity_support"],
                "gold_relations": m["gold_relation_support"],
                "projection_mean_record_entity_coverage": p.get(
                    "mean_record_entity_coverage", ""
                ),
                "dropped_entities_test": p.get("dropped_entities_test", ""),
                "dropped_relations_test": p.get("dropped_relations_test", ""),
            }
        )
        common_rows.append(
            {
                "variant": system,
                "entity_micro_f1": c["entity_micro_f1"],
                "entity_macro_f1": c["entity_macro_f1"],
                "strict_relation_micro_f1": c["strict_relation_micro_f1"],
                "relation_macro_f1": c["relation_macro_f1"],
                "n_test": c["n_records"],
                "gold_entities": c["gold_entity_support"],
                "gold_relations": c["gold_relation_support"],
            }
        )
        for row in m["entity_per_class"]:
            entity_rows.append({"variant": system, **row})
        for row in m["relation_per_class"]:
            relation_rows.append({"variant": system, **row})

    raw = all_metrics["raw"]
    parity = {
        "expected": EXPECTED_RAW,
        "observed": {
            k: raw[k] for k in EXPECTED_RAW
        },
        "tolerance_micro": RAW_MICRO_TOL,
        "tolerance_macro": RAW_MACRO_TOL,
        "checks": {},
    }
    parity_ok = True
    for key, expected in EXPECTED_RAW.items():
        observed = raw[key]
        tol = RAW_MICRO_TOL if "micro" in key else RAW_MACRO_TOL
        ok = abs(observed - expected) <= tol
        parity["checks"][key] = {
            "expected": expected,
            "observed": observed,
            "abs_difference": abs(observed - expected),
            "tolerance": tol,
            "pass": ok,
        }
        parity_ok = parity_ok and ok
    parity["pass"] = parity_ok

    raw_row = next(r for r in overall_rows if r["variant"] == "raw")
    for row in overall_rows:
        for metric in [
            "entity_micro_f1",
            "entity_macro_f1",
            "strict_relation_micro_f1",
            "relation_macro_f1",
        ]:
            row[f"delta_vs_raw_{metric}"] = row[metric] - raw_row[metric]

    model_paths = parse_model_paths()
    unique_models = sorted(set(model_paths.values()))
    same_model_ok = len(unique_models) <= 1

    manifest = {
        "status": (
            "complete"
            if parity_ok and same_model_ok
            else "invalid_requires_review"
        ),
        "training_performed": False,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "experiment": "five_way_normalization_to_frozen_full_spert",
        "systems": SYSTEMS,
        "frozen_test_n": 225,
        "common_full_projection_n": len(common_indices),
        "raw_parity_gate": parity,
        "model_paths_parsed_from_logs": model_paths,
        "same_model_path_across_variants": same_model_ok,
        "metrics": {
            system: {
                k: v
                for k, v in all_metrics[system].items()
                if k not in {"entity_per_class", "relation_per_class"}
            }
            for system in SYSTEMS
        },
        "common_full_projection_metrics": {
            system: {
                k: v
                for k, v in common_metrics[system].items()
                if k not in {"entity_per_class", "relation_per_class"}
            }
            for system in SYSTEMS
        },
        "methodology": {
            "model": "same pre-existing frozen FULL 9x11 aviation SpERT",
            "retraining": "none",
            "retuning": "none",
            "raw_condition": "authoritative existing outputs/spert/test.json",
            "normalized_gold": "projected frozen annotations",
            "strict_relation_definition": (
                "relation type plus exact typed head and tail entity spans"
            ),
            "primary_analysis": "all 225 frozen TEST records with projection QC reported",
            "sensitivity_analysis": (
                "records with complete projection in all five normalization systems"
            ),
        },
    }

    write_csv(
        REPORT_DIR / "normalization_spert_ablation.csv",
        overall_rows,
        list(overall_rows[0].keys()),
    )
    write_csv(
        REPORT_DIR / "normalization_spert_ablation_common_projection.csv",
        common_rows,
        list(common_rows[0].keys()),
    )
    write_csv(
        REPORT_DIR / "entity_per_class.csv",
        entity_rows,
        ["variant", "class", "precision", "recall", "f1", "support"],
    )
    write_csv(
        REPORT_DIR / "relation_per_class.csv",
        relation_rows,
        ["variant", "class", "precision", "recall", "f1", "support"],
    )
    dump_json(REPORT_DIR / "FINAL_NORMALIZATION_SPERT_MANIFEST.json", manifest)

    print("=" * 104)
    print("FIVE-WAY NORMALIZATION -> FROZEN FULL SpERT RESULTS")
    print("=" * 104)
    print(
        f"{'variant':18s} {'ent micro':>10s} {'ent macro':>10s} "
        f"{'rel micro':>10s} {'rel macro':>10s} {'dEnt':>6s} {'dRel':>6s}"
    )
    print("-" * 104)
    for row in overall_rows:
        print(
            f"{row['variant']:18s} "
            f"{row['entity_micro_f1']:10.4f} "
            f"{row['entity_macro_f1']:10.4f} "
            f"{row['strict_relation_micro_f1']:10.4f} "
            f"{row['relation_macro_f1']:10.4f} "
            f"{str(row['dropped_entities_test']):>6s} "
            f"{str(row['dropped_relations_test']):>6s}"
        )
    print("-" * 104)
    print(
        "RAW parity gate:",
        "PASS" if parity_ok else "FAIL",
        f"(expected entity micro≈{EXPECTED_RAW['entity_micro_f1']:.4f}, "
        f"strict relation micro≈{EXPECTED_RAW['strict_relation_micro_f1']:.4f})",
    )
    if model_paths:
        print(
            "Same parsed model path across variants:",
            "PASS" if same_model_ok else "FAIL",
        )
        for system, path in model_paths.items():
            print(f"  {system:17s} -> {path}")
    else:
        print(
            "Model-path parse: no 'model=...final_model' line found in logs; "
            "raw parity remains the checkpoint safety gate."
        )
    print(f"common projection sensitivity: {len(common_indices)}/225 records")
    print(f"reports -> {REPORT_DIR}")

    if not parity_ok:
        raise SystemExit(
            "RAW parity FAILED. Do not interpret normalized results. "
            "Send the terminal output and ablation logs for review."
        )
    if not same_model_ok:
        raise SystemExit(
            "Different model paths were detected across variants. "
            "Experiment invalid; do not interpret results."
        )


if __name__ == "__main__":
    main()
