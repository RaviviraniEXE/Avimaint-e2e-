"""Corrected five-way representation-matched normalization -> SpERT evaluation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
BASE = IE / "outputs" / "spert_normalized"
REFERENCE = IE / "outputs" / "spert"
REPORT = IE / "outputs" / "reports" / "normalization_spert_matched_v2"
SYSTEMS = ["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"]
EXPECTED_N = 225


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    z = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, z


def entity_key(e):
    return (str(e["type"]), int(e["start"]), int(e["end"]))


def relation_key(doc, r):
    ents = doc.get("entities", [])
    h = r.get("head")
    t = r.get("tail")
    hk = entity_key(h) if isinstance(h, dict) else entity_key(ents[int(h)])
    tk = entity_key(t) if isinstance(t, dict) else entity_key(ents[int(t)])
    return (str(r["type"]), hk, tk)


def counts(g, p):
    inter = g & p
    return sum(inter.values()), sum((p-g).values()), sum((g-p).values())


def evaluate(gold, pred):
    if len(gold) != len(pred):
        raise SystemExit(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    ge, pe, gr, pr = Counter(), Counter(), Counter(), Counter()
    ent_labels, rel_labels = set(), set()
    invalid_rel = 0

    for i, (gd, pd) in enumerate(zip(gold, pred)):
        for e in gd.get("entities", []):
            ge[(i,) + entity_key(e)] += 1
            ent_labels.add(e["type"])
        for e in pd.get("entities", []):
            pe[(i,) + entity_key(e)] += 1

        for r in gd.get("relations", []):
            try:
                gr[(i,) + relation_key(gd, r)] += 1
                rel_labels.add(r["type"])
            except Exception:
                invalid_rel += 1
        for r in pd.get("relations", []):
            try:
                pr[(i,) + relation_key(pd, r)] += 1
            except Exception:
                invalid_rel += 1

    etp, efp, efn = counts(ge, pe)
    ep, er, ef = f1(etp, efp, efn)
    rtp, rfp, rfn = counts(gr, pr)
    rp, rr, rf = f1(rtp, rfp, rfn)

    ent_rows, ent_f = [], []
    for label in sorted(ent_labels):
        g = Counter({k:v for k,v in ge.items() if k[1] == label})
        p = Counter({k:v for k,v in pe.items() if k[1] == label})
        tp, fp, fn = counts(g, p)
        pp, rrr, ff = f1(tp, fp, fn)
        ent_f.append(ff)
        ent_rows.append({
            "class": label, "precision": pp, "recall": rrr,
            "f1": ff, "support": sum(g.values())
        })

    rel_rows, rel_f = [], []
    for label in sorted(rel_labels):
        g = Counter({k:v for k,v in gr.items() if k[1] == label})
        p = Counter({k:v for k,v in pr.items() if k[1] == label})
        tp, fp, fn = counts(g, p)
        pp, rrr, ff = f1(tp, fp, fn)
        rel_f.append(ff)
        rel_rows.append({
            "class": label, "precision": pp, "recall": rrr,
            "f1": ff, "support": sum(g.values())
        })

    return {
        "entity_precision": ep,
        "entity_recall": er,
        "entity_micro_f1": ef,
        "entity_macro_f1": sum(ent_f)/len(ent_f),
        "strict_relation_precision": rp,
        "strict_relation_recall": rr,
        "strict_relation_micro_f1": rf,
        "relation_macro_f1": sum(rel_f)/len(rel_f),
        "gold_entities": sum(ge.values()),
        "gold_relations": sum(gr.values()),
        "pred_entities": sum(pe.values()),
        "pred_relations": sum(pr.values()),
        "invalid_relation_endpoints": invalid_rel,
        "entity_per_class": ent_rows,
        "relation_per_class": rel_rows,
    }


def ids(docs):
    out = []
    for i, doc in enumerate(docs):
        rid = doc.get("orig_id", doc.get("ident"))
        if rid is None:
            raise SystemExit(f"test document {i} has no orig_id/ident")
        out.append(str(rid))
    return out


def parse_conf(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([^#;\[=][^=]*?)\s*=\s*(.*?)\s*$", line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def newest_model(export: Path):
    models = list((export/"save").rglob("final_model")) if (export/"save").exists() else []
    models = [p for p in models if p.is_dir()]
    if not models:
        raise SystemExit(f"No final_model under {export/'save'}")
    return max(models, key=lambda p: p.stat().st_mtime)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit_path = REPORT / "REPRESENTATION_AUDIT.json"
    prep_path = BASE / "PREP_MANIFEST_V2.json"
    if not audit_path.exists() or not prep_path.exists():
        raise SystemExit("Corrected audit/preparation manifests are missing.")
    audit = load_json(audit_path)
    prep = load_json(prep_path)
    if audit.get("status") != "pass":
        raise SystemExit("Representation audit did not pass.")

    export = {s: BASE/s for s in SYSTEMS}
    gold, pred = {}, {}

    for system in SYSTEMS:
        gp = export[system] / "test.json"
        pp = export[system] / "predictions_test.json"
        if not gp.exists() or not pp.exists():
            raise SystemExit(f"{system}: missing test or prediction artifact")
        gold[system] = load_json(gp)
        pred[system] = load_json(pp)
        if len(gold[system]) != EXPECTED_N or len(pred[system]) != EXPECTED_N:
            raise SystemExit(f"{system}: expected 225 test docs/predictions")

    raw_ids = ids(gold["raw"])
    for system in SYSTEMS[1:]:
        if ids(gold[system]) != raw_ids:
            raise SystemExit(f"{system}: TEST order/membership differs from corrected raw")

    ref_support = (
        sum(len(d.get("entities", [])) for d in gold["raw"]),
        sum(len(d.get("relations", [])) for d in gold["raw"]),
    )
    for system in SYSTEMS[1:]:
        support = (
            sum(len(d.get("entities", [])) for d in gold[system]),
            sum(len(d.get("relations", [])) for d in gold[system]),
        )
        if support != ref_support:
            raise SystemExit(f"{system}: TEST support {support} differs from raw {ref_support}")

    # All five use the identical architecture/hyperparameters.  The historical
    # baseline config is only the fixed configuration source.
    compare_keys = [
        "model_type","model_path","tokenizer_path","train_batch_size",
        "eval_batch_size","neg_entity_count","neg_relation_count","epochs","lr",
        "lr_warmup","weight_decay","max_grad_norm","max_span_size",
        "rel_filter_threshold","size_embedding","prop_drop","final_eval",
        "store_predictions","store_examples","seed",
    ]
    ref_conf = parse_conf(REFERENCE / "avimaint_spert.conf")
    conf_checks = {}
    for system in SYSTEMS:
        cfg = parse_conf(export[system] / "avimaint_spert.conf")
        diffs = {
            key: {"reference": ref_conf.get(key), "variant": cfg.get(key)}
            for key in compare_keys if cfg.get(key) != ref_conf.get(key)
        }
        conf_checks[system] = {
            "same_fixed_hyperparameters": not bool(diffs),
            "differences": diffs,
        }
        if diffs:
            raise SystemExit(f"{system}: fixed hyperparameter mismatch: {diffs}")

    metrics, overall, ent_rows, rel_rows = {}, [], [], []
    for system in SYSTEMS:
        m = evaluate(gold[system], pred[system])
        if m["invalid_relation_endpoints"]:
            raise SystemExit(f"{system}: invalid relation endpoints encountered")
        metrics[system] = m
        overall.append({
            "variant": system,
            "entity_micro_f1": m["entity_micro_f1"],
            "entity_macro_f1": m["entity_macro_f1"],
            "strict_relation_micro_f1": m["strict_relation_micro_f1"],
            "relation_macro_f1": m["relation_macro_f1"],
            "n_test": EXPECTED_N,
            "gold_entities": m["gold_entities"],
            "gold_relations": m["gold_relations"],
        })
        ent_rows.extend({"variant": system, **row} for row in m["entity_per_class"])
        rel_rows.extend({"variant": system, **row} for row in m["relation_per_class"])

    base_row = overall[0]
    for row in overall:
        for key in (
            "entity_micro_f1","entity_macro_f1",
            "strict_relation_micro_f1","relation_macro_f1"
        ):
            row["delta_vs_raw_" + key] = row[key] - base_row[key]

    models = {}
    for system in SYSTEMS:
        model = newest_model(export[system])
        models[system] = {
            "normalization_system": system,
            "export_dir": str(export[system].relative_to(ROOT)),
            "final_model_path": str(model.relative_to(ROOT)),
            "test_prediction_path": str((export[system]/"predictions_test.json").relative_to(ROOT)),
            "config_path": str((export[system]/"avimaint_spert.conf").relative_to(ROOT)),
            "metrics": {
                key: metrics[system][key]
                for key in (
                    "entity_micro_f1","entity_macro_f1",
                    "strict_relation_micro_f1","relation_macro_f1"
                )
            },
        }

    # Historical annotation-representation model retained only as provenance/reference.
    annotation_reference = None
    ref_test = REFERENCE / "test.json"
    ref_pred = REFERENCE / "predictions_test.json"
    if ref_test.exists() and ref_pred.exists():
        rg = load_json(ref_test)
        rp = load_json(ref_pred)
        rm = evaluate(rg, rp)
        annotation_reference = {
            "role": "historical annotation-representation reference; excluded from five-way RQ1 comparison",
            "representation": "legacy normalized annotation representation",
            "entity_micro_f1": rm["entity_micro_f1"],
            "entity_macro_f1": rm["entity_macro_f1"],
            "strict_relation_micro_f1": rm["strict_relation_micro_f1"],
            "relation_macro_f1": rm["relation_macro_f1"],
        }

    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "matched_normalization_spert_ablation_v2.csv", overall)
    write_csv(REPORT / "entity_per_class_v2.csv", ent_rows)
    write_csv(REPORT / "relation_per_class_v2.csv", rel_rows)
    dump_json(REPORT / "MODEL_REGISTRY_V2.json", models)

    manifest = {
        "status": "complete_corrected_no_post_test_retuning_allowed",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "experiment": "corrected_representation_matched_normalization_to_full_spert",
        "systems": SYSTEMS,
        "models_in_main_comparison": 5,
        "new_models_trained_in_correction": ["raw"],
        "models_reused_without_retraining": SYSTEMS[1:],
        "frozen_split": {
            "train": 1275, "dev": 100, "test": 225,
            "membership_source": "outputs/splits.json",
        },
        "schema": {"entities": 9, "relations": 11},
        "same_fixed_hyperparameters": conf_checks,
        "representation_audit": audit,
        "preparation_manifest": prep,
        "metrics": {
            system: {
                key: value for key, value in metrics[system].items()
                if key not in {"entity_per_class", "relation_per_class"}
            }
            for system in SYSTEMS
        },
        "historical_annotation_representation_reference": annotation_reference,
        "model_registry": "MODEL_REGISTRY_V2.json",
        "interpretation_policy": {
            "primary_question": (
                "Does each normalization representation improve or degrade the "
                "strongest IE architecture when train/dev/test representations are matched?"
            ),
            "historical_outputs_spert_not_labeled_raw": True,
            "post_test_hyperparameter_tuning": "forbidden",
            "automatic_deployment_winner_selection": False,
            "old_matched_freeze_status": (
                "superseded for RQ1 because its row labeled raw reused the "
                "legacy normalized annotation-representation model"
            ),
        },
    }
    dump_json(REPORT / "FINAL_MATCHED_NORMALIZATION_SPERT_MANIFEST_V2.json", manifest)

    print("=" * 112)
    print("CORRECTED REPRESENTATION-MATCHED NORMALIZATION -> FULL SpERT RESULTS")
    print("=" * 112)
    print(
        f"{'variant':18s} {'ent micro':>10s} {'ent macro':>10s} "
        f"{'rel micro':>10s} {'rel macro':>10s} {'dEntMicro':>11s} {'dRelMicro':>11s}"
    )
    print("-" * 112)
    for row in overall:
        print(
            f"{row['variant']:18s} "
            f"{row['entity_micro_f1']:10.4f} "
            f"{row['entity_macro_f1']:10.4f} "
            f"{row['strict_relation_micro_f1']:10.4f} "
            f"{row['relation_macro_f1']:10.4f} "
            f"{row['delta_vs_raw_entity_micro_f1']:11.4f} "
            f"{row['delta_vs_raw_strict_relation_micro_f1']:11.4f}"
        )
    print("-" * 112)
    print("Representation audit: PASS (gold == normalized annotation representation 1600/1600)")
    print("Main raw row: NEW true System-A raw model")
    print("Historical outputs/spert baseline: excluded from the five-way RQ1 table")
    print("No post-TEST retuning is allowed.")
    print("Reports ->", REPORT)
    print("Model registry ->", REPORT / "MODEL_REGISTRY_V2.json")


if __name__ == "__main__":
    main()
