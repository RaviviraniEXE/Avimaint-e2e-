"""Generate FINAL aviation IE thesis figures from existing artifacts only.

NO MODEL TRAINING is performed. The script reads the already-frozen core 8x10
and full 9x11 result artifacts, re-scores only the existing core SpERT frozen
TEST predictions, and creates publication-oriented figures/tables in dedicated
``final_figures`` / ``final_tables`` folders.

Historical ``outputs/reports/figures`` and ``outputs/reports/tables`` are never
deleted or overwritten. If this script is run more than once, the previous
stable final-figure/table folders are archived under ``final_figure_history``.

Rare classes are labelled with frozen-TEST support, e.g. ``REFERENCE (n=2)``, so
a perfect F1 from two examples cannot be mistaken for strong statistical
support.
"""
import _bootstrap  # noqa: F401

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.gold import load_gold
from src.evaluate import entity_scores, relation_scores
from src.models.crf_ner import bio_to_entities


MODEL_ORDER = [
    "Tier1_CRF_LogReg",
    "Tier2_BiLSTM_Neural",
    "Tier3_Transformer",
    "Tier3b_SpERT",
]
NICE = {
    "Tier1_CRF_LogReg": "CRF + LogReg",
    "Tier2_BiLSTM_Neural": "BiLSTM-CRF + Neural RE",
    "Tier3_Transformer": "DistilBERT pipeline",
    "Tier3b_SpERT": "SpERT",
}
# Okabe-Ito colourblind-safe palette, fixed by model for reproducibility.
COLORS = {
    "Tier1_CRF_LogReg": "#0072B2",
    "Tier2_BiLSTM_Neural": "#E69F00",
    "Tier3_Transformer": "#009E73",
    "Tier3b_SpERT": "#D55E00",
}
ROOT_REPORT = Path("outputs/reports")
FINAL_FIG = ROOT_REPORT / "final_figures"
FINAL_TAB = ROOT_REPORT / "final_tables"
HISTORY = ROOT_REPORT / "final_figure_history"


def _load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _payload_metrics(payload):
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
        return payload["metrics"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Metric artifact must be a JSON object")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_previous(stamp):
    HISTORY.mkdir(parents=True, exist_ok=True)
    archive = HISTORY / stamp
    moved = []
    if FINAL_FIG.exists():
        archive.mkdir(parents=True, exist_ok=True)
        dst = archive / "final_figures"
        shutil.move(str(FINAL_FIG), str(dst))
        moved.append(str(dst).replace("\\", "/"))
    if FINAL_TAB.exists():
        archive.mkdir(parents=True, exist_ok=True)
        dst = archive / "final_tables"
        shutil.move(str(FINAL_TAB), str(dst))
        moved.append(str(dst).replace("\\", "/"))
    return moved


def _savefig(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png).replace("\\", "/"), str(pdf).replace("\\", "/")]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.24, linewidth=0.8)
    ax.set_axisbelow(True)


def _gold_test_support(path):
    docs = _load_json(path)
    ent = Counter(e["type"] for d in docs for e in d.get("entities", []))
    rel = Counter(r["type"] for d in docs for r in d.get("relations", []))
    return docs, ent, rel


def _corpus_support(glob_path):
    records = load_gold(glob_path)
    ent = Counter()
    rel = Counter()
    for d in records:
        for e in bio_to_entities(d["tokens"], d["bio"]):
            ent[e["type"]] += 1
        for r in d.get("relations", []):
            rel[r["type"]] += 1
    return len(records), ent, rel


def _score_existing_spert(gold_path, pred_path):
    gold = _load_json(gold_path)
    pred = _load_json(pred_path)
    if len(gold) != len(pred):
        raise RuntimeError(f"SpERT TEST alignment mismatch: {len(gold)} gold != {len(pred)} predictions")
    for i, (g, p) in enumerate(zip(gold, pred)):
        if g.get("tokens") != p.get("tokens"):
            raise RuntimeError(f"SpERT token-order mismatch at TEST index {i}")
    return {
        "entity": entity_scores(gold, pred),
        "relation": relation_scores(gold, pred),
    }


def _load_core_metrics():
    classical_path = Path("outputs/reports/ie_results_core.json")
    neural_path = Path("outputs/reports/ie_results_core_neural.json")
    spert_gold = Path("outputs/spert_core/test.json")
    spert_pred = Path("outputs/spert_core/predictions_test.json")
    metrics = {}
    metrics.update(_payload_metrics(_load_json(classical_path)))
    metrics.update(_payload_metrics(_load_json(neural_path)))
    metrics["Tier3b_SpERT"] = _score_existing_spert(spert_gold, spert_pred)
    missing = [m for m in MODEL_ORDER if m not in metrics]
    if missing:
        raise RuntimeError(f"Core metrics missing required models: {missing}")
    sources = [classical_path, neural_path, spert_gold, spert_pred]
    return {m: metrics[m] for m in MODEL_ORDER}, sources


def _load_full_metrics():
    combined = Path("outputs/reports/ie_results_combined_existing.json")
    metrics = _payload_metrics(_load_json(combined))
    missing = [m for m in MODEL_ORDER if m not in metrics]
    if missing:
        raise RuntimeError(f"Full metrics missing required models: {missing}")
    # Verify the full SpERT frozen-test artifacts still align, but do not replace
    # the frozen combined score. This is a data-integrity check only.
    full_gold = Path("outputs/spert/test.json")
    full_pred = Path("outputs/spert/predictions_test.json")
    _score_existing_spert(full_gold, full_pred)
    return {m: metrics[m] for m in MODEL_ORDER}, [combined, full_gold, full_pred]


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _overall_rows(metrics):
    rows = []
    for m in MODEL_ORDER:
        ent = metrics[m]["entity"]
        rel = metrics[m]["relation"]
        gold_rel = metrics[m].get("relation_gold_entities", {})
        rows.append({
            "model": m,
            "display_name": NICE[m],
            "entity_micro_p": ent.get("micro_p"),
            "entity_micro_r": ent.get("micro_r"),
            "entity_micro_f1": ent.get("micro_f1"),
            "entity_macro_f1": ent.get("macro_f1"),
            "relation_micro_p": rel.get("micro_p"),
            "relation_micro_r": rel.get("micro_r"),
            "relation_micro_f1": rel.get("micro_f1"),
            "relation_macro_f1": rel.get("macro_f1"),
            "gold_entity_relation_micro_f1": gold_rel.get("micro_f1"),
        })
    return rows


def _per_class_rows(metrics, support, kind):
    labels = sorted(support, key=lambda k: (-support[k], k))
    rows = []
    for label in labels:
        row = {"class": label, "test_support": support[label]}
        for m in MODEL_ORDER:
            vals = metrics[m][kind].get("per_class", {}).get(label, [0.0, 0.0, 0.0])
            row[f"{m}__precision"] = vals[0]
            row[f"{m}__recall"] = vals[1]
            row[f"{m}__f1"] = vals[2]
        rows.append(row)
    return rows


def _fig_overall(metrics, title, stem):
    groups = ["Entity\nmicro-F1", "Entity\nmacro-F1", "Strict relation\nmicro-F1", "Relation\nmacro-F1"]
    values = {
        m: [
            metrics[m]["entity"]["micro_f1"],
            metrics[m]["entity"]["macro_f1"],
            metrics[m]["relation"]["micro_f1"],
            metrics[m]["relation"]["macro_f1"],
        ] for m in MODEL_ORDER
    }
    x = np.arange(len(groups))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for i, m in enumerate(MODEL_ORDER):
        pos = x + (i - 1.5) * width
        bars = ax.bar(pos, values[m], width, label=NICE[m], color=COLORS[m])
        for b, v in zip(bars, values[m]):
            ax.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    return _savefig(fig, stem)


def _fig_per_class(metrics, support, kind, title, stem, rare_note=None):
    labels = sorted(support, key=lambda k: (-support[k], k))
    display = [f"{c}\n(n={support[c]})" for c in labels]
    x = np.arange(len(labels))
    width = 0.19
    fig_w = max(11.0, len(labels) * 1.18)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))
    for i, m in enumerate(MODEL_ORDER):
        vals = []
        for c in labels:
            vals.append(float(metrics[m][kind].get("per_class", {}).get(c, [0, 0, 0])[2]))
        ax.bar(x + (i - 1.5) * width, vals, width, label=NICE[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=35, ha="right", fontsize=8.2)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    _style(ax)
    if rare_note:
        fig.text(0.01, 0.01, rare_note, fontsize=8, ha="left", va="bottom")
        fig.subplots_adjust(bottom=0.30)
    else:
        fig.subplots_adjust(bottom=0.25)
    return _savefig(fig, stem)


def _fig_support(support, title, stem, ylabel="count"):
    items = sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels)*0.9), 5.0))
    bars = ax.bar(labels, vals, color="#0072B2")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    return _savefig(fig, stem)


def _fig_schema_compare(core, full, metric_path, title, stem):
    labels = [NICE[m] for m in MODEL_ORDER]
    core_vals, full_vals = [], []
    for m in MODEL_ORDER:
        a = core[m]
        b = full[m]
        for key in metric_path:
            a = a[key]
            b = b[key]
        core_vals.append(a)
        full_vals.append(b)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    b1 = ax.bar(x-width/2, core_vals, width, label="Core 8/10", color="#56B4E9")
    b2 = ax.bar(x+width/2, full_vals, width, label="Full 9/11", color="#D55E00")
    for bars, vals in ((b1, core_vals), (b2, full_vals)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.007, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return _savefig(fig, stem)


def _write_support_csv(path, support):
    _write_csv(path, [{"class": k, "support": v} for k, v in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))], ["class", "support"])


def main():
    _bootstrap.banner("FINAL AVIATION IE FIGURES FROM EXISTING RESULTS - NO TRAINING")
    print("  Old outputs/reports/figures are preserved and are NOT used as final figures.")
    print("  Existing model checkpoints are not loaded or trained.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = _archive_previous(stamp)
    FINAL_FIG.mkdir(parents=True, exist_ok=True)
    FINAL_TAB.mkdir(parents=True, exist_ok=True)

    core_metrics, core_sources = _load_core_metrics()
    full_metrics, full_sources = _load_full_metrics()

    core_test_docs, core_test_ent, core_test_rel = _gold_test_support("outputs/spert_core/test.json")
    full_test_docs, full_test_ent, full_test_rel = _gold_test_support("outputs/spert/test.json")
    core_n, core_corpus_ent, core_corpus_rel = _corpus_support("outputs/gold_core/*.jsonl")
    full_n, full_corpus_ent, full_corpus_rel = _corpus_support("outputs/gold/*.jsonl")

    if len(core_test_docs) != 225 or len(full_test_docs) != 225:
        raise RuntimeError(f"Expected frozen TEST=225; got core={len(core_test_docs)} full={len(full_test_docs)}")

    rare_entity_n = int(full_test_ent.get("REFERENCE", 0))
    rare_relation_n = int(full_test_rel.get("ACTION_FOLLOWS_REFERENCE", 0))
    rare_note = (
        f"Rare-class caution: REFERENCE n={rare_entity_n} and ACTION_FOLLOWS_REFERENCE n={rare_relation_n} "
        "on the frozen TEST. Perfect F1 is descriptive, not robust evidence of generalization."
    )

    generated = []
    for name, metrics, te, tr, ce, cr in (
        ("core_8x10", core_metrics, core_test_ent, core_test_rel, core_corpus_ent, core_corpus_rel),
        ("full_9x11", full_metrics, full_test_ent, full_test_rel, full_corpus_ent, full_corpus_rel),
    ):
        out = FINAL_FIG / name
        title_schema = "Core 8-entity / 10-relation schema" if name.startswith("core") else "Full 9-entity / 11-relation schema"
        note = rare_note if name.startswith("full") else None
        generated += _fig_overall(metrics, f"Aviation IE overall performance — {title_schema} (frozen TEST)", out / "overall_four_model_comparison")
        generated += _fig_per_class(metrics, te, "entity", f"Per-class entity F1 — {title_schema} (frozen TEST support)", out / "entity_per_class_f1_all_models", note)
        generated += _fig_per_class(metrics, tr, "relation", f"Per-class strict relation F1 — {title_schema} (frozen TEST support)", out / "relation_per_class_f1_all_models", note)
        generated += _fig_support(te, f"Entity support — {title_schema} frozen TEST", out / "test_entity_support", "gold spans")
        generated += _fig_support(tr, f"Relation support — {title_schema} frozen TEST", out / "test_relation_support", "gold relations")
        generated += _fig_support(ce, f"Entity support — {title_schema} full 1,600-record gold corpus", out / "corpus_entity_support", "gold spans")
        generated += _fig_support(cr, f"Relation support — {title_schema} full 1,600-record gold corpus", out / "corpus_relation_support", "gold relations")

        tab = FINAL_TAB / name
        overall = _overall_rows(metrics)
        _write_csv(tab / "overall_metrics.csv", overall, list(overall[0].keys()))
        ent_rows = _per_class_rows(metrics, te, "entity")
        rel_rows = _per_class_rows(metrics, tr, "relation")
        _write_csv(tab / "entity_per_class_all_models.csv", ent_rows, list(ent_rows[0].keys()))
        _write_csv(tab / "relation_per_class_all_models.csv", rel_rows, list(rel_rows[0].keys()))
        _write_support_csv(tab / "test_entity_support.csv", te)
        _write_support_csv(tab / "test_relation_support.csv", tr)
        _write_support_csv(tab / "corpus_entity_support.csv", ce)
        _write_support_csv(tab / "corpus_relation_support.csv", cr)

    cmp_dir = FINAL_FIG / "core_vs_full"
    generated += _fig_schema_compare(core_metrics, full_metrics, ["entity", "micro_f1"], "Core 8/10 vs full 9/11 — entity micro-F1", cmp_dir / "entity_micro_f1")
    generated += _fig_schema_compare(core_metrics, full_metrics, ["relation", "micro_f1"], "Core 8/10 vs full 9/11 — strict relation micro-F1", cmp_dir / "strict_relation_micro_f1")
    generated += _fig_schema_compare(core_metrics, full_metrics, ["relation", "macro_f1"], "Core 8/10 vs full 9/11 — relation macro-F1", cmp_dir / "relation_macro_f1")

    comparison_rows = []
    for m in MODEL_ORDER:
        comparison_rows.append({
            "model": m,
            "display_name": NICE[m],
            "core_entity_micro_f1": core_metrics[m]["entity"]["micro_f1"],
            "full_entity_micro_f1": full_metrics[m]["entity"]["micro_f1"],
            "delta_full_minus_core_entity": round(full_metrics[m]["entity"]["micro_f1"] - core_metrics[m]["entity"]["micro_f1"], 4),
            "core_relation_micro_f1": core_metrics[m]["relation"]["micro_f1"],
            "full_relation_micro_f1": full_metrics[m]["relation"]["micro_f1"],
            "delta_full_minus_core_relation": round(full_metrics[m]["relation"]["micro_f1"] - core_metrics[m]["relation"]["micro_f1"], 4),
            "core_relation_macro_f1": core_metrics[m]["relation"]["macro_f1"],
            "full_relation_macro_f1": full_metrics[m]["relation"]["macro_f1"],
            "delta_full_minus_core_relation_macro": round(full_metrics[m]["relation"]["macro_f1"] - core_metrics[m]["relation"]["macro_f1"], 4),
        })
    _write_csv(FINAL_TAB / "core_vs_full_model_comparison.csv", comparison_rows, list(comparison_rows[0].keys()))

    source_files = []
    for p in core_sources + full_sources:
        p = Path(p)
        if p.exists() and str(p) not in [str(x) for x in source_files]:
            source_files.append(p)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "existing-results-only",
        "training_performed": False,
        "old_legacy_figures_deleted": False,
        "legacy_figure_directory_preserved": "outputs/reports/figures",
        "stable_final_figure_directory": str(FINAL_FIG).replace("\\", "/"),
        "stable_final_table_directory": str(FINAL_TAB).replace("\\", "/"),
        "archived_previous_final_outputs": archived,
        "frozen_test": {"core": len(core_test_docs), "full": len(full_test_docs)},
        "gold_corpus_records": {"core": core_n, "full": full_n},
        "rare_class_test_support": {
            "REFERENCE": rare_entity_n,
            "ACTION_FOLLOWS_REFERENCE": rare_relation_n,
        },
        "rare_class_interpretation": "Perfect rare-class F1 is retained as measured but labelled with n=2 support and must not be presented as robust generalization.",
        "source_files": [
            {"path": str(p).replace("\\", "/"), "sha256": _sha256(p)} for p in source_files
        ],
        "generated_files": generated,
    }
    manifest_path = ROOT_REPORT / "FINAL_IE_FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\n[FINAL FIGURE SET CREATED]")
    print(f"  core figures : {FINAL_FIG / 'core_8x10'}")
    print(f"  full figures : {FINAL_FIG / 'full_9x11'}")
    print(f"  comparison   : {FINAL_FIG / 'core_vs_full'}")
    print(f"  final tables : {FINAL_TAB}")
    print(f"  manifest     : {manifest_path}")
    print(f"  REFERENCE TEST support={rare_entity_n}; ACTION_FOLLOWS_REFERENCE TEST support={rare_relation_n}")
    print("\nNO TRAINING PERFORMED. OLD outputs/reports/figures WERE NOT DELETED.")


if __name__ == "__main__":
    main()
