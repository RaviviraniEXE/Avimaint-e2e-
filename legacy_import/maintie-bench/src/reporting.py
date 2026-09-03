"""Evaluation reporting: per-class tables (with support), confusion matrices, and
publication-style comparison figures for the IE tiers.

Figures use the Okabe-Ito colourblind-safe palette. Everything is written to
outputs/reports/ (tables/*.csv, figures/*.png, metrics_full.json). Import the
helpers or run scripts/09_report.py.
"""
from __future__ import annotations

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import entity_scores, relation_scores

# Okabe-Ito colourblind-safe categorical palette (fixed order, never cycled).
# Colour is assigned per model name in fixed order, so ANY set of models — the
# three tiers here, or a MaintIE benchmark line-up — gets stable, distinct hues.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
            "#D55E00", "#F0E442", "#000000"]
_PREFERRED = {"Tier1_CRF_LogReg": "#0072B2", "Tier2_BiLSTM_Neural": "#E69F00",
              "Tier3_Transformer": "#009E73", "Tier3_SpERT": "#CC79A7"}
NICE = {"Tier1_CRF_LogReg": "Tier 1 · CRF+LogReg",
        "Tier2_BiLSTM_Neural": "Tier 2 · BiLSTM+Neural",
        "Tier3_Transformer": "Tier 3 · Transformer",
        "Tier3_SpERT": "Tier 3 · SpERT"}


def color_for(name, order=None):
    if name in _PREFERRED:
        return _PREFERRED[name]
    i = (order or 0) % len(_PALETTE)
    return _PALETTE[i]


class _ColorMap:
    """dict-like: returns a stable colour for any model name."""
    def __init__(self):
        self._seen = {}

    def get(self, name, default="#999999"):
        if name not in self._seen:
            self._seen[name] = color_for(name, len(self._seen))
        return self._seen[name]


MODEL_COLORS = _ColorMap()
FIGDIR = "outputs/reports/figures"
TABDIR = "outputs/reports/tables"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def _tag_type(t):
    return t[2:] if t not in ("O", "") and t[:2] in ("B-", "I-") else "O"


# ---------- tables ----------------------------------------------------------
def per_class_table(gold, pred, kind="entity"):
    """Return a DataFrame with precision/recall/F1/support per class."""
    scores = entity_scores(gold, pred) if kind == "entity" else relation_scores(gold, pred)
    if kind == "entity":
        sup = Counter(e["type"] for g in gold for e in g["entities"])
    else:
        sup = Counter(r["type"] for g in gold for r in g.get("relations", []))
    rows = []
    for t, (p, r, f) in scores["per_class"].items():
        rows.append({"class": t, "precision": p, "recall": r, "f1": f, "support": sup.get(t, 0)})
    df = pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)
    # micro / macro summary rows
    df.loc[len(df)] = {"class": "micro avg", "precision": scores["micro_p"],
                       "recall": scores["micro_r"], "f1": scores["micro_f1"],
                       "support": scores["support"]}
    df.loc[len(df)] = {"class": "macro avg", "precision": np.nan, "recall": np.nan,
                       "f1": scores["macro_f1"], "support": scores["support"]}
    return df


def confusion(gold_bio, pred_bio, types):
    """Token-level type confusion matrix (rows=gold, cols=pred, incl. O)."""
    labels = list(types) + ["O"]
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for gb, pb in zip(gold_bio, pred_bio):
        for g, p in zip(gb, pb):
            gt, pt = _tag_type(g), _tag_type(p)
            if gt in idx and pt in idx:
                M[idx[gt], idx[pt]] += 1
    return labels, M


# ---------- figures ---------------------------------------------------------
def fig_overall(metrics: dict, path):
    """Grouped bars: entity micro/macro-F1 and relation micro-F1 per model, with
    bootstrap 95% CI error bars on the two micro-F1 columns."""
    groups = ["Entity micro-F1", "Entity macro-F1", "Relation micro-F1"]
    models = list(metrics)
    x = np.arange(len(groups)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, m in enumerate(models):
        vals = [metrics[m]["entity"]["micro_f1"], metrics[m]["entity"]["macro_f1"],
                metrics[m]["relation"]["micro_f1"]]
        eci, rci = metrics[m]["entity"].get("ci"), metrics[m]["relation"].get("ci")
        lo = [vals[0] - eci["lo"] if eci else 0, 0, vals[2] - rci["lo"] if rci else 0]
        hi = [eci["hi"] - vals[0] if eci else 0, 0, rci["hi"] - vals[2] if rci else 0]
        pos = x + i * w - 0.4 + w / 2
        ax.bar(pos, vals, w, label=NICE.get(m, m), color=MODEL_COLORS.get(m, "#999999"),
               yerr=[lo, hi], capsize=3, ecolor="#444444", error_kw={"linewidth": 1})
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1"); ax.set_title("Overall performance by model (frozen test · 95% CI)")
    ax.legend(frameon=False, fontsize=9); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def memorization_baseline(train, test_records):
    """Per-word most-frequent-BIO-tag lookup (no model/context) — the memorization
    floor. The gap between a model and this shows how much is learned from context."""
    from collections import Counter, defaultdict
    from src.models.crf_ner import bio_to_entities
    wt = defaultdict(Counter)
    for d in train:
        for w, t in zip(d["tokens"], d["bio"]):
            wt[w.lower()][t] += 1
    maj = {w: c.most_common(1)[0][0] for w, c in wt.items()}
    gold = [{"tokens": d["tokens"], "entities": bio_to_entities(d["tokens"], d["bio"]), "relations": []}
            for d in test_records]
    pred = [{"tokens": d["tokens"],
             "entities": bio_to_entities(d["tokens"], [maj.get(w.lower(), "O") for w in d["tokens"]]),
             "relations": []} for d in test_records]
    from src.evaluate import entity_scores
    tr_words = {w.lower() for d in train for w in d["tokens"]}
    te_words = {w.lower() for d in test_records for w in d["tokens"]}
    oov = len(te_words - tr_words) / max(len(te_words), 1)
    return {"memorization_f1": entity_scores(gold, pred)["micro_f1"], "test_oov_rate": round(oov, 4)}


def fig_per_class(tables: dict, kind, path):
    """Grouped bars of per-class F1 across models, ordered by support."""
    models = list(tables)
    base = tables[models[0]]
    base = base[~base["class"].isin(["micro avg", "macro avg"])]
    classes = list(base.sort_values("support", ascending=False)["class"])
    x = np.arange(len(classes)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.9), 4.8))
    for i, m in enumerate(models):
        d = tables[m].set_index("class")
        vals = [float(d.loc[c, "f1"]) if c in d.index else 0.0 for c in classes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=NICE.get(m, m),
               color=MODEL_COLORS.get(m, "#999999"))
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.set_title(f"Per-class {kind} F1 by model (ordered by support)")
    ax.legend(frameon=False, fontsize=9); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_confusion(labels, M, title, path):
    """Row-normalised confusion heatmap with raw counts annotated."""
    Mn = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(labels), 1.0 + 0.7 * len(labels)))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.5 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_support(counts: dict, title, path, color="#0072B2"):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.2))
    bars = ax.bar(labels, vals, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("count (spans)"); ax.set_title(title); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_learning_curve(traj: pd.DataFrame, path):
    """F1 vs training-set size across rounds, one line per tier/metric, with 95% CI
    error bars where available."""
    def yerr(d, col):
        lo, hi = col + "_ci_lo", col + "_ci_hi"
        if lo in d and hi in d and d[lo].notna().any():
            return np.vstack([d[col] - d[lo], d[hi] - d[col]])
        return None
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for m in sorted(traj["tier"].unique()):
        d = traj[traj["tier"] == m].sort_values("n_train")
        c = MODEL_COLORS.get(m, "#999999")
        ax.errorbar(d["n_train"], d["ent_micro_f1"], yerr=yerr(d, "ent_micro_f1"),
                    fmt="-o", color=c, capsize=3, label=f"{NICE.get(m, m)} · entity")
        ax.errorbar(d["n_train"], d["rel_micro_f1"], yerr=yerr(d, "rel_micro_f1"),
                    fmt="--s", color=c, capsize=3, alpha=0.7, label=f"{NICE.get(m, m)} · relation")
    ax.set_xlabel("training gold records"); ax.set_ylabel("micro-F1"); ax.set_ylim(0, 1.02)
    ax.set_title("Learning curve — F1 vs corpus size (95% CI)")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_training_curve(history, path, title):
    """Two-panel training curve: train loss (top) and dev-F1 (bottom) vs epoch,
    with the best (early-stopping) epoch marked. `history` = [{epoch,loss,dev_f1}]."""
    if not history:
        return
    ep = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    devf = [h.get("dev_f1") for h in history]
    best_i = int(np.argmax([d if d is not None else -1 for d in devf]))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    a1.plot(ep, loss, "-o", color="#D55E00", markersize=3); a1.set_ylabel("train loss")
    a1.set_title(title); _style(a1)
    a2.plot(ep, devf, "-o", color="#0072B2", markersize=3); a2.set_ylabel("dev-F1")
    a2.set_xlabel("epoch")
    a2.axvline(ep[best_i], color="#009E73", linestyle="--", linewidth=1)
    a2.annotate(f"best (kept)\nepoch {ep[best_i]} · {devf[best_i]:.3f}",
                xy=(ep[best_i], devf[best_i]), xytext=(6, -28), textcoords="offset points",
                fontsize=8, color="#009E73")
    _style(a2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def ensure_dirs():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)

