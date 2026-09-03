"""Thesis figures for the IE corpus-building stage (random pilot + active learning).
  fig_ie_pipeline.png     methodology flowchart (random pilot, freeze, AL, 3 tiers)
  fig_annotation_plan.png gold growth across rounds + frozen test/dev
  fig_entity_support.png  per-entity corpus support
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = "outputs/figures"; os.makedirs(OUT, exist_ok=True)
INK, GRID = "#222222", "#DDDDDD"
OK = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999", "#666666"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.8, 12)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(cx, cy, w, h, t, fc="#F5F7FA", ec="#5A6B7B", fs=9.3, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.5,rounding_size=2", fc=fc, ec=ec, lw=1.4))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=fs, linespacing=1.4,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, c="#93A0AD", lw=1.5):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=13, color=c, lw=lw, shrinkA=2, shrinkB=3))

    ax.text(50, 98.5, "Figure.  IE corpus building (random pilot + active learning) and modelling",
            ha="center", fontsize=12.5, fontweight="bold")
    box(50, 93, 62, 5.5, "Normalized MaintNet corpus (6,169) → dedup → 5,122 unique pairs", fc="#EAF0F6", fs=9.6)
    box(50, 84.5, 56, 6, "RANDOM pilot (300)  —  unbiased sample", fc="#FBF3E4", bold=True)
    box(50, 75.5, 66, 6.5, "Weak pre-annotation  →  Label Studio tasks\ncorrect in Label Studio  →  import as GOLD", fc="#FDF0E0", ec="#C8892B")
    arr(50, 90.2, 50, 87.5); arr(50, 81.5, 50, 78.7)

    box(50, 65, 70, 6.5, "Random rounds 1–2:  CRF pre-labels next 500, 300  →  correct  →  gold ≈ 1,100", fc="#EAF2EC", ec="#2E7D5B", fs=9)
    arr(50, 72.2, 50, 68.3)
    box(16, 54.5, 26, 7, "FREEZE random\ntest (225) + dev (100)\n— never enriched", fc="#EDE7F6", ec="#6A4CA5", fs=8.6, bold=True)
    box(64, 54.5, 60, 7.5, "Active-learning rounds:  train CRF → predict pool →\nrank by rare-class + uncertainty → correct  (TRAIN only)", fc="#EAF2EC", ec="#2E7D5B", fs=9)
    arr(43, 61.7, 22, 58.3, c="#6A4CA5"); arr(57, 61.7, 64, 58.5, c="#2E7D5B")
    ax.text(64, 49.5, "gold ≈ 1,500  (train enriched; test/dev fixed)", ha="center", fontsize=8.4, style="italic", color="#2E7D5B")

    box(50, 40, 80, 5.5, "Train on TRAIN  ·  evaluate on FROZEN random TEST", fc="#F5F7FA")
    arr(64, 50.7, 55, 43); arr(16, 51, 40, 43, c="#6A4CA5")

    for x, lab, c in [(18, "Tier 1 · Baseline\nCRF + Logistic Regression", OK[0]),
                      (50, "Tier 2 · Mid\nBiLSTM-CRF + neural RE", OK[1]),
                      (82, "Tier 3 · Transformer\nSpERT (joint)", OK[2])]:
        box(x, 27, 30, 9, lab, fc="#FFFFFF", ec=c, bold=True, fs=9)
        arr(50, 37, x, 31.7, c="#B2BCC7", lw=1.1)
    box(50, 12, 82, 7.5, "Evaluation:  entity & relation micro / macro / per-class F1\n+  normalization → IE F1 study",
        fc="#EAF2EC", ec="#2E7D5B", fs=9.3)
    for x in (18, 50, 82):
        arr(x, 22.5, 50, 16, c="#B2BCC7", lw=1.1)
    plt.savefig(f"{OUT}/fig_ie_pipeline.png", dpi=200, bbox_inches="tight"); plt.close()


def fig_annotation_plan():
    rounds = ["Pilot", "Round 1", "Round 2", "Rare (AL)"]
    add = [300, 500, 300, 400]
    cum = [300, 800, 1100, 1500]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(range(4), cum, color=[OK[7], OK[0], OK[0], OK[2]], edgecolor="white", linewidth=1.3)
    for i, (b, a, c) in enumerate(zip(bars, add, cum)):
        ax.text(b.get_x() + b.get_width() / 2, c + 25, f"+{a}\n= {c}", ha="center", fontsize=9)
    ax.axhline(800, color="#6A4CA5", ls="--", lw=1.3)
    ax.text(3.4, 815, "freeze test/dev here", color="#6A4CA5", fontsize=8.5, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels(rounds, fontsize=10)
    ax.set_title("Gold-corpus growth plan (~1,500 total)", fontweight="bold")
    ax.set_ylabel("cumulative gold records"); ax.set_ylim(0, 1700)
    ax.legend(handles=[Patch(color=OK[7], label="random pilot"),
                       Patch(color=OK[0], label="random rounds"),
                       Patch(color=OK[2], label="active-learning (rare) → train only")],
              frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_annotation_plan.png", dpi=200, bbox_inches="tight"); plt.close()


def fig_entity_support():
    df = pd.read_csv("data/raw/Aircraft_Annotation_DataFile.csv", dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    both = (df["PROBLEM"] + " " + df["ACTION"]).str.lower(); N = len(df)
    cues = {"MAINT_ITEM": r"gasket|cylinder|baffle|cover|engine|intake|seal|screw|valve|tube|hose|plug|bolt",
            "ACTION": r"replac|remov|install|inspect|check|clean|tighten|torqu|fabricat|repair",
            "ABN_PROC": r"leak|vibrat|rough|sputter|overheat|chaf", "FAULT": r"crack|loose|broken|worn|missing|damag|stuck",
            "LOC": r"r/h|l/h|right|left|#\d|fwd|aft|forward|inboard", "TECH_OBS": r"compression|pressure|clearance|tappet|\d+ ?(psi|rpm)",
            "OUTCOME": r"\bgood\b|no leak|within limit|serviceable|could not duplicate", "OP_CTX": r"during|in flight|at idle|climb|taxi|takeoff",
            "REFERENCE": r"\bad \d|service bulletin|manual|chapter \d|iaw"}
    ents = list(cues); vals = [int(both.str.contains(p, regex=True).sum()) for p in cues.values()]
    o = sorted(range(len(ents)), key=lambda i: -vals[i]); ents = [ents[i] for i in o]; vals = [vals[i] for i in o]
    colors = [OK[0] if v >= 1000 else (OK[1] if v >= 400 else OK[5]) for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(range(len(ents))[::-1], vals, color=colors, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(v + N * 0.008, b.get_y() + b.get_height() / 2, f"{v}  ({100*v/N:.1f}%)", va="center", fontsize=9)
    ax.set_yticks(range(len(ents))[::-1]); ax.set_yticklabels(ents, fontsize=10)
    ax.set_title("Entity support in the corpus (records with a cue, of 6,169)", fontweight="bold")
    ax.set_xlabel("records"); ax.set_xlim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=OK[0], label="high (≥1000)"), Patch(color=OK[1], label="moderate (400–999)"),
                       Patch(color=OK[5], label="low (<400)")], frameon=False, fontsize=8.5, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_entity_support.png", dpi=200, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    fig_pipeline(); fig_annotation_plan(); fig_entity_support()
    print("figures ->", OUT)

