"""Planning & support: recurring-fault register + grounded job-card drafting.

A job card is assembled ONLY from recorded historical actions — deduplicated,
imperative-tensed, each line traceable to source work orders. No step is
invented. This is the deterministic planning handoff, made visible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import normalize as N


def _imperative(action_text: str) -> str:
    """Light past->imperative rendering; verbatim otherwise (no invented content)."""
    s = re.sub(r"\s+", " ", (action_text or "").strip())
    repl = {
        r"\bremoved\b": "Remove", r"\breplaced\b": "Replace", r"\binstalled\b": "Install",
        r"\bcleaned\b": "Clean", r"\binspected\b": "Inspect", r"\bchecked\b": "Check",
        r"\badjusted\b": "Adjust", r"\brepaired\b": "Repair", r"\bserviced\b": "Service",
        r"\btightened\b": "Tighten", r"\bfound\b": "Find", r"\bperformed\b": "Perform",
        r"\btested\b": "Test", r"\bre-?safetied\b": "Re-safety",
    }
    low = s
    for pat, word in repl.items():
        if re.match(pat, low, re.IGNORECASE):
            return word + s[len(re.match(pat, low, re.IGNORECASE).group(0)):]
    return s


@dataclass
class JobCard:
    title: str
    component: str
    fault: str
    work_orders: int
    problem_groups: int
    dominant_action: str
    steps: list[dict] = field(default_factory=list)      # {text, source_idents}
    references: list[str] = field(default_factory=list)
    outcome_positive: int = 0
    outcome_negative: int = 0
    outcome_unknown: int = 0
    source_idents: list[str] = field(default_factory=list)


def job_card_for_cluster(df: pd.DataFrame, cluster_id: str, max_steps: int = 8) -> JobCard:
    sub = df[df["cluster_id"] == str(cluster_id)]
    return _build_card(sub, title=sub["problem"].iloc[0] if len(sub) else "")


def job_card_for_component_fault(df: pd.DataFrame, component: str,
                                 fault: str | None = None) -> JobCard:
    mask = df["components"].map(lambda xs: component in xs)
    if fault:
        mask = mask & df["faults"].map(lambda xs: fault in xs)
    sub = df[mask]
    title = f"{component}" + (f" — {fault}" if fault else "")
    return _build_card(sub, title=title)


def _build_card(sub: pd.DataFrame, title: str) -> JobCard:
    if sub.empty:
        return JobCard(title=title, component="", fault="", work_orders=0,
                       problem_groups=0, dominant_action="")
    dominant = sub["action_family"].mode().iloc[0] if not sub["action_family"].mode().empty else ""
    # deduplicate recorded action steps, prefer positive/known outcomes
    steps: list[dict] = []
    seen: set[str] = set()
    ordered = sub.sort_values(
        by="outcome", key=lambda s: s.map({"positive": 0, "unknown": 1, "mixed": 2, "negative": 3}))
    for row in ordered.itertuples(index=False):
        act = getattr(row, "action", "").strip()
        norm = N.normalize_problem(act)
        if not norm or norm in seen:
            continue
        # only steps whose family is not a failed attempt
        if getattr(row, "outcome") in ("negative", "mixed"):
            continue
        seen.add(norm)
        steps.append({"text": _imperative(act), "source_idents": [str(getattr(row, "ident"))]})
        if len(steps) >= 8:
            break
    comp = sub["component"].mode().iloc[0] if not sub["component"].mode().empty else ""
    fault = sub["fault"].mode().iloc[0] if not sub["fault"].mode().empty else ""
    return JobCard(
        title=title, component=comp, fault=fault,
        work_orders=len(sub), problem_groups=int(sub["cluster_id"].nunique()),
        dominant_action=dominant, steps=steps,
        outcome_positive=int((sub["outcome"] == "positive").sum()),
        outcome_negative=int(sub["outcome"].isin(["negative", "mixed"]).sum()),
        outcome_unknown=int((sub["outcome"] == "unknown").sum()),
        source_idents=[str(x) for x in sub["ident"].head(25).tolist()],
    )

