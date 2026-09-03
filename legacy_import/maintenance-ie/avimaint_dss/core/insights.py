"""Frequency analytics over the corpus. Count-based only (no reliability rates).

Every function returns a tidy pandas DataFrame ready to chart. All wording is
"observed work orders" / "recorded", never failure rate or severity.
"""
from __future__ import annotations

import collections

import pandas as pd


def _explode_counts(df: pd.DataFrame, col: str) -> pd.Series:
    c = collections.Counter(x for xs in df[col] for x in xs if x and x != "(unspecified)")
    return pd.Series(c).sort_values(ascending=False)


def kpis(df: pd.DataFrame, recurring_min: int = 5) -> dict:
    support = df.groupby("cluster_id").size()
    recurring = int((support >= recurring_min).sum())
    comps = _explode_counts(df, "components")
    return {
        "work_orders": int(len(df)),
        "unique_problems": int(df["problem_norm"].nunique()),
        "problem_clusters": int(df["cluster_id"].nunique()),
        "components_tracked": int(len(comps)),
        "action_families": int((df["action_family"] != "Other").sum() and df["action_family"].nunique()),
        "recurring_faults": recurring,
        "recorded_outcomes_pct": round(100 * (df["outcome"] != "unknown").mean(), 1),
        "spert_backed": bool((df["source"] == "spert").any()),
    }


def top_recurring_problems(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    g = df.groupby("cluster_id")
    rows = []
    for cid, sub in g:
        rows.append({
            "cluster_id": cid,
            "problem": sub["problem"].iloc[0],
            "work_orders": len(sub),
            "component": sub["component"].mode().iloc[0] if not sub["component"].mode().empty else "",
            "fault": sub["fault"].mode().iloc[0] if not sub["fault"].mode().empty else "",
            "top_action": sub["action_family"].mode().iloc[0] if not sub["action_family"].mode().empty else "",
        })
    out = pd.DataFrame(rows).sort_values("work_orders", ascending=False).head(n).reset_index(drop=True)
    return out


def component_frequency(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    s = _explode_counts(df, "components").head(n)
    out = s.rename_axis("component").reset_index(name="work_orders")
    total = _explode_counts(df, "components").sum()
    out["cumulative_pct"] = (out["work_orders"].cumsum() / total * 100).round(1)
    out["share_pct"] = (out["work_orders"] / total * 100).round(1)
    return out


def fault_frequency(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    s = _explode_counts(df, "faults").head(n)
    out = s.rename_axis("fault").reset_index(name="work_orders")
    total = _explode_counts(df, "faults").sum()
    out["share_pct"] = (out["work_orders"] / total * 100).round(1)
    return out


def action_frequency(df: pd.DataFrame) -> pd.DataFrame:
    s = df["action_family"].value_counts()
    out = s.rename_axis("action_family").reset_index(name="work_orders")
    out = out[out["action_family"] != "Other"]
    return out.reset_index(drop=True)


def component_fault_matrix(df: pd.DataFrame, top_c: int = 8, top_f: int = 8) -> pd.DataFrame:
    comps = _explode_counts(df, "components").head(top_c).index.tolist()
    faults = _explode_counts(df, "faults").head(top_f).index.tolist()
    mat = pd.DataFrame(0, index=comps, columns=faults)
    for _, r in df.iterrows():
        for c in set(r["components"]):
            if c not in mat.index:
                continue
            for f in set(r["faults"]):
                if f in mat.columns:
                    mat.loc[c, f] += 1
    return mat


def problem_to_action(df: pd.DataFrame, by: str = "component", key: str | None = None,
                      n: int = 12) -> pd.DataFrame:
    """Action-family distribution for a given component or fault (or overall)."""
    if key:
        mask = df[by + "s"].map(lambda xs: key in xs) if by + "s" in df.columns else (df[by] == key)
        sub = df[mask]
    else:
        sub = df
    s = sub["action_family"].value_counts()
    s = s[s.index != "Other"].head(n)
    return s.rename_axis("action_family").reset_index(name="work_orders")


def outcome_mix(df: pd.DataFrame) -> pd.DataFrame:
    order = ["positive", "unknown", "negative", "mixed"]
    s = df["outcome"].value_counts()
    rows = [{"outcome": o, "work_orders": int(s.get(o, 0))} for o in order]
    return pd.DataFrame(rows)


def recurring_watchlist(df: pd.DataFrame, min_support: int = 5, n: int = 40) -> pd.DataFrame:
    """Chronic-defect register: recurring problem clusters, most-frequent first."""
    rows = []
    for cid, sub in df.groupby("cluster_id"):
        if len(sub) < min_support:
            continue
        pos = int((sub["outcome"] == "positive").sum())
        neg = int((sub["outcome"].isin(["negative", "mixed"])).sum())
        rows.append({
            "cluster_id": cid,
            "problem": sub["problem"].iloc[0],
            "component": sub["component"].mode().iloc[0] if not sub["component"].mode().empty else "",
            "fault": sub["fault"].mode().iloc[0] if not sub["fault"].mode().empty else "",
            "work_orders": len(sub),
            "dominant_action": sub["action_family"].mode().iloc[0] if not sub["action_family"].mode().empty else "",
            "positive_outcomes": pos,
            "negative_outcomes": neg,
            "outcome_unknown": len(sub) - pos - neg,
        })
    out = pd.DataFrame(rows).sort_values("work_orders", ascending=False).head(n).reset_index(drop=True)
    return out

