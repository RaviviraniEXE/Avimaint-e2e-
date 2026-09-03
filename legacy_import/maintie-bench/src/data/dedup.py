"""Deduplication: exact-pair and near-duplicate grouping.

Avi-Main has ~17% exact-duplicate problem-action pairs. Duplicates must be
grouped so that (a) the pilot samples *unique* patterns, and (b) all members of
a group are kept in the same train/val/test split later (no leakage).
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd


def _canon(text: str) -> str:
    """Aggressive canonical key for near-duplicate grouping: lowercase, strip
    positions/indices/numbers and punctuation, collapse whitespace."""
    t = (text or "").lower()
    t = re.sub(r"#\s?\d+[a-z]?", " ", t)          # #2, #4a
    t = re.sub(r"\bnumber \d+\b", " ", t)
    t = re.sub(r"\d+", " ", t)                      # any remaining digits
    t = re.sub(r"[^a-z ]", " ", t)                 # punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def add_duplicate_groups(df: pd.DataFrame, problem_col="PROBLEM", action_col="ACTION"):
    """Add exact_group_id and near_group_id. Returns (df, stats)."""
    df = df.copy()
    exact_key = (df[problem_col].fillna("") + " ||| " + df[action_col].fillna(""))
    df["exact_group_id"] = exact_key.map(lambda s: "ex_" + hashlib.md5(s.encode()).hexdigest()[:8])
    near_key = (df[problem_col].map(_canon) + " ||| " + df[action_col].map(_canon))
    df["near_group_id"] = near_key.map(lambda s: "nr_" + hashlib.md5(s.encode()).hexdigest()[:8])

    n = len(df)
    stats = {
        "records": n,
        "unique_exact_pairs": df["exact_group_id"].nunique(),
        "duplicate_rows": n - df["exact_group_id"].nunique(),
        "exact_dup_rate": round((n - df["exact_group_id"].nunique()) / n, 4),
        "unique_near_groups": df["near_group_id"].nunique(),
        "near_dup_rate": round((n - df["near_group_id"].nunique()) / n, 4),
    }
    return df, stats


def unique_pool(df: pd.DataFrame, by="exact_group_id") -> pd.DataFrame:
    """One representative row per duplicate group (the first by IDENT)."""
    return df.sort_values("IDENT").drop_duplicates(by, keep="first").reset_index(drop=True)

