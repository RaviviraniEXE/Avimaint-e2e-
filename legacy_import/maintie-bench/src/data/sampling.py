"""Sampling for the annotation plan.

The PILOT is a pure RANDOM sample from the deduplicated unique pool (unbiased —
no cue-based selection). Rare-class enrichment happens later via active learning
on the trained model's predictions (src/data/active_learning.py), not here.

The helper detectors (is_truncated, n_actions) are used only for *reporting* the
composition of the random sample, never for selecting it.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.lexicons import SINGLE, has_cue


def is_truncated(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t[-1] not in ".!?)\"'"


def n_actions(text: str) -> int:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return sum(1 for w in toks if w in SINGLE["ACTION"])


def random_sample(pool: pd.DataFrame, n: int, seed: int, exclude=None):
    """Random n records from the unique pool, excluding a set of IDENTs."""
    p = pool
    if exclude:
        p = p[~p["IDENT"].isin(set(exclude))]
    n = min(n, len(p))
    return p.sample(n=n, random_state=seed).reset_index(drop=True)


def describe(sample: pd.DataFrame, text_col="text", problem_col="PROBLEM", action_col="ACTION"):
    """Composition of a (random) sample — for the report only."""
    n = len(sample)
    d = {"n": n}
    d["truncated"] = int(sum(is_truncated(p) or is_truncated(a)
                             for p, a in zip(sample[problem_col], sample[action_col])))
    d["multi_action"] = int(sum(n_actions(t) >= 2 for t in sample[text_col]))
    for ent in ("TECH_OBS", "OP_CTX", "OUTCOME", "REFERENCE"):
        d[f"has_{ent}"] = int(sum(has_cue(t, ent) for t in sample[text_col]))
    return d

