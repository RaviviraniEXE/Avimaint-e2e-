"""Load corrected gold annotations and make leak-free splits.

Gold records are JSONL with {ident, tokens, bio, entities, relations,
exact_group_id?}. Splitting is grouped by duplicate cluster so identical /
near-identical records never straddle train and test.
"""
from __future__ import annotations

import glob
import json
import random
from typing import List


def load_gold(paths_glob: str) -> List[dict]:
    recs = []
    for f in sorted(glob.glob(paths_glob)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def grouped_split(recs, train=0.7, dev=0.15, seed=42, group_key="exact_group_id"):
    """Split by group so duplicates stay together. Falls back to ident if no group."""
    groups = {}
    for r in recs:
        g = r.get(group_key) or r.get("ident")
        groups.setdefault(g, []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_tr, n_dv = int(round(n * train)), int(round(n * dev))
    parts = {"train": keys[:n_tr], "dev": keys[n_tr:n_tr + n_dv], "test": keys[n_tr + n_dv:]}
    out = {k: [r for g in ks for r in groups[g]] for k, ks in parts.items()}
    return out["train"], out["dev"], out["test"]

