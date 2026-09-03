"""Freeze a random, grouped test/dev split — done ONCE, early, before any
active-learning enrichment, so the held-out sets reflect the true distribution.

Grouping is by duplicate cluster (exact_group_id) so identical records never
straddle splits. Frozen assignments are stored in outputs/splits.json and reused.
"""
from __future__ import annotations

import json
import os
import random

SPLITS_FILE = "outputs/splits.json"


def freeze(gold_records, test_n, dev_n, seed, path=SPLITS_FILE):
    """Assign whole duplicate groups to test/dev until sizes are reached; the
    rest are train. Writes {ident: split} to `path`. Idempotent-ish: if the file
    exists, it is extended (new records default to train), never reshuffled."""
    if os.path.exists(path):
        existing = json.load(open(path))
    else:
        existing = {"test": [], "dev": [], "train": [], "seed": seed}

    assigned = set(existing["test"]) | set(existing["dev"]) | set(existing["train"])
    # group unassigned records
    groups = {}
    for r in gold_records:
        if r["ident"] in assigned:
            continue
        g = r.get("exact_group_id") or r["ident"]
        groups.setdefault(g, []).append(r["ident"])

    if not existing["test"] and not existing["dev"]:   # first freeze
        keys = list(groups)
        random.Random(seed).shuffle(keys)
        test, dev = [], []
        for k in keys:
            if len(test) < test_n:
                test += groups[k]
            elif len(dev) < dev_n:
                dev += groups[k]
            else:
                existing["train"] += groups[k]
        existing["test"], existing["dev"] = test, dev
    else:                                              # later rounds -> all train
        for k, idents in groups.items():
            existing["train"] += idents

    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(existing, open(path, "w"), indent=1)
    return existing


def load_splits(path=SPLITS_FILE):
    return json.load(open(path)) if os.path.exists(path) else None


def assign(gold_records, path=SPLITS_FILE):
    """Return (train, dev, test) lists of records per the frozen split."""
    sp = load_splits(path)
    if not sp:
        return gold_records, [], []
    idx = {"train": [], "dev": [], "test": []}
    lut = {}
    for k in ("test", "dev", "train"):
        for i in sp[k]:
            lut[i] = k
    for r in gold_records:
        idx[lut.get(r["ident"], "train")].append(r)
    return idx["train"], idx["dev"], idx["test"]

