"""Entity and relation evaluation: micro/macro P/R/F1 and per-class F1.

Entity match = exact span + type. Relation match = (head span+type, tail span+type,
relation type). Works on lists of gold/pred docs with {tokens, entities, relations}.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List


def _ent_key(e):
    return (e["start"], e["end"], e["type"])


def _rel_key(ents, r):
    h, t = ents[r["head"]], ents[r["tail"]]
    return (_ent_key(h), _ent_key(t), r["type"])


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def entity_scores(gold: List[dict], pred: List[dict]) -> Dict:
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])  # type -> [tp, fp, fn]
    for g, p in zip(gold, pred):
        gset = {_ent_key(e) for e in g["entities"]}
        pset = {_ent_key(e) for e in p["entities"]}
        for k in pset & gset:
            tp += 1; per[k[2]][0] += 1
        for k in pset - gset:
            fp += 1; per[k[2]][1] += 1
        for k in gset - pset:
            fn += 1; per[k[2]][2] += 1
    micro = _prf(tp, fp, fn)
    per_class = {t: _prf(*c) for t, c in per.items()}
    macro_f1 = round(sum(v[2] for v in per_class.values()) / len(per_class), 4) if per_class else 0.0
    return {"micro_p": micro[0], "micro_r": micro[1], "micro_f1": micro[2],
            "macro_f1": macro_f1, "per_class": per_class, "support": tp + fn}


def bootstrap_ci(gold: List[dict], pred: List[dict], kind: str = "entity",
                 n: int = 1000, seed: int = 42, alpha: float = 0.05) -> Dict:
    """Bootstrap confidence interval for micro-F1: resample the (gold,pred) doc
    pairs with replacement `n` times and take the central (1-alpha) range. Answers
    'how much would this F1 wobble on a different test sample of the same size?'."""
    fn = entity_scores if kind == "entity" else relation_scores
    point = fn(gold, pred)["micro_f1"]
    rng = random.Random(seed)
    N = len(gold)
    if N == 0 or n <= 0:
        return {"point": round(point, 4), "lo": round(point, 4), "hi": round(point, 4), "n": n}
    vals = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        vals.append(fn([gold[i] for i in idx], [pred[i] for i in idx])["micro_f1"])
    vals.sort()
    lo = vals[int((alpha / 2) * n)]
    hi = vals[min(int((1 - alpha / 2) * n), n - 1)]
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}


def paired_bootstrap(gold, pred_a, pred_b, kind="entity", n=1000, seed=42) -> Dict:
    """Paired bootstrap significance test: is model A's micro-F1 really higher than
    B's, or within noise? Resamples the SAME test docs for both models n times and
    looks at the F1 difference. Returns the difference, its 95% CI, a p-value
    (fraction of resamples where A does NOT beat B), and whether it's significant
    (the difference CI excludes 0)."""
    fn = entity_scores if kind == "entity" else relation_scores
    da, db = fn(gold, pred_a)["micro_f1"], fn(gold, pred_b)["micro_f1"]
    rng = random.Random(seed)
    N = len(gold)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        g = [gold[i] for i in idx]
        diffs.append(fn(g, [pred_a[i] for i in idx])["micro_f1"]
                     - fn(g, [pred_b[i] for i in idx])["micro_f1"])
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[min(int(0.975 * n), n - 1)]
    p = sum(1 for d in diffs if d <= 0) / n
    return {"a_f1": round(da, 4), "b_f1": round(db, 4), "diff": round(da - db, 4),
            "diff_lo": round(lo, 4), "diff_hi": round(hi, 4), "p_value": round(p, 4),
            "significant": bool(lo > 0 or hi < 0)}


def relation_scores(gold: List[dict], pred: List[dict]) -> Dict:
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    for g, p in zip(gold, pred):
        gset = {_rel_key(g["entities"], r) for r in g.get("relations", [])}
        pset = {_rel_key(p["entities"], r) for r in p.get("relations", [])}
        for k in pset & gset:
            tp += 1; per[k[2]][0] += 1
        for k in pset - gset:
            fp += 1; per[k[2]][1] += 1
        for k in gset - pset:
            fn += 1; per[k[2]][2] += 1
    micro = _prf(tp, fp, fn)
    per_class = {t: _prf(*c) for t, c in per.items()}
    macro_f1 = round(sum(v[2] for v in per_class.values()) / len(per_class), 4) if per_class else 0.0
    return {"micro_p": micro[0], "micro_r": micro[1], "micro_f1": micro[2],
            "macro_f1": macro_f1, "per_class": per_class, "support": tp + fn}

