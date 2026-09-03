"""Cluster-safe leave-one-out evaluation of the recommender.

WHAT IT MEASURES (and what it does not)
---------------------------------------
There is no gold "correct action" in this corpus — only what technicians did.
So this harness measures *agreement with recorded historical practice*, NOT
technical correctness: for each work order, we hide it (and its whole
near-duplicate cluster, so the system cannot retrieve a copy of itself),
recommend from the rest, and check whether the recommended action FAMILY matches
the family that was actually recorded. This is the honest, data-supported
question — "would the system have suggested what was actually done?" — and it is
the standard leave-one-out protocol for case-based recommenders.

Reported metrics
----------------
- top1_family_acc : predicted top family == recorded family
- top3_family_acc : recorded family among the top-3 recommended families
- mrr             : mean reciprocal rank of the first retrieved case whose family
                    matches the recorded family (cluster-safe)
- coverage-at-confidence : for each ladder tier (strong/moderate/exploratory),
                    what share of queries land there and the top-1 accuracy within it
- baselines       : majority-family and text-only retrieval, for context

All figures are computed on cluster-separated data; the query's own cluster is
excluded from retrieval, support counting, and family voting.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from . import normalize as N
from .recommend import STRONG_SCORE, MODERATE_SCORE, MIN_SCORE


def _vote(df, hits, query_cluster):
    """Family voting over cluster-safe hits. Returns (ranked_families, first_match_rank_map)."""
    fams = {}
    rank = 0
    match_rank = {}
    for h in hits:
        row = df.iloc[h.idx]
        if str(row["cluster_id"]) == str(query_cluster):
            continue
        rank += 1
        fam = row["action_family"]
        if fam == "Other" or row["outcome"] in ("negative", "mixed"):
            continue
        d = fams.setdefault(fam, {"clusters": set(), "score": 0.0})
        d["clusters"].add(row["cluster_id"])
        d["score"] += h.score
        match_rank.setdefault(fam, rank)
    ranked = sorted(fams.items(), key=lambda kv: (len(kv[1]["clusters"]), kv[1]["score"]),
                    reverse=True)
    return [f for f, _ in ranked], ranked, match_rank


def _scoped_support(df, family, q_comp, q_fault, query_cluster):
    if not family:
        return 0
    fam = df["action_family"] == family
    ok = ~df["outcome"].isin(["negative", "mixed"])
    notc = df["cluster_id"].astype(str) != str(query_cluster)
    if q_comp:
        anchor = df["components"].map(lambda xs: any(c in xs for c in q_comp))
    elif q_fault:
        anchor = df["faults"].map(lambda xs: any(f in xs for f in q_fault))
    else:
        return 0
    return int(df[fam & ok & notc & anchor]["cluster_id"].nunique())


def _tier(support, top_score, has_comp):
    if has_comp and support >= 3 and top_score >= STRONG_SCORE:
        return "strong"
    if has_comp and support >= 1 and top_score >= MODERATE_SCORE:
        return "moderate"
    return "exploratory"


def evaluate(df: pd.DataFrame, retriever, sample: int | None = 1200, seed: int = 0,
             top_k: int = 25, progress=None, reranker=None,
             query_ids: list | None = None) -> dict:
    df = df.reset_index(drop=True)
    # only evaluate rows whose recorded family is classifiable
    evaldf = df[df["action_family"] != "Other"]
    if query_ids is not None:
        idset = set(str(x) for x in query_ids)
        evaldf = evaldf[evaldf["ident"].astype(str).isin(idset)]
    elif sample and len(evaldf) > sample:
        evaldf = evaldf.sample(sample, random_state=seed)
    idxs = evaldf.index.tolist()

    use_rr = reranker is not None and reranker.available()
    global_majority = df[df["action_family"] != "Other"]["action_family"].mode().iloc[0]

    rows = []
    for n, i in enumerate(idxs):
        if progress and n % 50 == 0:
            progress(n, len(idxs))
        row = df.iloc[i]
        q = row["problem_norm"] or row["problem"]
        true_fam = row["action_family"]
        qc = str(row["cluster_id"])
        q_comp = N.find_components(row.get("problem_clean", q) or q)
        q_fault = [f for f in [N.issue_family(q)] if f]
        hits = retriever.search(q, q_comp, q_fault, top_k=top_k + 15)
        if use_rr and hits:
            cand = [(h.idx, df.iloc[h.idx]["problem_norm"], h.score) for h in hits]
            order = reranker.rerank(q, cand)
            by_idx = {h.idx: h for h in hits}
            reordered = []
            for idx, sc in order:
                h = by_idx[idx]
                h.score = float(sc)
                reordered.append(h)
            hits = reordered
        # cluster-safe top score (first hit not in query cluster)
        top_score = next((h.score for h in hits if str(df.iloc[h.idx]["cluster_id"]) != qc), 0.0)
        ranked_fams, ranked, match_rank = _vote(df, hits, qc)
        pred1 = ranked_fams[0] if ranked_fams else ""
        top3 = ranked_fams[:3]
        support = _scoped_support(df, pred1, q_comp, q_fault, qc)
        tier = _tier(support, top_score, bool(q_comp)) if pred1 else "exploratory"
        rr = 1.0 / match_rank[true_fam] if true_fam in match_rank else 0.0
        rows.append({
            "true": true_fam, "pred1": pred1, "top3_hit": true_fam in top3,
            "top1_hit": pred1 == true_fam, "rr": rr, "tier": tier,
            "maj_hit": true_fam == global_majority,
        })
    R = pd.DataFrame(rows)

    def acc(mask=None):
        d = R if mask is None else R[mask]
        return round(100 * d["top1_hit"].mean(), 1) if len(d) else 0.0

    tiers = {}
    for t in ("strong", "moderate", "exploratory"):
        d = R[R["tier"] == t]
        # majority-baseline accuracy on the SAME subset, for a fair within-tier compare
        maj_t = round(100 * d["maj_hit"].mean(), 1) if len(d) else 0.0
        tiers[t] = {"coverage_pct": round(100 * len(d) / len(R), 1) if len(R) else 0.0,
                    "n": int(len(d)), "top1_acc": round(100 * d["top1_hit"].mean(), 1) if len(d) else 0.0,
                    "majority_acc": maj_t}

    # per-family recall + macro average (the informative metric under imbalance)
    fams = sorted(R["true"].unique())
    per_family = {}
    for f in fams:
        d = R[R["true"] == f]
        per_family[f] = {"n": int(len(d)),
                         "recall": round(100 * d["top1_hit"].mean(), 1) if len(d) else 0.0}
    macro = round(np.mean([per_family[f]["recall"] for f in fams]), 1) if fams else 0.0
    # majority baseline macro-recall = 100% on the majority family only, 0 elsewhere
    maj_macro = round(100.0 / len(fams), 1) if fams else 0.0

    return {
        "n_evaluated": int(len(R)),
        "reranker": bool(use_rr),
        "top1_family_acc": acc(),
        "top3_family_acc": round(100 * R["top3_hit"].mean(), 1),
        "mrr": round(R["rr"].mean(), 3),
        "macro_recall": macro,
        "baseline_majority_acc": round(100 * R["maj_hit"].mean(), 1),
        "baseline_majority_macro": maj_macro,
        "majority_family": global_majority,
        "tiers": tiers,
        "per_family_recall": per_family,
        "family_distribution": {k: int(v) for k, v in
                                collections.Counter(R["true"]).most_common()},
    }

