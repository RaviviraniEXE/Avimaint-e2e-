"""Leakage-safe retrieval and uncertainty evaluation used by RQ5 and RQ6.

The query representation is built only from PROBLEM. ACTION is accessed only
after ranking to derive the recorded action-family evaluation label. MaintIE is
never loaded by this module.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core.corpus import load_corpus
from core.retrieval import Retriever

DASHBOARD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_ROOT.parents[2]
SPLIT_PATH = PROJECT_ROOT / "data/splits/recommender_evaluation_split_v1.json"

SYSTEM_FILES = {
    "raw": DASHBOARD_ROOT / "data/Aircraft_Annotation_DataFile.csv",
    "rules_then_byt5": DASHBOARD_ROOT / "data/dashboard_dataset_D.csv",
}

REPRESENTATIONS = {
    "bm25": {"bm25": 1.0, "word": 0.0, "char": 0.0, "struct": 0.0},
    "tfidf": {"bm25": 0.0, "word": 0.65, "char": 0.35, "struct": 0.0},
    "structure": {"bm25": 0.0, "word": 0.0, "char": 0.0, "struct": 1.0},
    "hybrid": {"bm25": 0.34, "word": 0.24, "char": 0.14, "struct": 0.28},
}


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def load_system(system: str) -> pd.DataFrame:
    if system not in SYSTEM_FILES:
        raise ValueError(f"Unknown system {system!r}; choose {sorted(SYSTEM_FILES)}")
    corpus = load_corpus(SYSTEM_FILES[system], normalize=False, use_cache=False)
    frame = corpus.df.copy()
    frame["ident"] = frame["ident"].astype(str)
    split = load_split()
    frame["cluster_id"] = frame["ident"].map(split["case_to_cluster"])
    frame["frozen_split"] = frame["ident"].map(split["case_to_split"])
    if frame[["cluster_id", "frozen_split"]].isna().any().any():
        raise ValueError(
            "Dashboard corpus and frozen recommender split contain different identifiers"
        )
    return frame


def _rank_families(candidate_df: pd.DataFrame, hits, query_cluster: str):
    eligible = []
    family_meta: dict[str, dict[str, Any]] = {}
    for hit in hits:
        row = candidate_df.iloc[hit.idx]
        if str(row["cluster_id"]) == query_cluster:
            continue
        eligible.append(hit)
        family = str(row["action_family"])
        if family == "Other" or row["outcome"] in ("negative", "mixed"):
            continue
        meta = family_meta.setdefault(family, {"clusters": set(), "score": 0.0})
        meta["clusters"].add(str(row["cluster_id"]))
        meta["score"] += float(hit.score)
    ranked = sorted(
        family_meta,
        key=lambda family: (len(family_meta[family]["clusters"]), family_meta[family]["score"]),
        reverse=True,
    )
    return eligible, ranked, family_meta


def evaluate_partition(
    candidate_df: pd.DataFrame,
    query_df: pd.DataFrame,
    representation: str,
    top_k: int = 25,
) -> tuple[dict[str, Any], pd.DataFrame]:
    weights = REPRESENTATIONS[representation]
    candidates = candidate_df.reset_index(drop=True)
    retriever = Retriever(candidates, weights=weights)
    queries = query_df[query_df["action_family"] != "Other"].reset_index(drop=True)
    largest_cluster = int(candidates.groupby("cluster_id").size().max())
    rows: list[dict[str, Any]] = []

    for query in queries.itertuples(index=False):
        problem = str(query.problem_norm or query.problem)
        components = list(query.components)
        faults = list(query.faults)
        entity_types = list(getattr(query, "problem_entity_types", []) or [])
        relation_types = list(getattr(query, "problem_relation_types", []) or [])
        hits = retriever.search(
            problem,
            components,
            faults,
            top_k=min(len(candidates), top_k + largest_cluster),
            q_entity_types=entity_types,
            q_relation_types=relation_types,
        )
        eligible, ranked, meta = _rank_families(candidates, hits, str(query.cluster_id))
        true_family = str(query.action_family)
        predicted = ranked[0] if ranked else ""
        family_rank = ranked.index(true_family) + 1 if true_family in ranked else 0
        top_scores = [float(hit.score) for hit in eligible[:2]]
        top_score = top_scores[0] if top_scores else 0.0
        margin = top_score - (top_scores[1] if len(top_scores) > 1 else 0.0)
        support = len(meta[predicted]["clusters"]) if predicted else 0
        top_hit = eligible[0] if eligible else None
        channel_agreement = bool(top_hit and top_hit.text_sim > 0 and top_hit.struct > 0)
        structure_coverage = len(components) + len(faults) + len(entity_types) + len(relation_types)
        unique_top10 = {
            str(candidates.iloc[hit.idx]["action_family"])
            for hit in eligible[:10]
            if str(candidates.iloc[hit.idx]["action_family"]) != "Other"
        }
        rows.append(
            {
                "ident": str(query.ident),
                "cluster_id": str(query.cluster_id),
                "true_family": true_family,
                "predicted_family": predicted,
                "top1_correct": predicted == true_family,
                "top3_correct": bool(family_rank and family_rank <= 3),
                "family_rank": family_rank,
                "reciprocal_rank": 1.0 / family_rank if family_rank else 0.0,
                "ndcg_at_3": (1.0 / math.log2(family_rank + 1)) if 0 < family_rank <= 3 else 0.0,
                "top_score": top_score,
                "top1_top2_margin": margin,
                "supporting_cluster_count": support,
                "lexical_structure_agreement": channel_agreement,
                "extracted_information_coverage": structure_coverage,
                "candidate_family_diversity_at_10": len(unique_top10),
            }
        )

    result = pd.DataFrame(rows)
    families = sorted(result["true_family"].unique()) if len(result) else []
    per_family = {
        family: {
            "n": int((result["true_family"] == family).sum()),
            "recall_at_1": float(
                result.loc[result["true_family"] == family, "top1_correct"].mean()
            ),
        }
        for family in families
    }
    metrics = {
        "queries": int(len(result)),
        "candidate_cases": int(len(candidates)),
        "representation": representation,
        "top1_action_family_agreement": float(result["top1_correct"].mean())
        if len(result)
        else 0.0,
        "top3_action_family_agreement": float(result["top3_correct"].mean())
        if len(result)
        else 0.0,
        "mrr": float(result["reciprocal_rank"].mean()) if len(result) else 0.0,
        "ndcg_at_3": float(result["ndcg_at_3"].mean()) if len(result) else 0.0,
        "macro_action_family_recall": float(
            np.mean([item["recall_at_1"] for item in per_family.values()])
        )
        if per_family
        else 0.0,
        "mean_candidate_family_diversity_at_10": float(
            result["candidate_family_diversity_at_10"].mean()
        )
        if len(result)
        else 0.0,
        "per_family": per_family,
        "label_distribution": dict(Counter(result["true_family"])) if len(result) else {},
        "interpretation": "agreement with recorded action family, not technical correctness",
        "query_uses_action_text": False,
    }
    return metrics, result


def protocol_frames(frame: pd.DataFrame, protocol: str, partition: str = "test"):
    if protocol == "loo":
        return frame, frame
    if protocol != "frozen":
        raise ValueError("protocol must be 'loo' or 'frozen'")
    candidates = frame[frame["frozen_split"] == "train"]
    queries = frame[frame["frozen_split"] == partition]
    return candidates, queries


def calibration_features(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        "top_score",
        "top1_top2_margin",
        "supporting_cluster_count",
        "lexical_structure_agreement",
        "extracted_information_coverage",
    ]
    return frame[columns].astype(float).to_numpy()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def risk_coverage(y: np.ndarray, p: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(-p)
    ranked = y[order]
    points = []
    for coverage in np.linspace(0.1, 1.0, 10):
        n = max(1, int(math.ceil(len(ranked) * coverage)))
        points.append(
            {
                "coverage": float(coverage),
                "accuracy": float(ranked[:n].mean()),
                "risk": float(1.0 - ranked[:n].mean()),
            }
        )
    return points


def confidence_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Return transparent confidence-bin support, accuracy, and error rate."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if not mask.any():
            continue
        accuracy = float(y[mask].mean())
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n": int(mask.sum()),
                "mean_confidence": float(p[mask].mean()),
                "accuracy": accuracy,
                "error_rate": float(1.0 - accuracy),
            }
        )
    return rows
