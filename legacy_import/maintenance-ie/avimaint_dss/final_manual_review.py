from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.corpus import load_corpus
from core.retrieval import Retriever


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RUN_ROOT = REPO / "outputs" / "runs" / "rq4_case_retrieval"
MANUAL_DIR = RUN_ROOT / "manual_review"
PHASE_A = MANUAL_DIR / "PHASE_A_problem_relevance_BLINDED.csv"
PRIVATE = MANUAL_DIR / "PRIVATE_DO_NOT_OPEN_UNTIL_PHASE_A_COMPLETE.jsonl"
PHASE_B = MANUAL_DIR / "PHASE_B_action_applicability_BLINDED.csv"
RESULTS = MANUAL_DIR / "MANUAL_REVIEW_RESULTS.json"
MANIFEST = MANUAL_DIR / "MANUAL_REVIEW_MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_final_corpus() -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pred = (ROOT / cfg["data"]["problem_predictions_path"]).resolve()
    protocol = (ROOT / cfg["data"]["protocol_path"]).resolve()
    csv_path = (ROOT / cfg["data"]["csv_path"]).resolve()
    return load_corpus(
        csv_path,
        predictions_path=pred,
        protocol_path=protocol,
        require_predictions=True,
    ).df


def selected_mode() -> str:
    dev = RUN_ROOT / "dev" / "RQ4_DEV_SELECTION.json"
    lock = RUN_ROOT / "FINAL_TEST_LOCK.json"
    if not dev.exists() or not lock.exists():
        raise SystemExit("RQ4 DEV selection and locked final TEST must exist first.")
    d = json.loads(dev.read_text(encoding="utf-8"))
    l = json.loads(lock.read_text(encoding="utf-8"))
    if not l.get("locked"):
        raise SystemExit("RQ4 final TEST is not locked.")
    if d.get("selected_mode") != l.get("selected_mode"):
        raise SystemExit("DEV-selected mode and TEST lock disagree.")
    return str(d["selected_mode"])


def proportional_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic proportional sample stratified by recorded action family."""
    if len(frame) <= n:
        return frame.copy()
    counts = frame["action_family"].value_counts().sort_index()
    exact = counts / counts.sum() * n
    quotas = np.floor(exact).astype(int)
    remaining = n - int(quotas.sum())
    remainders = (exact - quotas).sort_values(ascending=False)
    for family in remainders.index[:remaining]:
        quotas.loc[family] += 1

    parts = []
    for i, (family, quota) in enumerate(quotas.items()):
        if quota <= 0:
            continue
        sub = frame[frame["action_family"] == family]
        parts.append(sub.sample(n=min(int(quota), len(sub)), random_state=seed + i))
    sampled = pd.concat(parts, ignore_index=False)
    if len(sampled) < n:
        rest = frame.drop(sampled.index)
        sampled = pd.concat(
            [sampled, rest.sample(n=n - len(sampled), random_state=seed + 10000)]
        )
    return sampled.sample(frac=1.0, random_state=seed + 20000).head(n)


def build_pool(query_count: int, top_k: int, seed: int) -> None:
    if query_count < 1 or top_k < 1:
        raise SystemExit("--queries and --top-k must be positive.")

    mode = selected_mode()
    df = load_final_corpus()
    train = df[df["frozen_split"] == "train"].reset_index(drop=True)
    test = df[(df["frozen_split"] == "test") & (df["action_family"] != "Other")].copy()
    if test.empty:
        raise SystemExit("No eligible TEST queries found.")

    sample = proportional_sample(test, min(query_count, len(test)), seed)
    retriever = Retriever(train)
    private_rows = []
    phase_a_rows = []

    for q in sample.itertuples(index=False):
        hits = retriever.search(
            q.problem_norm,
            q.components,
            q.faults,
            top_k=top_k,
            q_entity_types=q.problem_entity_types,
            q_relation_types=q.problem_relation_types,
            raw_query=q.problem,
            mode=mode,
            exclude_groups={str(q.leakage_group_id)},
            diversify=True,
        )
        for rank, hit in enumerate(hits, start=1):
            c = train.iloc[hit.idx]
            pair_seed = (
                f"{q.ident}|{c.ident}|{mode}|{rank}|rq4_manual_v1"
            )
            pair_id = "MR-" + hashlib.sha256(pair_seed.encode("utf-8")).hexdigest()[:16]

            phase_a_rows.append(
                {
                    "pair_id": pair_id,
                    "query_id": str(q.ident),
                    "query_problem": str(q.problem),
                    "candidate_problem": str(c.problem),
                    "problem_relevance_0_1_2": "",
                    "phase_a_notes": "",
                }
            )
            private_rows.append(
                {
                    "pair_id": pair_id,
                    "query_id": str(q.ident),
                    "candidate_id": str(c.ident),
                    "rank": rank,
                    "selected_mode": mode,
                    "query_true_historical_action_family": str(q.action_family),
                    "candidate_historical_action": str(c.action),
                    "candidate_historical_action_family": str(c.action_family),
                    "candidate_problem": str(c.problem),
                    "candidate_cluster_id": str(c.cluster_id),
                    "candidate_leakage_group_id": str(c.leakage_group_id),
                    "retrieval_score": float(hit.score),
                }
            )

    if not phase_a_rows:
        raise SystemExit("No manual-review candidates were produced.")

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.DataFrame(phase_a_rows)
    # Blind rank/order: preserve pair_id mapping privately but randomize visible row order.
    a = a.sample(frac=1.0, random_state=seed + 30000).reset_index(drop=True)
    a.to_csv(PHASE_A, index=False)

    with PRIVATE.open("w", encoding="utf-8") as f:
        for row in private_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_counts = sample["action_family"].value_counts().sort_index().to_dict()
    manifest = {
        "version": "manual_review_v1",
        "status": "phase_a_ready",
        "selected_retrieval_mode": mode,
        "source_partition": "locked TEST",
        "query_count": int(sample["ident"].nunique()),
        "candidate_depth": int(top_k),
        "visible_pair_count": int(len(a)),
        "sampling": "deterministic proportional stratification by recorded action family",
        "seed": int(seed),
        "query_family_counts": {str(k): int(v) for k, v in family_counts.items()},
        "blinding": {
            "phase_a_hides_actions": True,
            "phase_a_hides_rank": True,
            "phase_a_hides_scores": True,
            "phase_a_hides_candidate_action_family": True,
        },
        "rating_scale_phase_a": {
            "0": "not relevant",
            "1": "partially relevant / plausibly comparable",
            "2": "clearly relevant / strongly comparable",
        },
        "interpretation": (
            "Manual review evaluates relevance of retrieved historical evidence. "
            "It does not establish technical correctness, safety, or regulatory applicability."
        ),
        "phase_a_file": str(PHASE_A),
        "private_mapping_file": str(PRIVATE),
        "phase_a_sha256": sha256(PHASE_A),
        "private_mapping_sha256": sha256(PRIVATE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 82)
    print("MANUAL REVIEW PHASE A CREATED")
    print("=" * 82)
    print("selected retrieval mode :", mode)
    print("sampled TEST queries    :", sample["ident"].nunique())
    print("candidate depth          :", top_k)
    print("blinded problem pairs    :", len(a))
    print("PHASE A                  :", PHASE_A)
    print("PRIVATE MAPPING          :", PRIVATE)
    print("")
    print("Fill ONLY: problem_relevance_0_1_2 and optional phase_a_notes.")
    print("Do NOT open the PRIVATE mapping before Phase A is complete.")
    print("Then run FINAL_09_SCORE_MANUAL_REVIEW_AFTER_FILLING.bat.")
    print("=" * 82)


def read_private() -> pd.DataFrame:
    if not PRIVATE.exists():
        raise SystemExit("Private mapping missing. Run FINAL_08 first.")
    rows = [
        json.loads(line)
        for line in PRIVATE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def parse_rating(series: pd.Series, column: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    bad = out.isna() | ~out.isin([0, 1, 2])
    if bad.any():
        examples = series[bad].head(10).tolist()
        raise SystemExit(
            f"{column} is incomplete/invalid for {int(bad.sum())} rows. "
            f"Allowed values are 0, 1, 2. Examples: {examples}"
        )
    return out.astype(int)


def reveal_phase_b() -> None:
    if not PHASE_A.exists():
        raise SystemExit("Phase A file missing. Run FINAL_08 first.")
    a = pd.read_csv(PHASE_A, dtype=str, keep_default_na=False)
    a["problem_relevance_0_1_2"] = parse_rating(
        a["problem_relevance_0_1_2"], "problem_relevance_0_1_2"
    )
    private = read_private()
    merged = a.merge(private, on=["pair_id", "query_id"], how="left", validate="one_to_one")
    if merged["candidate_id"].isna().any():
        raise SystemExit("Phase A/private mapping mismatch.")

    relevant = merged[merged["problem_relevance_0_1_2"] >= 1].copy()
    phase_b = relevant[
        [
            "pair_id",
            "query_id",
            "query_problem",
            "candidate_problem_x",
            "problem_relevance_0_1_2",
            "candidate_historical_action",
            "candidate_historical_action_family",
        ]
    ].rename(columns={"candidate_problem_x": "candidate_problem"})
    phase_b["action_applicability_0_1_2"] = ""
    phase_b["phase_b_notes"] = ""
    phase_b = phase_b.sample(frac=1.0, random_state=424242).reset_index(drop=True)
    phase_b.to_csv(PHASE_B, index=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "phase_b_ready",
            "phase_a_completed_sha256": sha256(PHASE_A),
            "phase_b_rows": int(len(phase_b)),
            "phase_b_file": str(PHASE_B),
            "phase_b_sha256_at_creation": sha256(PHASE_B),
            "rating_scale_phase_b": {
                "0": "not useful historical action evidence for this query",
                "1": "partially useful / conditionally applicable historical evidence",
                "2": "clearly useful historical action evidence",
            },
            "phase_b_interpretation": (
                "Applicability means usefulness as historical planning evidence only; "
                "it is not a technical approval or safety judgment."
            ),
        }
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 82)
    print("PHASE A COMPLETE - PHASE B REVEALED")
    print("=" * 82)
    print("Phase A rows reviewed    :", len(a))
    print("problem-relevant pairs   :", len(phase_b))
    print("PHASE B                  :", PHASE_B)
    print("")
    print("Fill ONLY: action_applicability_0_1_2 and optional phase_b_notes.")
    print("Then run FINAL_09_SCORE_MANUAL_REVIEW_AFTER_FILLING.bat again.")
    print("=" * 82)


def ndcg_for_query(group: pd.DataFrame, k: int = 5) -> float:
    g = group.sort_values("rank").head(k)
    gains = g["problem_relevance_0_1_2"].astype(float).to_numpy()
    dcg = sum((2.0 ** gain - 1.0) / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = np.sort(gains)[::-1]
    idcg = sum((2.0 ** gain - 1.0) / math.log2(i + 2) for i, gain in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def score_complete() -> None:
    a = pd.read_csv(PHASE_A, dtype=str, keep_default_na=False)
    a["problem_relevance_0_1_2"] = parse_rating(
        a["problem_relevance_0_1_2"], "problem_relevance_0_1_2"
    )
    b = pd.read_csv(PHASE_B, dtype=str, keep_default_na=False)
    b["action_applicability_0_1_2"] = parse_rating(
        b["action_applicability_0_1_2"], "action_applicability_0_1_2"
    )
    private = read_private()

    aa = a.merge(private, on=["pair_id", "query_id"], how="left", validate="one_to_one")
    bb = b[
        ["pair_id", "action_applicability_0_1_2", "phase_b_notes"]
    ].copy()
    full = aa.merge(bb, on="pair_id", how="left", validate="one_to_one")
    # Irrelevant problem pairs intentionally receive action applicability 0.
    full["action_applicability_0_1_2"] = pd.to_numeric(
        full["action_applicability_0_1_2"], errors="coerce"
    ).fillna(0).astype(int)

    per_query = []
    for qid, g in full.groupby("query_id"):
        g = g.sort_values("rank")
        rel = (g["problem_relevance_0_1_2"] >= 1).astype(int).to_numpy()
        useful = (
            (g["problem_relevance_0_1_2"] >= 1)
            & (g["action_applicability_0_1_2"] >= 1)
        ).astype(int).to_numpy()
        per_query.append(
            {
                "query_id": qid,
                "problem_relevant_at_1": int(rel[:1].max()) if len(rel) else 0,
                "problem_relevant_at_3": int(rel[:3].max()) if len(rel) else 0,
                "problem_relevant_at_5": int(rel[:5].max()) if len(rel) else 0,
                "usable_action_evidence_at_1": int(useful[:1].max()) if len(useful) else 0,
                "usable_action_evidence_at_3": int(useful[:3].max()) if len(useful) else 0,
                "usable_action_evidence_at_5": int(useful[:5].max()) if len(useful) else 0,
                "ndcg_problem_relevance_at_5": ndcg_for_query(g, 5),
            }
        )

    pq = pd.DataFrame(per_query)
    phase_a_relevant = full["problem_relevance_0_1_2"] >= 1
    phase_b_applicable = full["action_applicability_0_1_2"] >= 1
    phase_b_clear = full["action_applicability_0_1_2"] == 2

    metrics = {
        "version": "manual_review_v1",
        "selected_retrieval_mode": selected_mode(),
        "reviewed_queries": int(pq["query_id"].nunique()),
        "reviewed_pairs": int(len(full)),
        "phase_a": {
            "problem_relevance_hit_at_1": float(pq["problem_relevant_at_1"].mean()),
            "problem_relevance_hit_at_3": float(pq["problem_relevant_at_3"].mean()),
            "problem_relevance_hit_at_5": float(pq["problem_relevant_at_5"].mean()),
            "mean_ndcg_problem_relevance_at_5": float(
                pq["ndcg_problem_relevance_at_5"].mean()
            ),
            "pairwise_relevant_or_partial_rate": float(phase_a_relevant.mean()),
            "clearly_relevant_pair_rate": float(
                (full["problem_relevance_0_1_2"] == 2).mean()
            ),
        },
        "phase_b": {
            "problem_relevant_pairs_revealed": int(phase_a_relevant.sum()),
            "applicable_or_partial_rate_among_problem_relevant_pairs": float(
                phase_b_applicable[phase_a_relevant].mean()
            )
            if phase_a_relevant.any()
            else None,
            "clearly_applicable_rate_among_problem_relevant_pairs": float(
                phase_b_clear[phase_a_relevant].mean()
            )
            if phase_a_relevant.any()
            else None,
            "usable_historical_action_evidence_hit_at_1": float(
                pq["usable_action_evidence_at_1"].mean()
            ),
            "usable_historical_action_evidence_hit_at_3": float(
                pq["usable_action_evidence_at_3"].mean()
            ),
            "usable_historical_action_evidence_hit_at_5": float(
                pq["usable_action_evidence_at_5"].mean()
            ),
        },
        "interpretation": (
            "Manual judgments concern relevance and usefulness of historical evidence. "
            "They do NOT validate technical correctness, aircraft-specific applicability, "
            "safety, or regulatory compliance."
        ),
        "single_reviewer_limitation": (
            "If only one reviewer is used, inter-rater agreement cannot be estimated."
        ),
    }

    full.to_csv(MANUAL_DIR / "manual_review_scored_pairs.csv", index=False)
    pq.to_csv(MANUAL_DIR / "manual_review_per_query.csv", index=False)
    RESULTS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "phase_a_final_sha256": sha256(PHASE_A),
            "phase_b_final_sha256": sha256(PHASE_B),
            "results_file": str(RESULTS),
            "results_sha256": sha256(RESULTS),
        }
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 82)
    print("MANUAL REVIEW COMPLETE")
    print("=" * 82)
    print(json.dumps(metrics, indent=2))
    print("RESULTS:", RESULTS)
    print("")
    print("Now FINAL_10_FREEZE_RQ4_RQ5.bat will include this manual-review folder.")
    print("=" * 82)


def advance() -> None:
    if not PHASE_A.exists():
        raise SystemExit("Phase A does not exist. Run FINAL_08 first.")
    if not PHASE_B.exists():
        reveal_phase_b()
        return
    score_complete()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blinded two-stage manual review for the final RQ4/RQ5 system."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--queries", type=int, default=100)
    b.add_argument("--top-k", type=int, default=5)
    b.add_argument("--seed", type=int, default=20260901)

    sub.add_parser("advance")

    args = parser.parse_args()
    if args.command == "build":
        build_pool(args.queries, args.top_k, args.seed)
    else:
        advance()


if __name__ == "__main__":
    main()
