"""Step 15 — cluster-safe, procedure-level dashboard recommender evaluation.

The old evaluator scored isolated action fragments and let repeated rows
contribute repeated queries. This version evaluates complete historical
procedures and gives each problem cluster one query.

Without ``--spert-url`` the stored problem-side graph is used as a development
proxy. Those numbers must not be reported as final thesis results. With a SpERT
URL, every held-out problem is processed by query-side SpERT inference, but the
reference procedures are still automatically extracted silver labels; final
claims additionally require a manually reviewed relevance set.

Typical command:

    python Scripts\\15_eval_recommender.py --name aviation

Query-side SpERT command:

    python Scripts\\15_eval_recommender.py --name aviation ^
      --spert-url http://127.0.0.1:8000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.core.config import load_yaml
from dashboard.core.data import load_cases
from dashboard.core.query_extraction import ApiSpERTExtractor, QueryGraph
from dashboard.core.recommender import (
    ProcedureRecommender,
    procedure_signature,
)
from dashboard.core.schema import SchemaCatalog


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _case_mode(case: dict[str, Any]) -> str | None:
    roles = {
        str(step.get("role") or "unresolved")
        for step in (case.get("procedure", {}) or {}).get("steps", [])
    }
    if "corrective" in roles:
        return "corrective"
    if "diagnostic" in roles:
        return "diagnostic"
    return None


def graph_from_case(case: dict[str, Any]) -> QueryGraph:
    """Reconstruct the stored problem-only graph for development evaluation."""
    graph = case.get("problem_graph", {}) or {}
    entities: list[dict[str, Any]] = []
    ids_by_type_text: dict[tuple[str, str], str] = {}

    def add(entity_type: str, values: list[str]) -> None:
        for value in values:
            text = _normalise(value)
            if not text:
                continue
            key = (entity_type, text.casefold())
            if key in ids_by_type_text:
                continue
            entity_id = f"q{len(entities)}"
            ids_by_type_text[key] = entity_id
            entities.append(
                {
                    "id": entity_id,
                    "type": entity_type,
                    "text": text,
                    "enabled": True,
                }
            )

    add("FAULT", graph.get("faults", []) or [])
    add("ABN_PROC", graph.get("abnormal_processes", []) or [])
    add("MAINT_ITEM", graph.get("items", []) or [])
    add("TECH_OBS", graph.get("observations", []) or [])
    add("OP_CTX", graph.get("contexts", []) or [])
    add("LOC", graph.get("locations", []) or [])

    relations: list[dict[str, Any]] = []
    for pair in graph.get("issue_item_pairs", []) or []:
        if isinstance(pair, dict):
            issue = _normalise(pair.get("issue"))
            item = _normalise(pair.get("item"))
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            issue = _normalise(pair[0])
            item = _normalise(pair[1])
        else:
            continue
        issue_id = (
            ids_by_type_text.get(("FAULT", issue.casefold()))
            or ids_by_type_text.get(("ABN_PROC", issue.casefold()))
        )
        item_id = ids_by_type_text.get(("MAINT_ITEM", item.casefold()))
        if issue_id and item_id:
            relations.append(
                {
                    "id": f"qr{len(relations)}",
                    "type": "ISSUE_ON_ITEM",
                    "source": issue_id,
                    "target": item_id,
                    "enabled": True,
                }
            )
    return QueryGraph(
        text=str(case.get("problem_text") or ""),
        entities=entities,
        relations=relations,
        engine="Stored problem-side SpERT graph — development proxy",
        warning=(
            "This is not independent query-side SpERT inference and must not "
            "be reported as the final thesis result."
        ),
    )


def cluster_queries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one equally weighted evaluation query per problem cluster."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cluster_id = _normalise(case.get("cluster_id"))
        if cluster_id and _normalise(case.get("problem_text")):
            groups[cluster_id].append(case)

    queries: list[dict[str, Any]] = []
    for cluster_id, members in sorted(groups.items()):
        eligible = [case for case in members if _case_mode(case)]
        if not eligible:
            continue
        representative = max(
            eligible,
            key=lambda case: (
                len(_normalise(case.get("problem_text"))),
                _normalise(case.get("case_id")),
            ),
        )
        targets = {
            procedure_signature(case)
            for case in eligible
            if _normalise(
                (case.get("procedure", {}) or {}).get("raw_solution_text")
                or case.get("solution_text")
            )
        }
        target_modes = {_case_mode(case) for case in eligible}
        if not targets:
            continue
        queries.append(
            {
                "cluster_id": cluster_id,
                "representative": representative,
                "target_signatures": targets,
                "target_mode": (
                    "corrective"
                    if "corrective" in target_modes
                    else "diagnostic"
                ),
                "source_record_count": len(members),
            }
        )
    return queries


def _eligible_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    corrective = [
        candidate for candidate in candidates if candidate.get("has_corrective")
    ]
    if corrective:
        return corrective
    return [
        candidate
        for candidate in candidates
        if candidate.get("has_diagnostic")
    ]


def _rank(
    signatures: list[str],
    relevant: set[str],
) -> int:
    for index, signature in enumerate(signatures, start=1):
        if signature in relevant:
            return index
    return 0


def _ndcg_at_k(
    signatures: list[str],
    relevant: set[str],
    k: int,
) -> float:
    gains = [
        1.0 if signature in relevant else 0.0
        for signature in signatures[:k]
    ]
    dcg = sum(
        gain / math.log2(index + 2)
        for index, gain in enumerate(gains)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_mode(
    recommender: ProcedureRecommender,
    queries: list[dict[str, Any]],
    mode: str,
    extractor: ApiSpERTExtractor | None,
    limit: int | None,
) -> dict[str, Any]:
    evaluated = hit1 = hit3 = hit5 = 0
    mrr = ndcg = 0.0
    covered = correct_when_covered = 0
    mode_correct = enabling_headline_errors = 0
    extraction_errors = 0

    selected = queries[: limit or len(queries)]
    for item in selected:
        representative = item["representative"]
        evaluated += 1
        try:
            query = (
                extractor.extract(representative["problem_text"])
                if extractor is not None
                else graph_from_case(representative)
            )
        except Exception as exc:
            extraction_errors += 1
            print(
                f"[warn] query extraction failed for cluster "
                f"{item['cluster_id']}: {exc}",
                file=sys.stderr,
            )
            continue

        candidates = _eligible_candidates(
            recommender.rank_procedures(
                query,
                mode=mode,
                exclude_clusters={item["cluster_id"]},
            )
        )
        signatures = [
            candidate["procedure_signature"] for candidate in candidates[:5]
        ]
        rank = _rank(signatures, item["target_signatures"])
        hit1 += int(rank == 1)
        hit3 += int(1 <= rank <= 3)
        hit5 += int(1 <= rank <= 5)
        if rank:
            mrr += 1.0 / rank
        ndcg += _ndcg_at_k(signatures, item["target_signatures"], 5)

        result = recommender.recommend(
            query,
            mode=mode,
            exclude_clusters={item["cluster_id"]},
        )
        if not result.get("abstain"):
            covered += 1
            primary = result.get("primary") or {}
            if primary.get("procedure_signature") in item["target_signatures"]:
                correct_when_covered += 1
            if result.get("mode") == item["target_mode"]:
                mode_correct += 1
            if not (
                primary.get("has_corrective")
                or primary.get("has_diagnostic")
            ):
                enabling_headline_errors += 1

    denominator = max(evaluated, 1)
    coverage_denominator = max(covered, 1)
    return {
        "system": mode,
        "unique_problem_clusters": evaluated,
        "query_extraction_errors": extraction_errors,
        "procedure_recall_at_1": hit1 / denominator,
        "procedure_recall_at_3": hit3 / denominator,
        "procedure_recall_at_5": hit5 / denominator,
        "mrr": mrr / denominator,
        "ndcg_at_5": ndcg / denominator,
        "recommendation_coverage": covered / denominator,
        "accuracy_when_covered": correct_when_covered / coverage_denominator,
        "selective_risk": (
            1.0 - correct_when_covered / coverage_denominator
            if covered
            else None
        ),
        "diagnostic_corrective_mode_accuracy_when_covered": (
            mode_correct / coverage_denominator
        ),
        "enabling_action_as_headline_error_rate": (
            enabling_headline_errors / coverage_denominator
        ),
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'system':<12}{'clusters':>10}{'R@1':>9}{'R@3':>9}"
        f"{'R@5':>9}{'MRR':>9}{'nDCG@5':>10}{'coverage':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['system']:<12}"
            f"{row['unique_problem_clusters']:>10}"
            f"{row['procedure_recall_at_1']:>9.3f}"
            f"{row['procedure_recall_at_3']:>9.3f}"
            f"{row['procedure_recall_at_5']:>9.3f}"
            f"{row['mrr']:>9.3f}"
            f"{row['ndcg_at_5']:>10.3f}"
            f"{row['recommendation_coverage']:>11.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate complete procedures with whole-cluster holdout."
    )
    parser.add_argument("--name", default="aviation")
    parser.add_argument(
        "--cases",
        help="Case file or directory; defaults to outputs/dashboard/<name>",
    )
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--config", default="config/recommender.yaml")
    parser.add_argument(
        "--spert-url",
        default=os.getenv("AVIMAINT_SPERT_URL", ""),
    )
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases_path = (
        Path(args.cases).expanduser()
        if args.cases
        else ROOT / "outputs" / "dashboard" / args.name
    )
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path
    cases = load_cases(cases_path)
    schema = SchemaCatalog.from_yaml(args.schema)
    recommender = ProcedureRecommender(
        cases, schema, load_yaml(args.config)
    )
    queries = cluster_queries(recommender.cases)
    if not queries:
        print(
            "No evaluable clusters contain an explicit corrective or "
            "diagnostic procedure.",
            file=sys.stderr,
        )
        return 2

    extractor = (
        ApiSpERTExtractor(args.spert_url, schema)
        if args.spert_url.strip()
        else None
    )
    systems = [
        evaluate_mode(
            recommender,
            queries,
            mode,
            extractor,
            args.limit,
        )
        for mode in ("text", "structured", "hybrid")
    ]
    status = (
        "query_side_spert_silver_reference"
        if extractor is not None
        else "stored_graph_development_proxy"
    )
    payload = {
        "status": status,
        "final_thesis_result": False,
        "warning": (
            "Query-side SpERT is active, but reference procedures are silver "
            "predictions. Final claims require manual relevance review."
            if extractor is not None
            else "Stored problem-side graphs are a development proxy, not "
            "independent query-side SpERT inference."
        ),
        "reference_standard": (
            "SpERT-derived complete historical procedure signatures (silver)"
        ),
        "evaluation_unit": "one equally weighted query per problem cluster",
        "cluster_holdout": True,
        "solution_hidden_from_retrieval": True,
        "query_side_spert": extractor is not None,
        "source_cases": len(recommender.cases),
        "eligible_unique_clusters": len(queries),
        "systems": systems,
        "backend_status": recommender.backend_status.as_dict(),
    }
    output = (
        Path(args.output).expanduser()
        if args.output
        else (
            cases_path
            if cases_path.is_dir()
            else cases_path.parent
        )
        / "recommender_metrics.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "\nDevelopment evaluation. "
        f"Status: {status}; unique clusters: {len(queries)}"
    )
    _print_table(systems)
    print(f"\nWritten: {output.resolve()}")
    print("Do not report this file as the final thesis result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

