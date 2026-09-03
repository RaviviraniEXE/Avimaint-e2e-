"""Deterministic cross-cutting retrieval robustness study.

Evaluates lexical truncation, spelling noise, and missing structured evidence.
Multi-problem records are exported for qualitative review because a single
recorded action-family label is not a valid gold label for two joined problems.
"""

from __future__ import annotations

import argparse
import json
import random

import pandas as pd
from research_evaluation import PROJECT_ROOT, evaluate_partition, load_system, protocol_frames


def truncate(text: str, fraction: float) -> str:
    tokens = str(text).split()
    keep = max(1, round(len(tokens) * (1.0 - fraction)))
    return " ".join(tokens[:keep])


def spelling_noise(text: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    chars = list(str(text))
    candidates = [i for i, char in enumerate(chars) if char.isalpha()]
    for index in rng.sample(candidates, min(len(candidates), round(len(candidates) * rate))):
        if rng.random() < 0.5:
            chars[index] = ""
        else:
            base = ord("A") if chars[index].isupper() else ord("a")
            chars[index] = chr(base + ((ord(chars[index]) - base + 1) % 26))
    return "".join(chars)


def perturb(frame: pd.DataFrame, kind: str, value: float) -> pd.DataFrame:
    changed = frame.copy(deep=True)
    if kind == "truncation":
        changed["problem_norm"] = changed["problem_norm"].map(lambda text: truncate(text, value))
    elif kind == "spelling_noise":
        changed["problem_norm"] = [
            spelling_noise(text, value, 42000 + i) for i, text in enumerate(changed["problem_norm"])
        ]
    elif kind == "missing_structure":
        changed["components"] = [[] for _ in range(len(changed))]
        changed["faults"] = [[] for _ in range(len(changed))]
        changed["problem_entity_types"] = [[] for _ in range(len(changed))]
        changed["problem_relation_types"] = [[] for _ in range(len(changed))]
    else:
        raise ValueError(kind)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["raw", "rules_then_byt5"], default="rules_then_byt5")
    parser.add_argument(
        "--representation", choices=["bm25", "tfidf", "structure", "hybrid"], default="hybrid"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Smoke-test limit; use 0 for the thesis run"
    )
    args = parser.parse_args()
    frame = load_system(args.system)
    candidates, clean_queries = protocol_frames(frame, "frozen", "test")
    if args.limit:
        clean_queries = clean_queries.head(args.limit)
    conditions = [
        ("clean", 0.0),
        ("truncation", 0.10),
        ("truncation", 0.20),
        ("truncation", 0.30),
        ("spelling_noise", 0.05),
        ("spelling_noise", 0.10),
        ("missing_structure", 1.0),
    ]
    output = PROJECT_ROOT / "outputs/runs/crosscut_robustness/retrieval"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, value in conditions:
        queries = clean_queries if kind == "clean" else perturb(clean_queries, kind, value)
        metrics, predictions = evaluate_partition(candidates, queries, args.representation)
        rows.append(
            {
                "condition": kind,
                "value": value,
                **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
            }
        )
        predictions.to_csv(output / f"predictions_{kind}_{value:.2f}.csv", index=False)
    summary = pd.DataFrame(rows)
    baseline = float(summary.iloc[0]["top1_action_family_agreement"])
    summary["top1_absolute_drop"] = baseline - summary["top1_action_family_agreement"]
    summary.to_csv(output / "robustness_metrics.csv", index=False)

    qualitative = clean_queries.reset_index(drop=True).head(50).copy()
    if len(qualitative) > 1:
        paired = pd.DataFrame(
            {
                "first_ident": qualitative["ident"].iloc[::2].reset_index(drop=True),
                "first_problem": qualitative["problem"].iloc[::2].reset_index(drop=True),
                "second_ident": qualitative["ident"].iloc[1::2].reset_index(drop=True),
                "second_problem": qualitative["problem"].iloc[1::2].reset_index(drop=True),
            }
        ).dropna()
        paired["combined_problem"] = paired["first_problem"] + " ; " + paired["second_problem"]
        paired["evaluation_status"] = "qualitative_only_no_single_valid_action_family_gold"
        paired.to_csv(output / "multi_problem_qualitative_set.csv", index=False)
    metadata = {
        "seed": 42,
        "protocol": "frozen_train_candidates_frozen_test_queries",
        "system": args.system,
        "representation": args.representation,
        "multi_problem_policy": "qualitative only; no misleading single-label accuracy",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Robustness study -> {output}")


if __name__ == "__main__":
    main()
