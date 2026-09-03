"""Fast deterministic readiness check; never trains models or evaluates test predictions."""

from __future__ import annotations

import argparse
import compileall
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def check(condition: bool, message: str, errors: list[str]) -> None:
    print(("[OK]   " if condition else "[FAIL] ") + message)
    if not condition:
        errors.append(message)


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []

    raw = pd.read_csv(ROOT / "data/aviation/raw/Aircraft_Annotation_DataFile.csv")
    ref = pd.read_csv(ROOT / "data/aviation/reference/amin_cleaned_dataset.csv")
    review = pd.read_csv(ROOT / "data/aviation/interim/normalization_manual_review.csv")
    check(len(raw) == 6169, "aviation raw corpus has 6,169 records", errors)
    check(len(ref) == 6169, "Amin expert-expanded reference has 6,169 records", errors)
    check(len(review) == 6169, "pair audit covers all 6,169 raw/reference pairs", errors)
    primary_statuses = {"approved_lexical", "approved_abbreviation"}
    primary_pairs = int(review["review_status"].isin(primary_statuses).sum())
    check(
        primary_pairs == 5124, "normalization primary population has 5,124 approved pairs", errors
    )
    check(
        review["review_status"].astype(str).str.len().gt(0).all(),
        "every normalization pair has a review decision",
        errors,
    )
    normalization_split = pd.read_csv(
        ROOT / "data/splits/normalization_cluster_safe_v1.csv", dtype=str
    )
    normalization_counts = normalization_split["split"].value_counts().to_dict()
    check(
        normalization_counts == {"train": 3457, "validation": 926, "test": 741},
        "normalization split is frozen at 3,457/926/741",
        errors,
    )
    check(
        normalization_split.groupby("cluster_id")["split"].nunique().max() == 1,
        "normalization clusters do not cross train/validation/test",
        errors,
    )
    sensitivity = pd.read_parquet(
        ROOT / "data/aviation/processed/normalization_expert_sensitivity.parquet"
    )
    check(
        len(sensitivity) == 1045,
        "expert-expansion sensitivity population has 1,045 separate pairs",
        errors,
    )

    source_files = sorted((ROOT / "data/aviation/annotations/source_gold").glob("*.jsonl"))
    gold_files = sorted((ROOT / "data/aviation/annotations/gold").glob("*.jsonl"))
    source = [record for path in source_files for record in jsonl(path)]
    gold = [record for path in gold_files for record in jsonl(path)]
    check(len(source) == 1600, "authoritative source annotations contain 1,600 records", errors)
    check(len(gold) == 1600, "audited training annotations contain 1,600 records", errors)
    check(
        len({str(record["ident"]) for record in gold}) == 1600,
        "audited annotations have 1,600 unique identifiers",
        errors,
    )

    audit = json.loads(
        (ROOT / "outputs/reports/annotation_audit/annotation_audit.json").read_text()
    )
    check(
        audit["structural_errors"] == 0, "annotation audit reports zero structural errors", errors
    )
    check(
        len(audit["unresolved_exact_duplicate_groups"]) == 0,
        "annotation audit reports zero unresolved exact duplicates",
        errors,
    )
    check(
        audit["entity_support"].get("REFERENCE", 0) == 67,
        "frozen full schema retains 67 REFERENCE entities",
        errors,
    )
    check(
        audit["relation_support"].get("ACTION_FOLLOWS_REFERENCE", 0) == 70,
        "frozen full schema retains 70 reference relations",
        errors,
    )

    split = json.loads((ROOT / "legacy_import/maintenance-ie/outputs/splits.json").read_text())
    check(
        (len(split["train"]), len(split["dev"]), len(split["test"])) == (1275, 100, 225),
        "aviation IE split is frozen at 1,275/100/225",
        errors,
    )
    split_audit = json.loads(
        (ROOT / "outputs/reports/annotation_audit/frozen_split_audit.json").read_text()
    )
    overlap = split_audit.get("exact_group_overlap", {})
    check(
        split_audit.get("valid") is True and all(value == 0 for value in overlap.values()),
        "frozen IE split has zero exact-duplicate group overlap",
        errors,
    )

    maintie = json.loads((ROOT / "data/maintie/raw/gold_release.json").read_text(encoding="utf-8"))
    check(len(maintie) == 1076, "MaintIE external benchmark has 1,076 records", errors)
    maintie_split = json.loads(
        (ROOT / "legacy_import/maintie-bench/outputs/splits.json").read_text()
    )
    check(
        (len(maintie_split["train"]), len(maintie_split["dev"]), len(maintie_split["test"]))
        == (860, 108, 108),
        "MaintIE split is 860/108/108",
        errors,
    )

    experiments = sorted((ROOT / "configs/experiments").glob("*.yaml"))
    rq_ids = {
        yaml.safe_load(path.read_text())["experiment"].get("research_question")
        for path in experiments
    }
    check(
        {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6"}.issubset(rq_ids),
        "experiment configs cover RQ1 through RQ6",
        errors,
    )
    check(
        (ROOT / "legacy_import/maintenance-ie/config/schema_core.yaml").is_file(),
        "derived core schema ablation is executable",
        errors,
    )
    check(
        (ROOT / "scripts/setup/clone_official_spert.bat").is_file(),
        "official SpERT clone-and-pin launcher is present",
        errors,
    )
    check(
        not (ROOT / "legacy_import/spert").exists(),
        "no obsolete vendored SpERT implementation is packaged",
        errors,
    )

    dashboard = ROOT / "legacy_import/maintenance-ie/avimaint_dss"
    check(
        (dashboard / "data/dashboard_dataset_D.csv").is_file(),
        "final dashboard corpus is present",
        errors,
    )
    dashboard_rows = pd.read_csv(dashboard / "data/dashboard_dataset_D.csv")
    check(len(dashboard_rows) == 6169, "final dashboard corpus has 6,169 cases", errors)
    check((dashboard / "run_rq5_retrieval.py").is_file(), "RQ5 evaluator is present", errors)
    check((dashboard / "run_rq6_uncertainty.py").is_file(), "RQ6 evaluator is present", errors)
    check(
        (dashboard / "run_robustness.py").is_file(),
        "cross-cutting robustness evaluator is present",
        errors,
    )

    if args.compile:
        for relative in [
            "src",
            "scripts",
            "legacy_import/maintenance-ie/src",
            "legacy_import/maintie-bench/src",
            "legacy_import/maintenance-ie/avimaint_dss",
        ]:
            check(
                compileall.compile_dir(ROOT / relative, quiet=1),
                f"Python compiles: {relative}",
                errors,
            )
    if errors:
        raise SystemExit(f"Readiness failed with {len(errors)} error(s).")
    print("\nProject data, schemas, source, launchers, and evaluation contracts are ready.")
    print("Heavy GPU training and final held-out evaluation remain explicit user-run steps.")


if __name__ == "__main__":
    main()
