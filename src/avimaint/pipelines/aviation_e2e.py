"""Plan or execute the isolated aviation research stages."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from avimaint.pipelines.common import load_experiment, summarize


ROOT = Path(__file__).resolve().parents[3]
STAGES = {
    "normalization-audit": ["scripts/normalization/01_audit_reference.bat"],
    "normalization-prepare": ["scripts/normalization/02_prepare_approved_pairs.bat"],
    "normalization-split": ["scripts/normalization/03_create_cluster_safe_split.bat"],
    "normalization-train": ["scripts/normalization/04_train_byt5_gold.bat"],
    "normalization-validate": ["scripts/normalization/05_run_validation_comparison.bat"],
    "annotations": ["scripts/ie/01_import_annotations.bat", "scripts/ie/02_freeze_split.bat"],
    "normalization-full": ["scripts/normalization/09_predict_full_corpus.bat",
                            "scripts/normalization/10_compare_downstream_ie.bat"],
    "ie": ["scripts/ie/03_train_classical.bat", "scripts/ie/04_train_neural.bat"],
    "spert-export": ["scripts/ie/05_export_spert.bat"],
    "spert": ["scripts/ie/06_train_and_test_spert.bat",
              "scripts/ie/07_import_spert_and_report.bat"],
    "full-corpus": ["scripts/ie/08_prepare_full_corpus.bat",
                    "scripts/ie/08b_predict_full_corpus_spert.bat"],
    "maintie": ["scripts/maintie/01_prepare.bat", "scripts/maintie/02_train_baselines.bat",
                "scripts/maintie/03_span_ablation.bat", "scripts/maintie/04_export_spert.bat"],
    "maintie-spert": ["scripts/maintie/05_train_and_test_spert.bat"],
    "dashboard-eval": ["scripts/dashboard/run_evaluation.bat"],
    "dashboard": ["scripts/dashboard/run_dashboard.bat"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/rq1_normalization.yaml")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--execute", action="store_true", help="Run the selected Windows launchers")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands")
    args = parser.parse_args()
    experiment, dataset = load_experiment(args.config)
    if not dataset.get("allow_recommender", False):
        raise ValueError("This dataset is not approved for aviation case retrieval.")
    print(summarize(experiment, dataset))
    selected = [name for name in STAGES if args.stage in ("all", name)]
    commands = [command for name in selected for command in STAGES[name]]
    print("\nExecution plan:")
    for index, command in enumerate(commands, 1):
        state = "ready" if (ROOT / command).is_file() else "missing"
        print(f"  {index:02d}. {command} [{state}]")
    if args.dry_run or not args.execute:
        print("\nPlan only. On Windows run the launchers above or pass --execute.")
        return
    if os.name != "nt":
        raise SystemExit("The .bat execution plan is Windows-only; use the documented Python commands on Linux.")
    for command in commands:
        subprocess.run(["cmd", "/c", str(ROOT / command)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
