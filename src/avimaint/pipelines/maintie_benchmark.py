"""Validate and expose the isolated MaintIE benchmark execution plan."""
from __future__ import annotations

import argparse
from pathlib import Path

from avimaint.pipelines.common import load_experiment, summarize


ROOT = Path(__file__).resolve().parents[3]
COMMANDS = [
    "scripts/maintie/01_prepare.bat",
    "scripts/maintie/02_train_baselines.bat",
    "scripts/maintie/03_span_ablation.bat",
    "scripts/maintie/04_export_spert.bat",
    "scripts/maintie/05_train_and_test_spert.bat",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/rq4_maintie.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment, dataset = load_experiment(args.config)
    if dataset.get("allow_recommender", True):
        raise ValueError("MaintIE must remain isolated from aviation recommendations.")
    print(summarize(experiment, dataset))
    print("\nMaintIE commands:")
    for index, command in enumerate(COMMANDS, 1):
        exists = (ROOT / command).is_file()
        print(f"  {index}. {command}  [{'ready' if exists else 'missing'}]")
    print("\nMaintIE scores are an external IE benchmark only; its records never enter the dashboard corpus.")


if __name__ == "__main__":
    main()
