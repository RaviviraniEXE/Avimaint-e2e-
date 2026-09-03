"""Shared pipeline configuration checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from avimaint.configuration import load_yaml


def load_experiment(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment_config = load_yaml(path)
    experiment = experiment_config["experiment"]
    dataset_config_path = Path(experiment["dataset_config"])
    dataset_config = load_yaml(dataset_config_path)
    return experiment, dataset_config["dataset"]


def summarize(experiment: dict[str, Any], dataset: dict[str, Any]) -> str:
    return (
        f"Experiment={experiment['id']} | "
        f"RQ={experiment['research_question']} | "
        f"Dataset={dataset['id']} | "
        f"Schema={dataset.get('schema_id', experiment.get('schema_id'))}"
    )

