"""Strict held-out evaluation with ID checks and bootstrap intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from avimaint.normalization.io import read_table, require_columns, write_json
from avimaint.normalization.metrics import _change_counts, _edit_distance, score_corpus
from avimaint.normalization.protected import validate_protected


def _row_statistics(frame: Any) -> dict[str, np.ndarray]:
    """Precompute additive row statistics once for fast exact bootstrapping."""
    statistics: dict[str, list[float]] = {
        "raw_word_errors": [],
        "model_word_errors": [],
        "reference_words": [],
        "model_character_errors": [],
        "reference_characters": [],
        "exact": [],
        "change_true_positive": [],
        "change_predicted": [],
        "change_gold": [],
        "protected": [],
    }
    for source, prediction, reference in zip(
        frame["raw_text"].astype(str),
        frame["prediction_text"].astype(str),
        frame["reference_text"].astype(str),
        strict=False,
    ):
        reference_words = reference.split()
        statistics["raw_word_errors"].append(
            _edit_distance(reference_words, source.split())
        )
        statistics["model_word_errors"].append(
            _edit_distance(reference_words, prediction.split())
        )
        statistics["reference_words"].append(len(reference_words))
        statistics["model_character_errors"].append(
            _edit_distance(list(reference), list(prediction))
        )
        statistics["reference_characters"].append(len(reference))
        statistics["exact"].append(float(prediction == reference))
        true_positive, predicted, gold = _change_counts(source, prediction, reference)
        statistics["change_true_positive"].append(true_positive)
        statistics["change_predicted"].append(predicted)
        statistics["change_gold"].append(gold)
        statistics["protected"].append(float(validate_protected(source, prediction).accepted))
    return {key: np.asarray(values, dtype=np.float64) for key, values in statistics.items()}


def _scores_from_statistics(
    statistics: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, float]:
    def total(name: str) -> float:
        return float(statistics[name][indices].sum())

    reference_words = total("reference_words")
    reference_characters = total("reference_characters")
    raw_wer = total("raw_word_errors") / reference_words if reference_words else 0.0
    model_wer = total("model_word_errors") / reference_words if reference_words else 0.0
    model_cer = (
        total("model_character_errors") / reference_characters
        if reference_characters
        else 0.0
    )
    predicted_total = total("change_predicted")
    gold_total = total("change_gold")
    true_positive = total("change_true_positive")
    precision = true_positive / predicted_total if predicted_total else 1.0
    recall = true_positive / gold_total if gold_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "wer": model_wer,
        "cer": model_cer,
        "exact_match": float(statistics["exact"][indices].mean()),
        "error_reduction_rate": (raw_wer - model_wer) / raw_wer if raw_wer > 0 else 0.0,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
        "protected_token_accuracy": float(statistics["protected"][indices].mean()),
    }


def _bootstrap(frame: Any, samples: int = 1000, seed: int = 42) -> dict[str, list[float]]:
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    statistics = _row_statistics(frame)
    for _ in range(samples):
        indices = generator.integers(0, len(frame), size=len(frame))
        scores = _scores_from_statistics(statistics, indices)
        for key, value in scores.items():
            collected.setdefault(key, []).append(float(value))
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in collected.items()
    }


def evaluate(config: dict[str, Any], split: str, system: str) -> Path:
    run = config["run"]
    gold_path = run.get("input_path") or run.get("gold_input_path")
    gold = read_table(gold_path)
    require_columns(gold, ["example_id", "split"], "frozen split")
    expected = set(gold.loc[gold["split"] == split, "example_id"].astype(str))
    prediction_path = Path(run["prediction_dir"]) / f"{split}_{system}.csv"
    predictions = read_table(prediction_path)
    require_columns(
        predictions,
        ["example_id", "raw_text", "prediction_text", "reference_text"],
        "predictions",
    )
    observed = set(predictions["example_id"].astype(str))
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(f"Prediction ID mismatch; missing={missing}, unexpected={unexpected}")
    if predictions["example_id"].duplicated().any():
        raise ValueError("Predictions contain duplicate example IDs")
    scores = score_corpus(
        predictions["raw_text"],
        predictions["prediction_text"],
        predictions["reference_text"],
    )
    payload = {
        "run_id": run["id"],
        "split": split,
        "system": system,
        "records": len(predictions),
        "metrics": scores,
        "bootstrap_95_ci": _bootstrap(predictions, seed=int(run["seed"])),
        "prediction_path": str(prediction_path),
    }
    output = Path(run["prediction_dir"]) / f"{split}_{system}_metrics.json"
    write_json(payload, output)
    return output
