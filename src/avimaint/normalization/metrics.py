"""Intrinsic metrics for safe lexical normalization."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

import numpy as np

from avimaint.normalization.protected import validate_protected


def sanitize_generated_token_ids(
    token_ids: np.ndarray, pad_token_id: int, vocabulary_size: int
) -> np.ndarray:
    """Replace Trainer padding and invalid generated ids before decoding."""

    values = np.asarray(token_ids)
    if values.ndim == 3:
        values = np.argmax(values, axis=-1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D token ids or 3-D logits, received shape {values.shape}")
    return np.where(
        (values >= 0) & (values < vocabulary_size), values, pad_token_id
    ).astype(np.int64, copy=False)


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_value != right_value)
            current.append(min(current[-1] + 1, previous[right_index] + 1, substitution))
        previous = current
    return previous[-1]


def _error_rate(references: list[str], hypotheses: list[str], characters: bool) -> float:
    errors = 0
    units = 0
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        reference_units = list(reference) if characters else reference.split()
        hypothesis_units = list(hypothesis) if characters else hypothesis.split()
        errors += _edit_distance(reference_units, hypothesis_units)
        units += len(reference_units)
    return errors / units if units else 0.0


def _change_counts(source: str, prediction: str, reference: str) -> tuple[int, int, int]:
    source_tokens = source.split()
    predicted_changes: Counter[tuple[str, ...]] = Counter()
    gold_changes: Counter[tuple[str, ...]] = Counter()
    for label, target, bucket in (
        ("prediction", prediction.split(), predicted_changes),
        ("reference", reference.split(), gold_changes),
    ):
        matcher = SequenceMatcher(None, source_tokens, target)
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if tag != "equal":
                bucket[(tag, *source_tokens[left_start:left_end], "=>", *target[right_start:right_end])] += 1
    true_positive = sum((predicted_changes & gold_changes).values())
    return true_positive, sum(predicted_changes.values()), sum(gold_changes.values())


def score_corpus(
    sources: Iterable[str], predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    source_list = [str(value) for value in sources]
    prediction_list = [str(value) for value in predictions]
    reference_list = [str(value) for value in references]
    if not (len(source_list) == len(prediction_list) == len(reference_list)):
        raise ValueError("sources, predictions and references must have equal length")
    if not source_list:
        raise ValueError("Cannot score an empty corpus")

    raw_wer = _error_rate(reference_list, source_list, characters=False)
    model_wer = _error_rate(reference_list, prediction_list, characters=False)
    model_cer = _error_rate(reference_list, prediction_list, characters=True)
    exact = float(np.mean([prediction == reference for prediction, reference in zip(prediction_list, reference_list, strict=False)]))
    error_reduction = (raw_wer - model_wer) / raw_wer if raw_wer > 0 else 0.0

    true_positive = predicted_total = gold_total = 0
    protection_scores: list[float] = []
    for source, prediction, reference in zip(source_list, prediction_list, reference_list, strict=False):
        true, predicted, gold = _change_counts(source, prediction, reference)
        true_positive += true
        predicted_total += predicted
        gold_total += gold
        protection_scores.append(float(validate_protected(source, prediction).accepted))
    precision = true_positive / predicted_total if predicted_total else 1.0
    recall = true_positive / gold_total if gold_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "wer": model_wer,
        "cer": model_cer,
        "exact_match": exact,
        "error_reduction_rate": error_reduction,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
        "protected_token_accuracy": float(np.mean(protection_scores)),
    }
