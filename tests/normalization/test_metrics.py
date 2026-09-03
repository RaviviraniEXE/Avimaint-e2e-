import numpy as np
import pandas as pd

from avimaint.normalization.evaluation import _scores_from_statistics, _row_statistics
from avimaint.normalization.metrics import sanitize_generated_token_ids, score_corpus
from avimaint.normalization.protected import validate_protected
from avimaint.normalization.rules import normalize_rules


def test_perfect_prediction() -> None:
    scores = score_corpus(["repl pump"], ["replaced pump"], ["replaced pump"])
    assert scores["wer"] == 0.0
    assert scores["exact_match"] == 1.0
    assert scores["change_f1"] == 1.0


def test_protected_identifier_failure() -> None:
    scores = score_corpus(
        ["removed PUMP-12"],
        ["removed pump"],
        ["removed PUMP-12"],
    )
    assert scores["protected_token_accuracy"] == 0.0


def test_generated_token_sanitizer_replaces_trainer_padding_and_invalid_ids() -> None:
    token_ids = np.array([[3, 4, -100], [3, 999999, 0]])
    cleaned = sanitize_generated_token_ids(token_ids, pad_token_id=0, vocabulary_size=384)
    assert cleaned.tolist() == [[3, 4, 0], [3, 0, 0]]
    assert cleaned.dtype == np.int64


def test_orientation_abbreviation_may_expand_without_safety_failure() -> None:
    result = validate_protected("LH/ ENG GASKET", "LEFT-HAND ENGINE GASKET")
    assert result.accepted


def test_identifier_still_requires_exact_preservation() -> None:
    result = validate_protected("CHECK PUMP-12", "CHECK PUMP")
    assert not result.accepted


def test_rules_match_uppercase_reference_contract() -> None:
    result = normalize_rules(
        "L/H ENG & CYL",
        {"l/h": "left-hand", "eng": "engine", "&": "and", "cyl": "cylinder"},
    )
    assert result == "LEFT-HAND ENGINE AND CYLINDER"


def test_numeric_words_may_replace_grounded_digits() -> None:
    result = validate_protected("CYL #2, 3", "CYLINDER NUMBER TWO AND NUMBER THREE")
    assert result.accepted


def test_unsupported_numeric_completion_is_rejected() -> None:
    result = validate_protected("CYL # ROCKER COVER", "CYLINDER NUMBER THREE ROCKER COVER")
    assert not result.accepted
    assert "UNSUPPORTED_NUMBER:3" in result.missing_tokens


def test_precomputed_bootstrap_scores_match_direct_scoring() -> None:
    frame = pd.DataFrame(
        {
            "raw_text": ["REPL PUMP-12", "CYL #2", "LH ENG"],
            "prediction_text": ["REPLACED PUMP-12", "CYLINDER NUMBER TWO", "LEFT-HAND ENGINE"],
            "reference_text": ["REPLACED PUMP-12", "CYLINDER NUMBER TWO", "LEFT-HAND ENGINE"],
        }
    )
    indices = np.array([2, 0, 2, 1, 1])
    direct = score_corpus(
        frame.iloc[indices]["raw_text"],
        frame.iloc[indices]["prediction_text"],
        frame.iloc[indices]["reference_text"],
    )
    precomputed = _scores_from_statistics(_row_statistics(frame), indices)
    assert precomputed == direct
