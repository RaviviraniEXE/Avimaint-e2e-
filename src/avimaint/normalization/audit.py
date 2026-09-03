"""Build and audit raw-to-reference pairs before any model training."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from avimaint.normalization.io import read_table, require_columns, write_table

WORD_RE = re.compile(r"[A-Za-z0-9#./-]+")


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _canonical_id(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def classify_pair(raw_text: str, reference_text: str) -> tuple[str, str]:
    """Return a conservative heuristic category and review reason.

    The category is a review aid, never an automatic gold-quality judgment.
    """
    raw = _clean(raw_text)
    target = _clean(reference_text)
    if not raw or not target:
        return "unmatched_or_empty", "source or reference is empty"
    if raw == target:
        return "unchanged", "exactly identical"
    if raw.lower() == target.lower():
        return "formatting_only", "case differs only"

    raw_tokens = _tokens(raw)
    target_tokens = _tokens(target)
    ratio = SequenceMatcher(None, raw.lower(), target.lower()).ratio()
    target_extra = max(0, len(target_tokens) - len(raw_tokens))
    extra_ratio = target_extra / max(1, len(raw_tokens))

    # Abrupt final fragments are common in fixed-width records and require review.
    last_raw = raw_tokens[-1] if raw_tokens else ""
    last_target = target_tokens[-1] if target_tokens else ""
    looks_completed = (
        last_raw
        and last_target.startswith(last_raw)
        and len(last_target) >= len(last_raw) + 2
        and not raw.endswith((".", "!", "?"))
    )
    if looks_completed:
        return "possible_truncation_reconstruction", "final source token may be completed"
    if extra_ratio > 0.35 or (ratio < 0.55 and target_extra > 2):
        return "possible_unsupported_addition", "reference adds substantial text"
    if any(len(source) <= 5 < len(target) for source, target in zip(raw_tokens, target_tokens)):
        return "possible_abbreviation_expansion", "short tokens may be expanded"
    return "lexical_edit", "spelling, token or punctuation edits"


def _combined_raw(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    dataset = config["dataset"]
    require_columns(
        raw,
        [
            dataset["raw_id_column"],
            dataset["raw_problem_column"],
            dataset["raw_action_column"],
        ],
        "raw dataset",
    )
    separator = config["pairing"].get("combined_separator", " ")
    output = pd.DataFrame(
        {
            "record_id": raw[dataset["raw_id_column"]].map(_canonical_id),
            "raw_problem": raw[dataset["raw_problem_column"]].map(_clean),
            "raw_action": raw[dataset["raw_action_column"]].map(_clean),
        }
    )
    output["raw_combined"] = (
        output["raw_problem"] + separator + output["raw_action"]
    ).str.strip()
    return output


def build_audit(config: dict[str, Any]) -> pd.DataFrame:
    dataset = config["dataset"]
    raw = _combined_raw(read_table(dataset["raw_path"]), config)
    reference_path = Path(dataset["reference_path"])
    sheet_name = dataset.get("reference_sheet_name")
    if reference_path.suffix.lower() in {".xlsx", ".xls"} and sheet_name:
        reference = pd.read_excel(reference_path, sheet_name=sheet_name)
    else:
        reference = read_table(reference_path)
    reference_id = dataset["reference_id_column"]
    require_columns(reference, [reference_id], "reference dataset")
    reference = reference.copy()
    reference["record_id"] = reference[reference_id].map(_canonical_id)
    if dataset.get("drop_reference_rows_without_id", True):
        reference = reference[reference["record_id"] != ""].copy()

    problem_column = dataset.get("reference_problem_column")
    action_column = dataset.get("reference_action_column")
    separate_reference_fields = bool(problem_column and action_column)
    if separate_reference_fields:
        require_columns(reference, [problem_column, action_column], "reference dataset")
        reference["reference_problem"] = reference[problem_column].map(_clean)
        reference["reference_action"] = reference[action_column].map(_clean)
    else:
        raw_column = dataset.get("reference_raw_column")
        target_column = dataset["reference_target_column"]
        source_problem = dataset.get("reference_source_problem_column")
        source_action = dataset.get("reference_source_action_column")
        if source_problem and source_action:
            require_columns(
                reference,
                [source_problem, source_action, target_column],
                "reference dataset",
            )
            separator = config["pairing"].get("combined_separator", " ")
            reference["reference_source_problem"] = reference[source_problem].map(_clean)
            reference["reference_source_action"] = reference[source_action].map(_clean)
            reference["reference_source_text"] = (
                reference["reference_source_problem"]
                + separator
                + reference["reference_source_action"]
            ).str.strip()
        elif raw_column:
            require_columns(reference, [raw_column, target_column], "reference dataset")
            reference["reference_source_text"] = reference[raw_column].map(_clean)
        else:
            raise ValueError(
                "Reference requires either reference_raw_column or both "
                "reference_source_problem_column and reference_source_action_column"
            )
        reference["reference_text"] = reference[target_column].map(_clean)
        edit_column = dataset.get("reference_edit_column")
        if edit_column:
            require_columns(reference, [edit_column], "reference dataset")
            reference["reference_edit"] = reference[edit_column].map(_clean)
        else:
            reference["reference_edit"] = ""

    if config["pairing"].get("require_unique_ids", True):
        for name, frame in (("raw", raw), ("reference", reference)):
            duplicate_ids = frame.loc[frame["record_id"].duplicated(), "record_id"]
            if not duplicate_ids.empty:
                raise ValueError(f"{name} contains duplicate IDs, e.g. {duplicate_ids.iloc[0]}")

    if separate_reference_fields:
        merged_records = raw.merge(
            reference[["record_id", "reference_problem", "reference_action"]],
            on="record_id",
            how="outer",
            indicator=True,
        )
        problem_rows = merged_records.copy()
        problem_rows["field"] = "problem"
        problem_rows["raw_text"] = problem_rows["raw_problem"].map(_clean)
        problem_rows["reference_text"] = problem_rows["reference_problem"].map(_clean)
        action_rows = merged_records.copy()
        action_rows["field"] = "action"
        action_rows["raw_text"] = action_rows["raw_action"].map(_clean)
        action_rows["reference_text"] = action_rows["reference_action"].map(_clean)
        merged = pd.concat([problem_rows, action_rows], ignore_index=True)
        merged["source_matches_reference_source"] = True
    else:
        selected = ["record_id", "reference_text"]
        if "reference_source_text" in reference:
            selected.append("reference_source_text")
        if "reference_edit" in reference:
            selected.append("reference_edit")
        merged = raw.merge(reference[selected], on="record_id", how="outer", indicator=True)
        merged["field"] = "combined"
        merged["raw_text"] = merged["raw_combined"].map(_clean)
        merged["source_matches_reference_source"] = True
        if "reference_source_text" in merged:
            merged["source_matches_reference_source"] = (
                merged["raw_text"].str.lower()
                == merged["reference_source_text"].map(_clean).str.lower()
            )
    if "reference_edit" not in merged:
        merged["reference_edit"] = ""
    categories = [
        classify_pair(source, target)
        for source, target in zip(merged["raw_text"], merged["reference_text"], strict=False)
    ]
    merged["automatic_category"] = [item[0] for item in categories]
    merged["review_reason"] = [item[1] for item in categories]
    completed = merged["reference_edit"].str.contains("CUTOFF - C", na=False)
    anonymized = merged["reference_edit"].str.contains("ANON", na=False)
    truncated = merged["reference_edit"].str.contains("CUTOFF - T", na=False)
    merged.loc[completed, "automatic_category"] = "reference_completion"
    merged.loc[completed, "review_reason"] = (
        "Amin marks this fixed-width source as appropriately completed; "
        "exclude from primary lexical normalization unless manually justified"
    )
    merged.loc[anonymized, "automatic_category"] = "anonymized_completion"
    merged.loc[anonymized, "review_reason"] = (
        "Amin inserted 'reference manual' for anonymized content; evaluate separately"
    )
    merged.loc[truncated & ~completed & ~anonymized, "automatic_category"] = (
        "reference_truncation"
    )
    merged.loc[truncated & ~completed & ~anonymized, "review_reason"] = (
        "Amin marks the cutoff text as appropriately truncated; review as a separate subset"
    )
    source_mismatch = ~merged["source_matches_reference_source"].fillna(False)
    merged.loc[source_mismatch, "automatic_category"] = "source_mismatch"
    merged.loc[source_mismatch, "review_reason"] = (
        "Amin original problem/action does not exactly match the local MaintNet source"
    )
    merged["review_status"] = "needs_review"
    merged["review_comment"] = ""
    merged["valid_primary"] = False
    merged["example_id"] = merged["record_id"] + "::" + merged["field"]
    return merged[
        [
            "example_id",
            "record_id",
            "field",
            "raw_problem",
            "raw_action",
            "raw_text",
            "reference_text",
            "reference_edit",
            "automatic_category",
            "review_reason",
            "source_matches_reference_source",
            "review_status",
            "review_comment",
            "valid_primary",
            "_merge",
        ]
    ]


def run_audit(config: dict[str, Any]) -> Path:
    audit = build_audit(config)
    output = Path(config["outputs"]["pair_audit_csv"])
    review_output = Path(config["outputs"]["review_template_csv"])
    write_table(audit, output)
    # Never destroy a completed expert review when the deterministic audit is
    # reproduced.  A reviewed file is reusable only when its example IDs still
    # match the newly audited source/reference pair set exactly.
    if review_output.exists():
        existing = read_table(review_output)
        require_columns(existing, ["example_id", "review_status"], "existing manual review")
        if set(existing["example_id"].astype(str)) != set(audit["example_id"].astype(str)):
            raise ValueError(
                "Existing manual review IDs differ from the current audit. Move it aside and "
                "reconcile the source/reference versions; it was not overwritten."
            )
        reviewed = ~existing["review_status"].astype(str).isin(["", "needs_review"])
        if not reviewed.any():
            write_table(audit, review_output)
    else:
        write_table(audit, review_output)
    return output


def prepare_approved_pairs(config: dict[str, Any]) -> Path:
    review_path = Path(config["outputs"]["review_template_csv"])
    review = read_table(review_path)
    require_columns(
        review,
        ["example_id", "record_id", "raw_text", "reference_text", "review_status"],
        "review table",
    )
    approved = set(config["pairing"]["primary_valid_statuses"])
    selected = review[review["review_status"].isin(approved)].copy()
    if selected.empty:
        raise ValueError(
            "No approved pairs. Review the audit CSV and set review_status to an approved value."
        )
    selected["input_text"] = [
        f"normalize {field}: {_clean(value)}"
        for field, value in zip(selected["field"], selected["raw_text"], strict=False)
    ]
    selected["target_text"] = selected["reference_text"].map(_clean)
    selected["valid_primary"] = True
    output = Path(config["outputs"]["valid_pairs_parquet"])
    write_table(selected, output)
    return output
