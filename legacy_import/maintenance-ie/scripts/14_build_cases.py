"""Step 14 — build the leak-safe case library used by the v2 dashboard.

This is a dashboard-integration step. It does not train or change any IE model.
It joins the full-corpus SpERT predictions to the original paired source fields
and writes ``outputs/dashboard/<name>/cases.jsonl``.

Important safeguards:

* ``problem_text`` always comes from the original problem column.
* ``solution_text`` always comes from the original solution/action column.
* There is no combined-text fallback.
* Retrieval features contain only problem-side entities and relations.
* Invalid relation signatures are quarantined instead of used as evidence.
* ``ACTION_ON_ITEM`` alone never makes an action corrective.
* Near-duplicate groups are based on problem text only.

Typical Windows command (run from the project root):

    python Scripts\\14_build_cases.py ^
      --pred outputs\\kg\\predictions_full.json ^
      --raw data\\raw\\Aircraft_Annotation_DataFile.csv ^
      --index outputs\\kg\\full_index.jsonl ^
      --source-dataset MaintNet ^
      --name aviation ^
      --strict

If the prediction file already contains the original IDs, ``--index`` is not
required. Use ``--allow-positional-join`` only when row order has been verified.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.core.data import normalize_case
from dashboard.core.schema import SchemaCatalog
from dashboard.core.validator import validate_cases


ISSUE_TYPES = {"FAULT", "ABN_PROC"}
PROBLEM_ENTITY_TYPES = {
    "FAULT",
    "ABN_PROC",
    "MAINT_ITEM",
    "TECH_OBS",
    "OP_CTX",
    "LOC",
}
PROBLEM_RELATIONS = {
    "ISSUE_ON_ITEM",
    "OBSERVATION_OF_ITEM",
    "HAS_LOCATION",
    "OCCURS_UNDER_CONTEXT",
    "HAS_PART",
}
ENABLING_HINTS = {
    "access",
    "accessed",
    "close",
    "closed",
    "disconnect",
    "disconnected",
    "disassemble",
    "disassembled",
    "gain access",
    "gained access",
    "open",
    "opened",
    "reattach",
    "reattached",
    "refit",
    "refitted",
    "reinstall",
    "reinstalled",
    "remove",
    "removed",
}
FALSE_VALUES = {
    "0",
    "false",
    "incomplete",
    "no",
    "n",
    "truncated",
}


def _normalise(value: Any) -> str:
    """Return a stable single-line string without turning NaN into ``'nan'``."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalise(value).casefold()).strip()


def _character_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalise(value).casefold())


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = _normalise(value)
        canonical = text.casefold()
        if text and canonical not in seen:
            seen.add(canonical)
            output.append(text)
    return output


def _stable_hash(prefix: str, value: str, length: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            value = value.get(
                "predictions",
                value.get("records", value.get("cases", [value])),
            )
        values = value
    if not isinstance(values, list) or not all(
        isinstance(record, dict) for record in values
    ):
        raise ValueError(f"Expected a JSON list of records in {path}")
    return values


def load_predictions(pattern: str) -> list[dict[str, Any]]:
    raw_pattern = str(Path(pattern).expanduser())
    if not Path(raw_pattern).is_absolute():
        raw_pattern = str(ROOT / raw_pattern)
    paths = sorted(Path(value) for value in glob.glob(raw_pattern))
    direct = Path(raw_pattern)
    if not paths and direct.exists():
        paths = [direct]
    if not paths:
        raise FileNotFoundError(f"No prediction files matched: {pattern}")
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_read_json_records(path))
    if not records:
        raise ValueError("The prediction input is empty.")
    return records


def read_raw_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Original paired dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ValueError("The original paired dataset must be CSV or Parquet.")


def _prediction_id(record: dict[str, Any]) -> str:
    return _normalise(
        record.get("case_id")
        or record.get("ident")
        or record.get("id")
        or record.get("record_id")
    )


def attach_index(
    predictions: list[dict[str, Any]],
    index_path: str | None,
) -> str | None:
    """Attach tokens/IDs from the full-corpus index, preserving row order."""
    selected = index_path
    if not selected and any(not _prediction_id(record) for record in predictions):
        default_index = ROOT / "outputs" / "kg" / "full_index.jsonl"
        if default_index.exists():
            selected = str(default_index)
    if not selected:
        return None
    index = _read_json_records(_project_path(selected))
    if len(index) != len(predictions):
        raise ValueError(
            "Prediction/index row counts differ: "
            f"{len(predictions)} predictions versus {len(index)} index rows."
        )
    for prediction, lookup in zip(predictions, index):
        if not prediction.get("tokens"):
            prediction["tokens"] = list(lookup.get("tokens", []) or [])
        if not _prediction_id(prediction):
            for key in ("case_id", "ident", "id", "record_id"):
                value = lookup.get(key)
                if value not in (None, ""):
                    prediction[key] = value
                    break
        if lookup.get("problem_token_end") is not None:
            prediction["problem_token_end"] = lookup["problem_token_end"]
    return str(_project_path(selected))


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[Any, str] = {}
    cleaned: list[str] = []
    for column in frame.columns:
        value = str(column).strip().lstrip("\ufeff")
        if value in cleaned:
            raise ValueError(
                f"Two raw columns become the same name after trimming: {value!r}"
            )
        cleaned.append(value)
        mapping[column] = value
    return frame.rename(columns=mapping)


def join_raw_rows(
    predictions: list[dict[str, Any]],
    raw: pd.DataFrame,
    id_column: str,
    problem_column: str,
    solution_column: str,
    allow_positional: bool,
) -> list[tuple[int, dict[str, Any], dict[str, Any], str]]:
    raw = _clean_columns(raw)
    required = [id_column, problem_column, solution_column]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(
            f"Original dataset is missing columns {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    raw_ids = [_normalise(value) for value in raw[id_column].tolist()]
    duplicate_raw = sorted(
        value
        for value, count in Counter(raw_ids).items()
        if value and count > 1
    )
    if duplicate_raw:
        raise ValueError(
            "The original ID column is not unique. Example duplicate IDs: "
            + ", ".join(duplicate_raw[:10])
        )
    raw_by_id = {
        raw_id: raw.iloc[index].to_dict()
        for index, raw_id in enumerate(raw_ids)
        if raw_id
    }

    prediction_ids = [_prediction_id(record) for record in predictions]
    duplicate_predictions = sorted(
        value
        for value, count in Counter(prediction_ids).items()
        if value and count > 1
    )
    if duplicate_predictions:
        raise ValueError(
            "Prediction IDs are not unique. Example duplicate IDs: "
            + ", ".join(duplicate_predictions[:10])
        )

    if all(prediction_ids) and all(value in raw_by_id for value in prediction_ids):
        return [
            (index, prediction, raw_by_id[prediction_id], "id")
            for index, (prediction, prediction_id) in enumerate(
                zip(predictions, prediction_ids)
            )
        ]

    if not allow_positional:
        blank = sum(not value for value in prediction_ids)
        absent = sum(
            bool(value) and value not in raw_by_id for value in prediction_ids
        )
        raise ValueError(
            "Predictions could not be joined safely by ID "
            f"(blank prediction IDs={blank}, IDs absent from raw data={absent}). "
            "Pass --index outputs/kg/full_index.jsonl. Use "
            "--allow-positional-join only if prediction and raw row order were "
            "independently verified."
        )

    if len(predictions) != len(raw):
        raise ValueError(
            "Positional join refused because row counts differ: "
            f"{len(predictions)} predictions versus {len(raw)} raw rows."
        )
    return [
        (index, prediction, raw.iloc[index].to_dict(), "position")
        for index, prediction in enumerate(predictions)
    ]


def _surface(tokens: list[str], entity: dict[str, Any]) -> str:
    explicit = entity.get("text") or entity.get("surface")
    if explicit:
        return _normalise(explicit)
    try:
        start = int(entity["start"])
        end = int(entity["end"])
        if 0 <= start < end <= len(tokens):
            return _normalise(" ".join(tokens[start:end]))
    except (KeyError, TypeError, ValueError):
        pass
    return ""


def _token_form(value: Any) -> str:
    return (
        _normalise(value)
        .casefold()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def problem_token_boundary(
    prediction: dict[str, Any],
    problem_text: str,
) -> tuple[int | None, str]:
    """Find the end of the problem span in the combined SpERT token sequence."""
    tokens = list(prediction.get("tokens", []) or [])
    supplied = prediction.get("problem_token_end")
    try:
        supplied_int = int(supplied)
        if 0 <= supplied_int <= len(tokens):
            return supplied_int, "index_supplied"
    except (TypeError, ValueError):
        pass

    # Prefer the project's exact tokenizer when this script runs in the full
    # repository. This is the same tokenizer used by step 12.
    try:
        from src.data.preannotate import tokenize as project_tokenize

        problem_tokens = [token for token, _, _ in project_tokenize(problem_text)]
        prefix = tokens[: len(problem_tokens)]
        if [_token_form(value) for value in prefix] == [
            _token_form(value) for value in problem_tokens
        ]:
            return len(problem_tokens), "project_tokenizer_prefix"
    except (ImportError, TypeError, ValueError):
        pass

    # Conservative fallback: compare alphanumeric character streams until the
    # original problem text is exhausted. No entity is assigned to the problem
    # side if this alignment fails.
    target = _character_key(problem_text)
    accumulated = ""
    if target:
        for index, token in enumerate(tokens):
            accumulated += _character_key(token)
            if accumulated == target:
                return index + 1, "alphanumeric_prefix"
            if accumulated and not target.startswith(accumulated):
                break
    return None, "unresolved"


def entity_side(
    entity: dict[str, Any],
    surface_text: str,
    boundary: int | None,
    problem_text: str,
    solution_text: str,
) -> str:
    if boundary is not None:
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            if end <= boundary:
                return "problem"
            if start >= boundary:
                return "solution"
        except (KeyError, TypeError, ValueError):
            pass
    value = _key(surface_text)
    if not value:
        return "unknown"
    in_problem = value in _key(problem_text)
    in_solution = value in _key(solution_text)
    if in_problem and not in_solution:
        return "problem"
    if in_solution and not in_problem:
        return "solution"
    return "unknown"


def _resolve_entity_index(
    value: Any,
    raw_entities: list[dict[str, Any]],
) -> int | None:
    if isinstance(value, dict):
        for key in ("index", "entity_index"):
            candidate = value.get(key)
            if isinstance(candidate, int) and 0 <= candidate < len(raw_entities):
                return candidate
        if value.get("id") is not None:
            value = value["id"]
        else:
            signature = (
                value.get("start"),
                value.get("end"),
                value.get("type") or value.get("label"),
            )
            for index, entity in enumerate(raw_entities):
                other = (
                    entity.get("start"),
                    entity.get("end"),
                    entity.get("type") or entity.get("label"),
                )
                if signature == other:
                    return index
            return None
    if isinstance(value, int) and 0 <= value < len(raw_entities):
        return value
    for index, entity in enumerate(raw_entities):
        if str(entity.get("id", index)) == str(value):
            return index
    return None


def derive_action_role(
    action_text: str,
    addresses: list[str],
    investigates: list[str],
    outcomes: list[str],
    has_target: bool,
) -> str:
    """Derive a functional role without creating a new annotation label."""
    if addresses:
        return "corrective"
    if outcomes:
        return "verification"
    if investigates:
        return "diagnostic"
    action_key = _key(action_text)
    if has_target and any(
        action_key == hint or action_key.startswith(f"{hint} ")
        for hint in ENABLING_HINTS
    ):
        return "enabling"
    return "unresolved"


def _is_complete_solution(
    solution_text: str,
    raw_row: dict[str, Any],
    complete_column: str | None,
) -> bool:
    if not solution_text:
        return False
    if not complete_column:
        return True
    value = _normalise(raw_row.get(complete_column)).casefold()
    return value not in FALSE_VALUES


def load_cluster_map(
    path: str | None,
    id_column: str,
    cluster_column: str,
) -> dict[str, str]:
    if not path:
        return {}
    resolved = _project_path(path)
    frame = _clean_columns(
        pd.read_csv(resolved, dtype=str, keep_default_na=False)
    )
    required = {id_column, cluster_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Cluster map must contain {sorted(required)}; "
            f"available columns are {list(frame.columns)}"
        )
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        case_id = _normalise(row[id_column])
        cluster_id = _normalise(row[cluster_column])
        if not case_id or not cluster_id:
            continue
        if case_id in mapping and mapping[case_id] != cluster_id:
            raise ValueError(
                f"Cluster map assigns case {case_id!r} to two clusters."
            )
        mapping[case_id] = cluster_id
    return mapping


def build_case(
    row_number: int,
    prediction: dict[str, Any],
    raw_row: dict[str, Any],
    join_method: str,
    schema: SchemaCatalog,
    args: argparse.Namespace,
    cluster_lookup: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problem_text = _normalise(raw_row.get(args.problem_column))
    solution_text = _normalise(raw_row.get(args.solution_column))
    case_id = _normalise(raw_row.get(args.id_column) or _prediction_id(prediction))
    generated_case_id = False
    if not case_id:
        generated_case_id = True
        case_id = _stable_hash(
            "ROW-",
            f"{row_number}|{_key(problem_text)}|{_key(solution_text)}",
        )

    tokens = list(prediction.get("tokens", []) or [])
    boundary, boundary_method = problem_token_boundary(prediction, problem_text)
    raw_entities = list(prediction.get("entities", []) or [])
    entities: list[dict[str, Any]] = []
    for index, raw_entity in enumerate(raw_entities):
        entity_type = _normalise(
            raw_entity.get("type") or raw_entity.get("label")
        )
        text = _surface(tokens, raw_entity)
        entities.append(
            {
                "id": f"e{index}",
                "source_id": raw_entity.get("id"),
                "type": entity_type,
                "text": text,
                "start": raw_entity.get("start"),
                "end": raw_entity.get("end"),
                "confidence": raw_entity.get(
                    "confidence",
                    raw_entity.get("score", raw_entity.get("probability")),
                ),
                "side": entity_side(
                    raw_entity,
                    text,
                    boundary,
                    problem_text,
                    solution_text,
                ),
            }
        )

    valid_relations: list[dict[str, Any]] = []
    invalid_relations: list[dict[str, Any]] = []
    outgoing: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
    for relation_index, raw_relation in enumerate(
        prediction.get("relations", []) or []
    ):
        source_index = _resolve_entity_index(
            raw_relation.get("source", raw_relation.get("head")),
            raw_entities,
        )
        target_index = _resolve_entity_index(
            raw_relation.get("target", raw_relation.get("tail")),
            raw_entities,
        )
        relation_type = _normalise(
            raw_relation.get("type") or raw_relation.get("label")
        )
        invalid: dict[str, Any] | None = None
        if source_index is None or target_index is None:
            invalid = {
                "reason": "unresolved endpoint",
                "source_type": "",
                "target_type": "",
            }
        else:
            source_type = entities[source_index]["type"]
            target_type = entities[target_index]["type"]
            if not schema.valid_relation(
                relation_type, source_type, target_type
            ):
                invalid = {
                    "reason": "invalid endpoint signature",
                    "source_type": source_type,
                    "target_type": target_type,
                }
        if invalid is not None:
            invalid_relations.append(
                {
                    "case_id": case_id,
                    "relation_index": relation_index,
                    "relation_type": relation_type,
                    **invalid,
                    "raw_relation": json.dumps(
                        raw_relation, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
            continue

        assert source_index is not None and target_index is not None
        relation = {
            "id": f"r{relation_index}",
            "source_id": raw_relation.get("id"),
            "type": relation_type,
            "source": entities[source_index]["id"],
            "target": entities[target_index]["id"],
            "confidence": raw_relation.get(
                "confidence",
                raw_relation.get("score", raw_relation.get("probability")),
            ),
        }
        valid_relations.append(relation)
        outgoing[source_index].append((relation_type, target_index))

    def problem_values(entity_type: str) -> list[str]:
        return _unique(
            entity["text"]
            for entity in entities
            if entity["type"] == entity_type
            and entity.get("side") == "problem"
        )

    faults = problem_values("FAULT")
    abnormal = problem_values("ABN_PROC")
    observations = problem_values("TECH_OBS")
    contexts = problem_values("OP_CTX")
    locations = problem_values("LOC")
    problem_items = problem_values("MAINT_ITEM")
    issue_item_pairs: list[dict[str, str]] = []
    parts: list[dict[str, str]] = []
    for source_index, relations in outgoing.items():
        for relation_type, target_index in relations:
            if relation_type not in PROBLEM_RELATIONS:
                continue
            source = entities[source_index]
            target = entities[target_index]
            if (
                source.get("side") != "problem"
                or target.get("side") != "problem"
            ):
                continue
            if relation_type == "ISSUE_ON_ITEM":
                problem_items.append(target["text"])
                issue_item_pairs.append(
                    {"issue": source["text"], "item": target["text"]}
                )
            elif relation_type == "OBSERVATION_OF_ITEM":
                problem_items.append(target["text"])
            elif relation_type == "HAS_PART":
                parts.append(
                    {"whole": source["text"], "part": target["text"]}
                )

    action_indices = sorted(
        (
            index
            for index, entity in enumerate(entities)
            if entity["type"] == "ACTION"
            and entity.get("side") == "solution"
        ),
        key=lambda index: (
            entities[index].get("start")
            if entities[index].get("start") is not None
            else 10**9,
            index,
        ),
    )
    steps: list[dict[str, Any]] = []
    for order, action_index in enumerate(action_indices, start=1):
        buckets: defaultdict[str, list[str]] = defaultdict(list)
        evidence: list[str] = []
        for relation_type, target_index in outgoing.get(action_index, []):
            target = entities[target_index]["text"]
            if relation_type == "ACTION_ON_ITEM":
                buckets["targets"].append(target)
            elif relation_type == "ACTION_ADDRESSES_ISSUE":
                buckets["addresses"].append(target)
            elif relation_type == "ACTION_INVESTIGATES_ISSUE":
                buckets["investigates"].append(target)
            elif relation_type == "ACTION_USES_ITEM":
                buckets["used_items"].append(target)
            elif relation_type == "ACTION_RESULTS_IN_OUTCOME":
                buckets["outcomes"].append(target)
            elif relation_type == "ACTION_FOLLOWS_REFERENCE":
                buckets["references"].append(target)
            else:
                continue
            evidence.append(relation_type)
        action_text = entities[action_index]["text"]
        steps.append(
            {
                "order": order,
                "action": action_text,
                "role": derive_action_role(
                    action_text,
                    buckets["addresses"],
                    buckets["investigates"],
                    buckets["outcomes"],
                    bool(buckets["targets"]),
                ),
                "targets": _unique(buckets["targets"]),
                "issues": _unique(
                    [*buckets["addresses"], *buckets["investigates"]]
                ),
                "used_items": _unique(buckets["used_items"]),
                "outcomes": _unique(buckets["outcomes"]),
                "references": _unique(buckets["references"]),
                "relation_evidence": _unique(evidence),
            }
        )

    generated_cluster_id = False
    cluster_method = ""
    cluster_id = cluster_lookup.get(case_id, "")
    if cluster_id:
        cluster_method = "supplied_cluster_map"
    elif args.cluster_column:
        cluster_id = _normalise(raw_row.get(args.cluster_column))
        if cluster_id:
            cluster_method = "raw_cluster_column"
    if not cluster_id:
        generated_cluster_id = True
        cluster_id = _stable_hash("PX-", _key(problem_text))
        cluster_method = "exact_normalized_problem_hash"

    split = (
        _normalise(raw_row.get(args.split_column))
        if args.split_column
        else ""
    ) or "unspecified"
    complete = _is_complete_solution(
        solution_text, raw_row, args.complete_column
    )
    exact_group_id = _stable_hash(
        "EX-",
        f"{_key(problem_text)}||{_key(solution_text)}",
    )
    role_counts = Counter(step["role"] for step in steps)
    case = normalize_case(
        {
            "case_id": case_id,
            "source_dataset": args.source_dataset,
            "problem_text": problem_text,
            "solution_text": solution_text,
            "full_text": f"{problem_text} {solution_text}".strip(),
            "cluster_id": cluster_id,
            "exact_group_id": exact_group_id,
            "split": split,
            "entities": entities,
            "relations": valid_relations,
            "problem_graph": {
                "faults": faults,
                "abnormal_processes": abnormal,
                "issues": _unique([*faults, *abnormal]),
                "items": _unique(problem_items),
                "observations": observations,
                "contexts": contexts,
                "locations": locations,
                "issue_item_pairs": issue_item_pairs,
                "parts": parts,
            },
            "procedure": {
                "procedure_id": _stable_hash("P-", _key(solution_text)),
                "raw_solution_text": solution_text,
                "complete": complete,
                "steps": steps,
            },
            "metadata": {
                "schema": schema.title,
                "ie_model": "SpERT",
                "ie_tier": 4,
                "prediction_row": row_number,
                "raw_join_method": join_method,
                "problem_token_boundary": boundary,
                "problem_token_boundary_method": boundary_method,
                "cluster_method": cluster_method,
            },
            "quality": {
                "contract_version": "2.1",
                "generated_case_id": generated_case_id,
                "generated_cluster_id": generated_cluster_id,
                "problem_solution_boundary_verified": boundary is not None,
                "problem_graph_problem_only": True,
                "invalid_relations_quarantined": len(invalid_relations),
                "unresolved_actions": role_counts.get("unresolved", 0),
                "solution_marked_complete": complete,
            },
        },
        row_number,
    )
    return case, invalid_relations


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_parquet_if_available(
    cases: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    temporary = output_path.with_suffix(".tmp.parquet")
    try:
        rows: list[dict[str, Any]] = []
        for case in cases:
            row = dict(case)
            for key in (
                "entities",
                "relations",
                "problem_graph",
                "procedure",
                "metadata",
                "quality",
            ):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(output_path)
        return True
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        if output_path.exists():
            output_path.unlink()
        print(
            f"[warn] Optional Parquet export skipped: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False


def write_invalid_relations(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    fieldnames = [
        "case_id",
        "relation_index",
        "relation_type",
        "source_type",
        "target_type",
        "reason",
        "raw_relation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the v2 dashboard case library without solution leakage."
    )
    parser.add_argument(
        "--pred",
        default="outputs/kg/predictions_full.json",
        help="SpERT prediction JSON/JSONL path or glob",
    )
    parser.add_argument(
        "--raw",
        default="data/raw/Aircraft_Annotation_DataFile.csv",
        help="Original paired CSV or Parquet",
    )
    parser.add_argument(
        "--index",
        "--tokens",
        dest="index",
        help="Optional full_index JSON/JSONL carrying tokens and source IDs",
    )
    parser.add_argument("--name", default="aviation")
    parser.add_argument(
        "--output-dir",
        help="Override outputs/dashboard/<name>",
    )
    parser.add_argument("--source-dataset", default="MaintNet")
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--id-column", default="IDENT")
    parser.add_argument("--problem-column", default="PROBLEM")
    parser.add_argument("--solution-column", default="ACTION")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--complete-column")
    parser.add_argument("--cluster-column")
    parser.add_argument("--cluster-map")
    parser.add_argument("--cluster-map-id-column", default="case_id")
    parser.add_argument("--cluster-map-cluster-column", default="cluster_id")
    parser.add_argument(
        "--allow-positional-join",
        action="store_true",
        help="Use only when prediction/raw order is independently verified",
    )
    parser.add_argument(
        "--require-near-duplicate-map",
        action="store_true",
        help="Fail if any case uses only an exact problem hash as its cluster",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status when any blocking case finding remains",
    )
    args = parser.parse_args()

    schema = SchemaCatalog.from_yaml(args.schema)
    predictions = load_predictions(args.pred)
    attached_index = attach_index(predictions, args.index)
    raw = read_raw_table(_project_path(args.raw))
    joined = join_raw_rows(
        predictions,
        raw,
        args.id_column,
        args.problem_column,
        args.solution_column,
        args.allow_positional_join,
    )
    cluster_lookup = load_cluster_map(
        args.cluster_map,
        args.cluster_map_id_column,
        args.cluster_map_cluster_column,
    )

    output = (
        _project_path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / "dashboard" / args.name
    )
    output.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    invalid_relations: list[dict[str, Any]] = []
    for row_number, prediction, raw_row, join_method in joined:
        case, invalid = build_case(
            row_number,
            prediction,
            raw_row,
            join_method,
            schema,
            args,
            cluster_lookup,
        )
        cases.append(case)
        invalid_relations.extend(invalid)

    report = validate_cases(cases, schema)
    write_jsonl(cases, output / "cases.jsonl")
    parquet_written = write_parquet_if_available(
        cases, output / "cases.parquet"
    )
    write_invalid_relations(
        invalid_relations, output / "invalid_relations.csv"
    )
    (output / "validation.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    roles = Counter(
        step.get("role", "unresolved")
        for case in cases
        for step in (case.get("procedure", {}) or {}).get("steps", [])
    )
    joins = Counter(
        (case.get("metadata", {}) or {}).get("raw_join_method", "unknown")
        for case in cases
    )
    boundaries = Counter(
        (case.get("metadata", {}) or {}).get(
            "problem_token_boundary_method", "unknown"
        )
        for case in cases
    )
    cluster_methods = Counter(
        (case.get("metadata", {}) or {}).get("cluster_method", "unknown")
        for case in cases
    )
    excluded = len(cases) - report.recommendation_ready_cases
    metadata = {
        "dataset": {
            "id": args.source_dataset,
            "display_name": (
                f"{args.source_dataset} — SpERT-predicted maintenance cases"
            ),
            "domain": "Aviation maintenance work-order narratives",
            "source_name": args.source_dataset,
            "record_unit": (
                "One original problem narrative paired with one original "
                "historical solution narrative"
            ),
            "language": "English",
            "corpus_status": (
                "Automatically extracted silver corpus; not manual gold"
            ),
            "ie_model": "SpERT (Tier 4), AviMaint-DSS-IE v1.1",
            "description": (
                "Dashboard case library joined to the original problem and "
                "solution fields. Retrieval indexes only problem_text."
            ),
        },
        "fields": {
            "case_id": "Original source-record identifier.",
            "problem_text": "Original problem narrative; the only text indexed for retrieval.",
            "solution_text": "Original complete historical action/solution narrative.",
            "cluster_id": "Problem-only exact or near-duplicate group used for independent support.",
            "problem_graph": "Problem-side SpERT entities and valid relations.",
            "procedure": "One complete historical solution and its ordered extracted actions.",
        },
        "computed": {
            "records": len(cases),
            "recommendation_ready": report.recommendation_ready_cases,
            "excluded_from_recommendation": excluded,
            "unique_clusters": len(
                {str(case.get("cluster_id")) for case in cases}
            ),
            "invalid_relations_quarantined": len(invalid_relations),
            "action_roles": dict(roles),
            "join_methods": dict(joins),
            "boundary_methods": dict(boundaries),
            "cluster_methods": dict(cluster_methods),
            "parquet_written": parquet_written,
            "prediction_index": attached_index or "not supplied",
        },
        "limitations": [
            "SpERT predictions are silver data, not manually reviewed gold annotations.",
            "A missing outcome means unknown; it does not prove success or failure.",
            "Only valid AviMaint-DSS-IE v1.1 relation signatures are used as evidence.",
            "Final recommender claims require a manually reviewed, cluster-safe test set.",
            *(
                [
                    "Some cluster IDs are exact normalized problem hashes. Supply "
                    "--cluster-map for final near-duplicate-safe evaluation."
                ]
                if cluster_methods.get("exact_normalized_problem_hash")
                else []
            ),
            *(
                [
                    "Some problem/solution token boundaries could not be aligned. "
                    "Those records use conservative text-only retrieval features."
                ]
                if boundaries.get("unresolved")
                else []
            ),
        ],
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    issue_counts = report.issue_counts()
    summary_lines = [
        "AviMaint-DSS dashboard case-library build",
        f"records={len(cases)}",
        f"recommendation_ready={report.recommendation_ready_cases}",
        f"excluded_from_recommendation={excluded}",
        f"unique_problem_clusters={metadata['computed']['unique_clusters']}",
        f"invalid_relations_quarantined={len(invalid_relations)}",
        f"blocking_findings={report.blocking_count}",
        f"warnings={report.warning_count}",
        f"action_roles={dict(roles)}",
        f"cluster_methods={dict(cluster_methods)}",
        f"boundary_methods={dict(boundaries)}",
        f"validation_issue_counts={issue_counts}",
    ]
    (output / "cases_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines))
    print(f"Output: {output.resolve()}")

    if report.recommendation_ready_cases == 0:
        print(
            "FAILED: no recommendation-ready cases were produced. "
            "Inspect validation.json.",
            file=sys.stderr,
        )
        return 2
    if (
        args.require_near_duplicate_map
        and cluster_methods.get("exact_normalized_problem_hash")
    ):
        print(
            "FAILED: final evaluation requires a supplied near-duplicate cluster map.",
            file=sys.stderr,
        )
        return 2
    if args.strict and report.blocking_count:
        print(
            "FAILED STRICT CHECK: blocking records remain. "
            "They are excluded by the dashboard; inspect validation.json.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

