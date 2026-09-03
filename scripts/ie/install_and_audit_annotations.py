"""Install and structurally audit the authoritative aviation gold annotations.

The original JSONL batches are copied into ``data/.../source_gold`` unchanged.
This script writes an audited training view to ``data/.../gold`` and mirrors it
to the legacy IE runtime. Corrections are deliberately limited to objective,
reproducible consistency defects:

* recompute exact-duplicate groups from the visible token sequence;
* remove identical duplicated relation triples; and
* harmonize exact duplicate records only when one annotation is a strict
  superset of the other and their entity spans are identical.

No entity span, entity type, or non-duplicate expert relation is rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ("pilot", "round1", "rare1", "rare2", "rare3")
EXPECTED_COUNTS = {"pilot": 300, "round1": 500, "rare1": 300, "rare2": 300, "rare3": 200}
RANDOM_BATCHES = {"pilot", "round1"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def exact_group(tokens: list[str]) -> str:
    visible = "\u241f".join(str(token).casefold() for token in tokens)
    return "ex_" + hashlib.sha256(visible.encode("utf-8")).hexdigest()[:16]


def entity_signature(entity: dict[str, Any]) -> tuple[str, int, int]:
    return str(entity["type"]), int(entity["start"]), int(entity["end"])


def relation_signature(relation: dict[str, Any]) -> tuple[str, int, int]:
    return str(relation["type"]), int(relation["head"]), int(relation["tail"])


def validate_record(
    record: dict[str, Any],
    entity_types: set[str],
    relation_rules: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required = {"ident", "tokens", "bio", "entities", "relations"}
    missing = sorted(required - set(record))
    if missing:
        return [f"missing keys: {', '.join(missing)}"]

    tokens = record["tokens"]
    entities = record["entities"]
    relations = record["relations"]
    bio = record["bio"]
    if not isinstance(tokens, list) or not tokens:
        errors.append("tokens must be a non-empty list")
        return errors
    if len(bio) != len(tokens):
        errors.append(f"BIO length {len(bio)} differs from token length {len(tokens)}")

    expected_bio = ["O"] * len(tokens)
    occupied: list[tuple[int, int]] = []
    for index, entity in enumerate(entities):
        entity_type = entity.get("type")
        start, end = entity.get("start"), entity.get("end")
        if entity_type not in entity_types:
            errors.append(f"entity {index} has unknown type {entity_type!r}")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not (0 <= start < end <= len(tokens))
        ):
            errors.append(f"entity {index} has invalid span {start}:{end}")
            continue
        if any(
            not (end <= other_start or start >= other_end) for other_start, other_end in occupied
        ):
            errors.append(f"entity {index} overlaps an earlier entity")
        occupied.append((start, end))
        expected_bio[start] = f"B-{entity_type}"
        for token_index in range(start + 1, end):
            expected_bio[token_index] = f"I-{entity_type}"
    if bio != expected_bio:
        errors.append("BIO tags and entity spans disagree")

    for index, relation in enumerate(relations):
        relation_type = relation.get("type")
        head, tail = relation.get("head"), relation.get("tail")
        rule = relation_rules.get(str(relation_type))
        if rule is None:
            errors.append(f"relation {index} has unknown type {relation_type!r}")
            continue
        if (
            not isinstance(head, int)
            or not isinstance(tail, int)
            or not (0 <= head < len(entities) and 0 <= tail < len(entities))
        ):
            errors.append(f"relation {index} has an invalid entity index")
            continue
        head_type, tail_type = entities[head]["type"], entities[tail]["type"]
        if head_type not in rule["head"] or tail_type not in rule["tail"]:
            errors.append(f"relation {index} violates {relation_type}: {head_type}->{tail_type}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/aviation/annotations/source_gold",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=ROOT / "data/aviation/annotations/gold",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / "legacy_import/maintenance-ie/outputs/gold",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs/reports/annotation_audit",
    )
    args = parser.parse_args()

    schema_path = ROOT / "legacy_import/maintenance-ie/config/schema.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    entity_types = set(schema["entities"])
    relation_rules = schema["relations"]

    records: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    structural_errors: list[dict[str, Any]] = []
    per_batch: dict[str, list[dict[str, Any]]] = {}

    for batch in BATCHES:
        source_path = args.source_dir / f"{batch}.jsonl"
        if not source_path.is_file():
            raise SystemExit(f"Missing authoritative batch: {source_path}")
        batch_rows = read_jsonl(source_path)
        if len(batch_rows) != EXPECTED_COUNTS[batch]:
            raise SystemExit(
                f"{batch} has {len(batch_rows)} records; expected {EXPECTED_COUNTS[batch]}"
            )
        for line_number, row in enumerate(batch_rows, start=1):
            row = json.loads(json.dumps(row))
            row["annotation_batch"] = batch
            row["sampling_population"] = "random" if batch in RANDOM_BATCHES else "rare_enriched"
            old_group = row.get("exact_group_id")
            new_group = exact_group(row["tokens"])
            row["source_exact_group_id"] = old_group
            row["exact_group_id"] = new_group
            if old_group != new_group:
                corrections.append(
                    {
                        "batch": batch,
                        "line": line_number,
                        "ident": row.get("ident"),
                        "type": "exact_group_recomputed",
                        "detail": f"{old_group}->{new_group}",
                    }
                )

            unique_relations: list[dict[str, Any]] = []
            seen_relations: set[tuple[str, int, int]] = set()
            for relation in row.get("relations", []):
                signature = relation_signature(relation)
                if signature in seen_relations:
                    corrections.append(
                        {
                            "batch": batch,
                            "line": line_number,
                            "ident": row.get("ident"),
                            "type": "duplicate_relation_removed",
                            "detail": repr(signature),
                        }
                    )
                    continue
                seen_relations.add(signature)
                unique_relations.append(relation)
            row["relations"] = unique_relations

            for error in validate_record(row, entity_types, relation_rules):
                structural_errors.append(
                    {
                        "batch": batch,
                        "line": line_number,
                        "ident": row.get("ident"),
                        "error": error,
                    }
                )
            records.append(row)
        per_batch[batch] = records[-len(batch_rows) :]

    identifiers = [str(row["ident"]) for row in records]
    duplicate_identifiers = sorted(
        ident for ident, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_identifiers:
        structural_errors.append(
            {
                "batch": "all",
                "line": 0,
                "ident": "",
                "error": f"duplicate identifiers: {duplicate_identifiers}",
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[row["exact_group_id"]].append(row)

    unresolved_groups: list[str] = []
    harmonized_records = 0
    for group_id, duplicates in by_group.items():
        if len(duplicates) < 2:
            continue
        entity_sets = [
            tuple(entity_signature(entity) for entity in row["entities"]) for row in duplicates
        ]
        if len(set(entity_sets)) != 1:
            unresolved_groups.append(group_id)
            continue
        relation_sets = [set(map(relation_signature, row["relations"])) for row in duplicates]
        richest_index = max(range(len(duplicates)), key=lambda index: len(relation_sets[index]))
        richest = relation_sets[richest_index]
        if not all(candidate <= richest for candidate in relation_sets):
            unresolved_groups.append(group_id)
            continue
        canonical_relations = duplicates[richest_index]["relations"]
        for row, existing in zip(duplicates, relation_sets):
            if existing == richest:
                continue
            row["relations"] = json.loads(json.dumps(canonical_relations))
            harmonized_records += 1
            corrections.append(
                {
                    "batch": row["annotation_batch"],
                    "line": 0,
                    "ident": row["ident"],
                    "type": "exact_duplicate_relation_harmonized",
                    "detail": f"group={group_id}; {len(existing)}->{len(richest)} relations",
                }
            )

    if structural_errors:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "structural_errors.json").write_text(
            json.dumps(structural_errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(
            f"Annotation installation stopped: {len(structural_errors)} structural errors"
        )

    by_batch_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_batch_output[row["annotation_batch"]].append(row)
    for batch in BATCHES:
        write_jsonl(args.gold_dir / f"{batch}.jsonl", by_batch_output[batch])
        write_jsonl(args.runtime_dir / f"{batch}.jsonl", by_batch_output[batch])

    entity_support = Counter(entity["type"] for row in records for entity in row["entities"])
    relation_support = Counter(relation["type"] for row in records for relation in row["relations"])
    batch_summary: dict[str, Any] = {}
    for batch, rows in by_batch_output.items():
        batch_summary[batch] = {
            "records": len(rows),
            "sampling_population": "random" if batch in RANDOM_BATCHES else "rare_enriched",
            "entities": dict(
                sorted(Counter(e["type"] for r in rows for e in r["entities"]).items())
            ),
            "relations": dict(
                sorted(Counter(rel["type"] for r in rows for rel in r["relations"]).items())
            ),
        }

    report = {
        "schema": "aviation_compact_v1",
        "source_batches_are_immutable": True,
        "records": len(records),
        "unique_identifiers": len(set(identifiers)),
        "random_records": sum(len(by_batch_output[batch]) for batch in RANDOM_BATCHES),
        "rare_enriched_records": sum(
            len(by_batch_output[batch]) for batch in BATCHES if batch not in RANDOM_BATCHES
        ),
        "exact_duplicate_groups": sum(len(rows) > 1 for rows in by_group.values()),
        "largest_exact_duplicate_group": max(map(len, by_group.values())),
        "unresolved_exact_duplicate_groups": unresolved_groups,
        "structural_errors": 0,
        "duplicate_relation_instances_removed": sum(
            c["type"] == "duplicate_relation_removed" for c in corrections
        ),
        "duplicate_records_harmonized": harmonized_records,
        "entity_support": dict(sorted(entity_support.items())),
        "relation_support": dict(sorted(relation_support.items())),
        "batches": batch_summary,
        "split_policy": {
            "development_and_test_source": ["pilot", "round1"],
            "rare_batches_train_only": ["rare1", "rare2", "rare3"],
            "group_key": "exact_group_id",
            "seed": 42,
            "test_records": 225,
            "development_records": 100,
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "annotation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.report_dir / "annotation_corrections.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "line", "ident", "type", "detail"])
        writer.writeheader()
        writer.writerows(corrections)

    print(f"Installed {len(records)} structurally valid gold records.")
    print("Random population: 800 (pilot + round1); rare-enriched training population: 800.")
    print(
        f"Exact duplicate groups: {report['exact_duplicate_groups']}; "
        f"unresolved: {len(unresolved_groups)}."
    )
    print(
        f"Mechanical duplicate relations removed: {report['duplicate_relation_instances_removed']}."
    )
    print(f"Duplicate records harmonized: {harmonized_records}.")
    print(f"Audit report: {args.report_dir / 'annotation_audit.json'}")


if __name__ == "__main__":
    main()
