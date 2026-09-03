"""Create RQ2 core-schema views from the frozen full-hybrid annotations.

This is a deterministic ablation, not a second annotation project. The full
view is always authoritative. The core view removes REFERENCE spans and
ACTION_FOLLOWS_REFERENCE links, then rebuilds entity indices and BIO tags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_ENTITIES = {"REFERENCE"}
EXCLUDED_RELATIONS = {"ACTION_FOLLOWS_REFERENCE"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def derive_core(row: dict[str, Any]) -> dict[str, Any]:
    kept_entities: list[dict[str, Any]] = []
    old_to_new: dict[int, int] = {}
    for old_index, entity in enumerate(row["entities"]):
        if entity["type"] in EXCLUDED_ENTITIES:
            continue
        old_to_new[old_index] = len(kept_entities)
        kept_entities.append(entity)

    kept_relations: list[dict[str, Any]] = []
    for relation in row["relations"]:
        if relation["type"] in EXCLUDED_RELATIONS:
            continue
        if relation["head"] not in old_to_new or relation["tail"] not in old_to_new:
            continue
        kept_relations.append(
            {
                **relation,
                "head": old_to_new[relation["head"]],
                "tail": old_to_new[relation["tail"]],
            }
        )

    bio = ["O"] * len(row["tokens"])
    for entity in kept_entities:
        bio[entity["start"]] = f"B-{entity['type']}"
        for index in range(entity["start"] + 1, entity["end"]):
            bio[index] = f"I-{entity['type']}"
    return {
        **row,
        "schema_view": "aviation_core_v1",
        "entities": kept_entities,
        "relations": kept_relations,
        "bio": bio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "legacy_import/maintenance-ie/outputs/gold",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "legacy_import/maintenance-ie/outputs/gold_core",
    )
    args = parser.parse_args()

    total = removed_entities = removed_relations = 0
    for source_path in sorted(args.input_dir.glob("*.jsonl")):
        rows = read_jsonl(source_path)
        core_rows = [derive_core(row) for row in rows]
        removed_entities += sum(
            len(row["entities"]) - len(core["entities"]) for row, core in zip(rows, core_rows)
        )
        removed_relations += sum(
            len(row["relations"]) - len(core["relations"]) for row, core in zip(rows, core_rows)
        )
        total += len(rows)
        write_jsonl(args.output_dir / source_path.name, core_rows)

    if total == 0:
        raise SystemExit(f"No JSONL files found in {args.input_dir}")
    print(f"Core-schema view: {total} records -> {args.output_dir}")
    print(f"Excluded hybrid spans: {removed_entities}; excluded hybrid links: {removed_relations}")


if __name__ == "__main__":
    main()
