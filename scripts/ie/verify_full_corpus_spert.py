"""Strict verifier for the final operational full-corpus SpERT extraction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_index(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise SystemExit(
                        f"Invalid index JSONL line {line_no}: {exc}"
                    ) from exc
    return rows


def labels(section) -> set[str]:
    if isinstance(section, dict):
        return set(section)
    if isinstance(section, list):
        result = set()
        for item in section:
            if isinstance(item, str):
                result.add(item)
            elif isinstance(item, dict):
                value = item.get("type") or item.get("name") or item.get("label")
                if value:
                    result.add(str(value))
        return result
    return set()


def main():
    p = argparse.ArgumentParser()
    for name in (
        "dataset", "predictions", "index", "types", "model",
        "prep-manifest", "model-registry", "manifest",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--representation", required=True)
    p.add_argument("--model-system", required=True)
    p.add_argument("--expected-records", type=int, default=6169)
    p.add_argument("--expected-entity-types", type=int, default=9)
    p.add_argument("--expected-relation-types", type=int, default=11)
    args = p.parse_args()

    paths = {
        "dataset": Path(args.dataset),
        "predictions": Path(args.predictions),
        "index": Path(args.index),
        "types": Path(args.types),
        "model": Path(args.model),
        "prep": Path(args.prep_manifest),
        "registry": Path(args.model_registry),
        "manifest": Path(args.manifest),
    }
    for name, path in paths.items():
        if name != "manifest" and not path.exists():
            raise SystemExit(f"Missing required artifact: {path}")

    dataset = load_json(paths["dataset"])
    predictions = load_json(paths["predictions"])
    index = load_index(paths["index"])
    types = load_json(paths["types"])
    prep = load_json(paths["prep"])
    registry = load_json(paths["registry"])

    if prep.get("representation") != args.representation:
        raise SystemExit(
            f"Preparation representation={prep.get('representation')!r}; "
            f"expected={args.representation!r}"
        )
    parity = prep.get("representation_parity", {})
    if (
        parity.get("status") != "pass"
        or int(parity.get("projected_equals_trained_export", 0)) != 1600
        or int(parity.get("operational_tokenizer_matches", 0)) != 1600
        or int(parity.get("mismatches", -1)) != 0
    ):
        raise SystemExit(
            "Preparation manifest lacks complete 1600/1600 representation parity."
        )

    expected = args.expected_records
    if (
        not isinstance(dataset, list)
        or not isinstance(predictions, list)
        or len(dataset) != expected
        or len(predictions) != expected
        or len(index) != expected
    ):
        raise SystemExit(
            f"Record count mismatch: dataset={len(dataset)} "
            f"predictions={len(predictions)} index={len(index)} "
            f"expected={expected}"
        )

    ids = [str(row.get("ident", "")).strip() for row in index]
    if any(not rid for rid in ids) or len(set(ids)) != expected:
        raise SystemExit(
            f"IDENT validation failed: unique={len(set(ids))}, "
            f"expected={expected}"
        )
    if {
        str(row.get("representation", "")).strip() for row in index
    } != {args.representation}:
        raise SystemExit("Index contains an unexpected representation.")
    if any(
        "problem_raw" not in row or "action_raw" not in row
        for row in index
    ):
        raise SystemExit("Raw source provenance is missing from index rows.")

    entity_types = labels(types.get("entities"))
    relation_types = labels(types.get("relations"))
    if len(entity_types) != args.expected_entity_types:
        raise SystemExit(
            f"Entity type count={len(entity_types)}, "
            f"expected={args.expected_entity_types}"
        )
    if len(relation_types) != args.expected_relation_types:
        raise SystemExit(
            f"Relation type count={len(relation_types)}, "
            f"expected={args.expected_relation_types}"
        )

    entity_mentions = 0
    relation_mentions = 0

    for i, (source, pred, idx) in enumerate(
        zip(dataset, predictions, index)
    ):
        source_tokens = source.get("tokens")
        if source_tokens != idx.get("tokens"):
            raise SystemExit(
                f"Dataset/index token mismatch at row={i}, IDENT={ids[i]}"
            )
        if pred.get("tokens") is not None and pred.get("tokens") != source_tokens:
            raise SystemExit(
                f"Prediction token mismatch at row={i}, IDENT={ids[i]}"
            )

        entities = pred.get("entities", [])
        relations = pred.get("relations", [])

        for j, entity in enumerate(entities):
            etype = entity.get("type")
            if etype not in entity_types:
                raise SystemExit(
                    f"Unknown entity type row={i} entity={j}: {etype}"
                )
            start, end = entity.get("start"), entity.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start < end <= len(source_tokens))
            ):
                raise SystemExit(
                    f"Invalid entity span row={i} entity={j}: "
                    f"{start}:{end}"
                )

        for j, relation in enumerate(relations):
            rtype = relation.get("type")
            if rtype not in relation_types:
                raise SystemExit(
                    f"Unknown relation type row={i} relation={j}: {rtype}"
                )
            head, tail = relation.get("head"), relation.get("tail")
            if (
                not isinstance(head, int)
                or not isinstance(tail, int)
                or not (0 <= head < len(entities))
                or not (0 <= tail < len(entities))
            ):
                raise SystemExit(
                    f"Invalid relation endpoint row={i} relation={j}: "
                    f"head={head} tail={tail}"
                )

        entity_mentions += len(entities)
        relation_mentions += len(relations)

    registered = registry.get(args.model_system)
    if not isinstance(registered, dict):
        raise SystemExit(
            f"Model registry has no {args.model_system!r} entry."
        )
    registered_model = str(registered.get("final_model_path", ""))
    if not registered_model:
        raise SystemExit("Registered model path is empty.")

    weight = next(
        (
            p for p in (
                paths["model"] / "model.safetensors",
                paths["model"] / "pytorch_model.bin",
            )
            if p.is_file()
        ),
        None,
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "artifact_role": (
            "final operational normalized silver structured extraction"
        ),
        "training_performed": False,
        "representation": args.representation,
        "normalization_system": "Selective ByT5",
        "records": expected,
        "unique_identifiers": len(set(ids)),
        "raw_source_retained": True,
        "schema": {
            "entity_type_count": len(entity_types),
            "relation_type_count": len(relation_types),
            "entity_types": sorted(entity_types),
            "relation_types": sorted(relation_types),
        },
        "prediction_counts": {
            "entity_mentions": entity_mentions,
            "relation_mentions": relation_mentions,
        },
        "model": {
            "registry_system": args.model_system,
            "path": str(paths["model"].resolve()),
            "registry_final_model_path": registered_model,
            "weights": str(weight.resolve()) if weight else None,
            "weights_sha256": sha256(weight) if weight else None,
        },
        "provenance": {
            "preparation_manifest_sha256": sha256(paths["prep"]),
            "model_registry_sha256": sha256(paths["registry"]),
        },
        "artifacts": {
            "dataset_sha256": sha256(paths["dataset"]),
            "predictions_sha256": sha256(paths["predictions"]),
            "index_sha256": sha256(paths["index"]),
            "types_sha256": sha256(paths["types"]),
        },
        "checks": {
            "records_6169": True,
            "unique_identifiers_6169": True,
            "projected_equals_trained_1600_of_1600": True,
            "tokenizer_matches_trained_1600_of_1600": True,
            "raw_source_provenance_retained": True,
            "dataset_index_tokens_match": True,
            "prediction_tokens_match_when_present": True,
            "entity_spans_valid": True,
            "relation_endpoints_valid": True,
        },
    }

    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("FINAL SELECTIVE-BYT5 -> SpERT FULL-CORPUS EXTRACTION VERIFIED")
    print("=" * 92)
    print(f"records            : {expected}/{expected}")
    print(f"unique IDENTs      : {len(set(ids))}")
    print(f"representation     : {args.representation}")
    print(
        f"schema             : {len(entity_types)} entities / "
        f"{len(relation_types)} relations"
    )
    print(f"entity mentions    : {entity_mentions}")
    print(f"relation mentions  : {relation_mentions}")
    print("raw source retained: True")
    print("training performed : False")
    print(f"model              : {paths['model'].resolve()}")
    print(f"manifest           : {paths['manifest']}")
    print("=" * 92)


if __name__ == "__main__":
    main()
