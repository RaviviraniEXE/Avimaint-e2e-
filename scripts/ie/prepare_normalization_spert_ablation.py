"""Prepare five frozen-test normalization variants for SpERT prediction.

No training occurs here.

RAW is copied from the authoritative existing full-schema SpERT test.json so the
raw condition must reproduce the already-frozen SpERT result. The four
normalized conditions are assembled from projected gold variants and ordered by
the same frozen test IDs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
SPERT = IE / "outputs" / "spert"
GOLD_VARIANTS = IE / "outputs" / "gold_variants"
OUT = IE / "outputs" / "normalization_spert_ablation"
PREPARED = OUT / "prepared"

SYSTEMS = ["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_variant_map(system: str):
    mapping = {}
    folder = GOLD_VARIANTS / system
    if not folder.exists():
        raise SystemExit(f"Missing projected gold variant folder: {folder}")
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("ident", ""))
                if not rid:
                    raise SystemExit(f"{path}: record without ident")
                if rid in mapping:
                    raise SystemExit(f"Duplicate ident {rid} in {folder}")
                mapping[rid] = rec
    if not mapping:
        raise SystemExit(f"No JSONL records found in {folder}")
    return mapping


def load_test_ids(raw_test):
    ids = []
    for doc in raw_test:
        rid = doc.get("orig_id", doc.get("ident"))
        if rid is None:
            ids = []
            break
        ids.append(str(rid))
    if ids and len(ids) == len(raw_test):
        return ids, "outputs/spert/test.json::orig_id"

    split_path = IE / "outputs" / "splits.json"
    if not split_path.exists():
        raise SystemExit(
            "Raw SpERT test.json has no orig_id and outputs/splits.json is missing."
        )
    splits = load_json(split_path)
    ids = [str(x) for x in splits.get("test", [])]
    if len(ids) != len(raw_test):
        raise SystemExit(
            f"Could not establish test order: splits.test={len(ids)} vs "
            f"spert/test.json={len(raw_test)}"
        )
    return ids, "outputs/splits.json::test"


def entity_relation_schema(docs):
    ent_types = set()
    rel_types = set()
    ent_support = 0
    rel_support = 0
    for d in docs:
        ents = d.get("entities", [])
        rels = d.get("relations", [])
        ent_support += len(ents)
        rel_support += len(rels)
        ent_types.update(e.get("type") for e in ents)
        rel_types.update(r.get("type") for r in rels)
    ent_types.discard(None)
    rel_types.discard(None)
    return sorted(ent_types), sorted(rel_types), ent_support, rel_support


def load_qc(test_ids):
    qc_path = GOLD_VARIANTS / "projection_qc.csv"
    if not qc_path.exists():
        raise SystemExit(f"Missing projection QC: {qc_path}")
    wanted = set(test_ids)
    by_system = defaultdict(dict)
    with qc_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = str(row["record_id"])
            if rid not in wanted:
                continue
            by_system[row["system"]][rid] = row

    missing = []
    for system in SYSTEMS:
        for rid in test_ids:
            if rid not in by_system[system]:
                missing.append((system, rid))
    if missing:
        raise SystemExit(
            f"Projection QC missing {len(missing)} system/test-id rows; "
            f"examples={missing[:10]}"
        )
    return by_system


def truthy(value: str):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main():
    raw_test_path = SPERT / "test.json"
    if not raw_test_path.exists():
        raise SystemExit(f"Missing authoritative full-schema SpERT test: {raw_test_path}")

    raw_test = load_json(raw_test_path)
    if not isinstance(raw_test, list):
        raise SystemExit("SpERT test.json must contain a JSON list")
    if len(raw_test) != 225:
        raise SystemExit(
            f"Expected frozen aviation test size 225, found {len(raw_test)}. "
            "Refusing to run on a different split."
        )

    test_ids, order_source = load_test_ids(raw_test)
    if len(test_ids) != len(set(test_ids)):
        raise SystemExit("Duplicate IDs detected in frozen test order")

    ent_types, rel_types, ent_support, rel_support = entity_relation_schema(raw_test)
    if len(ent_types) != 9 or len(rel_types) != 11:
        raise SystemExit(
            "Expected FULL aviation schema 9 entities / 11 relations, found "
            f"{len(ent_types)} / {len(rel_types)}.\n"
            f"entities={ent_types}\nrelations={rel_types}"
        )
    if "REFERENCE" not in ent_types or "ACTION_FOLLOWS_REFERENCE" not in rel_types:
        raise SystemExit(
            "The authoritative test does not look like the FULL 9x11 schema. "
            "REFERENCE/ACTION_FOLLOWS_REFERENCE are required for this experiment."
        )

    PREPARED.mkdir(parents=True, exist_ok=True)

    # RAW is the exact authoritative SpERT test. Add orig_id only when absent so
    # evaluation can track records without altering tokens/entities/relations.
    raw_prepared = []
    for rid, doc in zip(test_ids, raw_test):
        out = dict(doc)
        out.setdefault("orig_id", rid)
        raw_prepared.append(out)
    dump_json(PREPARED / "raw_test.json", raw_prepared)

    for system in SYSTEMS[1:]:
        mapping = load_variant_map(system)
        missing = [rid for rid in test_ids if rid not in mapping]
        if missing:
            raise SystemExit(
                f"{system}: projected gold is missing {len(missing)} frozen test IDs; "
                f"examples={missing[:10]}"
            )
        docs = []
        for rid in test_ids:
            rec = mapping[rid]
            docs.append(
                {
                    "tokens": rec["tokens"],
                    "entities": rec.get("entities", []),
                    "relations": rec.get("relations", []),
                    "orig_id": rid,
                }
            )
        dump_json(PREPARED / f"{system}_test.json", docs)

    qc = load_qc(test_ids)
    common_ids = []
    projection_summary = {}
    for system in SYSTEMS:
        rows = [qc[system][rid] for rid in test_ids]
        dropped_e = sum(int(float(r["dropped_entities"])) for r in rows)
        dropped_r = sum(int(float(r["dropped_relations"])) for r in rows)
        full_records = 0
        coverages = []
        for r in rows:
            cov = float(r["entity_coverage"])
            coverages.append(cov)
            if (
                cov == 1.0
                and int(float(r["dropped_entities"])) == 0
                and int(float(r["dropped_relations"])) == 0
                and truthy(r["prediction_found"])
            ):
                full_records += 1
        projection_summary[system] = {
            "n_test": len(rows),
            "mean_record_entity_coverage": sum(coverages) / len(coverages),
            "dropped_entities_test": dropped_e,
            "dropped_relations_test": dropped_r,
            "full_projection_records": full_records,
        }

    for rid in test_ids:
        ok = True
        for system in SYSTEMS:
            r = qc[system][rid]
            if not (
                float(r["entity_coverage"]) == 1.0
                and int(float(r["dropped_entities"])) == 0
                and int(float(r["dropped_relations"])) == 0
                and truthy(r["prediction_found"])
            ):
                ok = False
                break
        if ok:
            common_ids.append(rid)

    dump_json(OUT / "common_full_projection_ids.json", common_ids)
    manifest = {
        "experiment": "normalization_to_frozen_spert_full_9x11",
        "training_performed": False,
        "systems": SYSTEMS,
        "frozen_test_n": len(test_ids),
        "test_order_source": order_source,
        "full_schema": {
            "entity_types": ent_types,
            "relation_types": rel_types,
            "gold_entity_support_raw": ent_support,
            "gold_relation_support_raw": rel_support,
        },
        "common_full_projection_n": len(common_ids),
        "projection_test_summary": projection_summary,
        "prepared_dir": str(PREPARED.relative_to(ROOT)),
        "prediction_driver": "scripts/ie/06_predict_existing_spert.ps1",
    }
    dump_json(OUT / "PREP_MANIFEST.json", manifest)

    print("=" * 72)
    print("FROZEN SpERT NORMALIZATION ABLATION - PREPARED")
    print("=" * 72)
    print(f"test records       : {len(test_ids)}")
    print(f"schema             : {len(ent_types)} entities / {len(rel_types)} relations")
    print(f"raw gold support   : entities={ent_support} relations={rel_support}")
    print(f"common full-proj   : {len(common_ids)}/{len(test_ids)} records")
    for system in SYSTEMS:
        s = projection_summary[system]
        print(
            f"{system:17s}: full={s['full_projection_records']:3d}/{len(test_ids)} "
            f"dropped_ent={s['dropped_entities_test']} "
            f"dropped_rel={s['dropped_relations_test']}"
        )
    print(f"prepared -> {PREPARED}")


if __name__ == "__main__":
    main()
