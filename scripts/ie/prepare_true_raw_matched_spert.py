"""Prepare ONLY the missing true System-A raw SpERT condition.

The prior matched-normalization experiment already trained:
  rules, byt5, selective_byt5, rules_then_byt5

A representation audit established that historical outputs/spert is a legacy
normalized annotation baseline. This script therefore creates:
  outputs/spert_normalized/raw/{train,dev,test}.json

from outputs/gold_variants/raw, preserving the exact frozen split membership and
order. The established full 9x11 SpERT configuration is reused only as the
fixed architecture/hyperparameter source.

V3 fix:
The prior V2 compared a raw TEST support dictionary containing
{"records", "entities", "relations"} against existing-condition support
dictionaries containing only {"entities", "relations"}. This caused a false
failure even when entity/relation support was identical. V3 compares the same
keys on both sides and separately checks record count/order.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
REFERENCE = IE / "outputs" / "spert"
VARIANTS = IE / "outputs" / "gold_variants"
BASE = IE / "outputs" / "spert_normalized"
TARGET = BASE / "raw"
SYSTEMS = ["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"]
EXISTING = SYSTEMS[1:]
EXPECTED = {"train": 1275, "dev": 100, "test": 225}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_id(doc: dict) -> str:
    rid = doc.get("orig_id", doc.get("ident"))
    if rid is None or str(rid) == "":
        raise SystemExit("Document without orig_id/ident")
    return str(rid)


def variant_map(system: str) -> dict[str, dict]:
    folder = VARIANTS / system
    if not folder.exists():
        raise SystemExit(f"Missing projected gold folder: {folder}")
    out = {}
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("ident", "")).strip()
                if not rid:
                    raise SystemExit(f"{path}: projected record without ident")
                if rid in out:
                    raise SystemExit(f"Duplicate projected IDENT {rid}")
                out[rid] = rec
    return out


def projected_doc(rec: dict, rid: str) -> dict:
    return {
        "tokens": rec["tokens"],
        "entities": rec.get("entities", []),
        "relations": rec.get("relations", []),
        "orig_id": rid,
    }


def rewrite_config(source: str, target_dir: Path) -> None:
    root = target_dir.resolve().as_posix()
    replacements = {
        "label": "avimaint_spert_norm_raw",
        "train_path": f"{root}/train.json",
        "valid_path": f"{root}/dev.json",
        "types_path": f"{root}/avimaint_types.json",
        "save_path": f"{root}/save",
        "log_path": f"{root}/log",
    }
    text = source
    for key, value in replacements.items():
        pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
        if not pattern.search(text):
            raise SystemExit(f"Reference SpERT config missing key: {key}")
        text = pattern.sub(f"{key} = {value}", text)
    (target_dir / "avimaint_spert.conf").write_text(text, encoding="utf-8")


def main():
    audit = IE / "outputs" / "reports" / "normalization_spert_matched_v2" / "REPRESENTATION_AUDIT.json"
    if not audit.exists():
        raise SystemExit(
            "Representation audit missing. "
            "Run audit_spert_annotation_representation.py first."
        )
    audit_data = load_json(audit)
    if audit_data.get("status") != "pass":
        raise SystemExit("Representation audit did not pass.")

    for name in (
        "train.json", "dev.json", "test.json",
        "avimaint_types.json", "avimaint_spert.conf",
    ):
        if not (REFERENCE / name).exists():
            raise SystemExit(
                f"Missing historical reference artifact: {REFERENCE / name}"
            )

    frozen_path = IE / "outputs" / "splits.json"
    frozen = load_json(frozen_path)
    for split, expected in EXPECTED.items():
        vals = [str(x) for x in frozen.get(split, [])]
        if len(vals) != expected:
            raise SystemExit(
                f"Frozen {split}: expected {expected}, found {len(vals)}"
            )

    all_ids = [
        str(x)
        for split in ("train", "dev", "test")
        for x in frozen[split]
    ]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Frozen split overlap detected.")

    # Historical exports are used ONLY as the already-frozen order source.
    order = {}
    for split, expected in EXPECTED.items():
        docs = load_json(REFERENCE / f"{split}.json")
        if len(docs) != expected:
            raise SystemExit(f"Historical {split}.json count mismatch")
        ids = [doc_id(d) for d in docs]
        if set(ids) != set(str(x) for x in frozen[split]):
            raise SystemExit(
                f"Historical {split}.json membership differs from frozen split"
            )
        order[split] = ids

    raw_map = variant_map("raw")
    missing = [rid for rid in all_ids if rid not in raw_map]
    if missing:
        raise SystemExit(
            f"Projected raw condition missing {len(missing)} frozen IDs; "
            f"examples={missing[:10]}"
        )

    TARGET.mkdir(parents=True, exist_ok=True)

    raw_support = {}
    for split, ids in order.items():
        docs = [projected_doc(raw_map[rid], rid) for rid in ids]
        dump_json(TARGET / f"{split}.json", docs)
        raw_support[split] = {
            "records": len(docs),
            "entities": sum(len(d.get("entities", [])) for d in docs),
            "relations": sum(len(d.get("relations", [])) for d in docs),
        }

    shutil.copy2(
        REFERENCE / "avimaint_types.json",
        TARGET / "avimaint_types.json",
    )
    rewrite_config(
        (REFERENCE / "avimaint_spert.conf").read_text(encoding="utf-8"),
        TARGET,
    )

    types = load_json(TARGET / "avimaint_types.json")
    if (len(types.get("entities", {})), len(types.get("relations", {}))) != (9, 11):
        raise SystemExit("Expected full schema 9 entities / 11 relations.")

    raw_test_ids = [doc_id(d) for d in load_json(TARGET / "test.json")]

    # V3: compare identical support keys only.
    raw_test_support = {
        "entities": raw_support["test"]["entities"],
        "relations": raw_support["test"]["relations"],
    }

    existing_registry = {}
    for system in EXISTING:
        export = BASE / system
        for name in (
            "train.json", "dev.json", "test.json",
            "avimaint_types.json", "avimaint_spert.conf",
            "predictions_test.json",
        ):
            if not (export / name).exists():
                raise SystemExit(
                    f"Existing {system} condition missing {name}; "
                    "do not retrain blindly."
                )

        test_docs = load_json(export / "test.json")

        # Record-count/order checks are separate from support comparison.
        if len(test_docs) != EXPECTED["test"]:
            raise SystemExit(
                f"{system}: expected {EXPECTED['test']} TEST records, "
                f"found {len(test_docs)}"
            )
        if [doc_id(d) for d in test_docs] != raw_test_ids:
            raise SystemExit(
                f"{system}: TEST order/membership differs from corrected raw condition"
            )

        support = {
            "entities": sum(len(d.get("entities", [])) for d in test_docs),
            "relations": sum(len(d.get("relations", [])) for d in test_docs),
        }
        if support != raw_test_support:
            raise SystemExit(
                f"{system}: TEST support {support} differs from raw "
                f"{raw_test_support}"
            )

        models = (
            [p for p in (export / "save").rglob("final_model")]
            if (export / "save").exists()
            else []
        )
        models = [p for p in models if p.is_dir()]
        if not models:
            raise SystemExit(
                f"{system}: existing final_model not found; "
                "correction must reuse completed model."
            )
        newest = max(models, key=lambda p: p.stat().st_mtime)
        existing_registry[system] = str(newest.relative_to(ROOT))

        print(
            f"{system}: TEST records={len(test_docs)} "
            f"support={support} -> PASS; model reused"
        )

    manifest = {
        "status": "prepared_true_raw_condition_only",
        "correction": (
            "Historical outputs/spert was proven to use the legacy normalized "
            "annotation representation. A new System-A raw condition is "
            "therefore prepared from outputs/gold_variants/raw."
        ),
        "v3_fix": (
            "TEST support comparison now compares entities/relations against "
            "entities/relations; record count and order are checked separately."
        ),
        "split_counts": EXPECTED,
        "schema": {"entities": 9, "relations": 11},
        "raw_export": str(TARGET.relative_to(ROOT)),
        "raw_support": raw_support,
        "existing_four_models_reused_without_retraining": existing_registry,
        "configuration_source": str(
            (REFERENCE / "avimaint_spert.conf").relative_to(ROOT)
        ),
        "configuration_role": (
            "architecture/hyperparameter reference only; "
            "not a raw-representation baseline"
        ),
        "test_policy": (
            "no hyperparameter changes; evaluate corrected five-way table only "
            "after raw final_model and predictions exist"
        ),
        "post_test_retuning": "forbidden",
    }
    dump_json(BASE / "PREP_MANIFEST_V2.json", manifest)

    print("=" * 86)
    print("CORRECTED MATCHED SpERT EXPORT PREPARED")
    print("=" * 86)
    print("new condition        : TRUE System-A raw")
    print("target               :", TARGET)
    print("split                : train=1275 dev=100 test=225")
    print("raw TEST support     :", raw_test_support)
    print(
        "existing models      : reused rules / byt5 / "
        "selective_byt5 / rules_then_byt5"
    )
    print("models to train now  : 1 (raw only)")
    print("manifest             :", BASE / "PREP_MANIFEST_V2.json")
    print("=" * 86)


if __name__ == "__main__":
    main()
