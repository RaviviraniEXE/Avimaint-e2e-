"""Prepare representation-matched SpERT train/dev/test exports.

This is the main downstream normalization ablation preparation step.
The frozen aviation split membership is never changed.  The authoritative raw
SpERT export remains untouched.  Four normalized representations are exported
under outputs/spert_normalized/<system>/ using the SAME record IDs/order as the
raw SpERT train/dev/test files.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
RAW = IE / "outputs" / "spert"
VARIANTS = IE / "outputs" / "gold_variants"
OUT = IE / "outputs" / "spert_normalized"
REPORT_OLD = IE / "outputs" / "reports" / "normalization_spert_ablation"

SYSTEMS = ["rules", "byt5", "selective_byt5", "rules_then_byt5"]
EXPECTED_COUNTS = {"train": 1275, "dev": 100, "test": 225}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_id(doc: dict) -> str:
    rid = doc.get("orig_id", doc.get("ident"))
    if rid is None or str(rid) == "":
        raise SystemExit("A raw SpERT document has no orig_id/ident; cannot preserve frozen split identity.")
    return str(rid)


def load_variant_map(system: str) -> dict[str, dict]:
    folder = VARIANTS / system
    if not folder.exists():
        raise SystemExit(f"Missing projected gold folder: {folder}")
    mapping: dict[str, dict] = {}
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
        raise SystemExit(f"No projected records found in {folder}")
    return mapping


def projected_doc(rec: dict, rid: str) -> dict:
    return {
        "tokens": rec["tokens"],
        "entities": rec.get("entities", []),
        "relations": rec.get("relations", []),
        "orig_id": rid,
    }


def rewrite_config(raw_text: str, target: Path, system: str) -> None:
    root = target.resolve().as_posix()
    replacements = {
        "label": f"avimaint_spert_norm_{system}",
        "train_path": f"{root}/train.json",
        "valid_path": f"{root}/dev.json",
        "types_path": f"{root}/avimaint_types.json",
        "save_path": f"{root}/save",
        "log_path": f"{root}/log",
    }
    text = raw_text
    for key, value in replacements.items():
        pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
        if not pattern.search(text):
            raise SystemExit(f"Raw SpERT config is missing required key: {key}")
        text = pattern.sub(f"{key} = {value}", text)
    (target / "avimaint_spert.conf").write_text(text, encoding="utf-8")


def split_projection_summary(raw_order: dict[str, list[str]]) -> dict:
    qc_path = VARIANTS / "projection_qc.csv"
    if not qc_path.exists():
        raise SystemExit(f"Missing projection QC: {qc_path}")
    split_of = {}
    for split, ids in raw_order.items():
        for rid in ids:
            split_of[rid] = split
    summary = {s: {k: {"dropped_entities": 0, "dropped_relations": 0, "rows": 0}
                   for k in EXPECTED_COUNTS} for s in ["raw"] + SYSTEMS}
    with qc_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = str(row["record_id"])
            split = split_of.get(rid)
            system = row.get("system")
            if split is None or system not in summary:
                continue
            d = summary[system][split]
            d["rows"] += 1
            d["dropped_entities"] += int(float(row["dropped_entities"]))
            d["dropped_relations"] += int(float(row["dropped_relations"]))
    return summary


def main():
    required_raw = ["train.json", "dev.json", "test.json", "avimaint_types.json", "avimaint_spert.conf"]
    for name in required_raw:
        if not (RAW / name).exists():
            raise SystemExit(f"Missing authoritative raw SpERT artifact: {RAW / name}")

    splits_path = IE / "outputs" / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"Frozen split file missing: {splits_path}")
    frozen = load_json(splits_path)
    for split, n in EXPECTED_COUNTS.items():
        ids = [str(x) for x in frozen.get(split, [])]
        if len(ids) != n:
            raise SystemExit(f"Frozen split {split}: expected {n}, found {len(ids)}")
    all_ids = [str(x) for k in ("train", "dev", "test") for x in frozen[k]]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Frozen train/dev/test IDs overlap. Refusing to prepare ablation.")

    raw_docs = {split: load_json(RAW / f"{split}.json") for split in EXPECTED_COUNTS}
    raw_order: dict[str, list[str]] = {}
    for split, docs in raw_docs.items():
        if len(docs) != EXPECTED_COUNTS[split]:
            raise SystemExit(f"Raw SpERT {split}.json has {len(docs)} docs, expected {EXPECTED_COUNTS[split]}")
        ids = [doc_id(d) for d in docs]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Duplicate IDs in raw SpERT {split}.json")
        if set(ids) != set(str(x) for x in frozen[split]):
            raise SystemExit(f"Raw SpERT {split}.json membership differs from outputs/splits.json")
        raw_order[split] = ids

    types = load_json(RAW / "avimaint_types.json")
    n_ent = len(types.get("entities", {}))
    n_rel = len(types.get("relations", {}))
    if (n_ent, n_rel) != (9, 11):
        raise SystemExit(f"Expected FULL schema 9 entities / 11 relations, found {n_ent}/{n_rel}")

    raw_conf = (RAW / "avimaint_spert.conf").read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    model_dirs = {}
    supports = {}

    for system in SYSTEMS:
        mapping = load_variant_map(system)
        missing = [rid for rid in all_ids if rid not in mapping]
        if missing:
            raise SystemExit(f"{system}: missing {len(missing)} frozen IDs; examples={missing[:10]}")

        target = OUT / system
        target.mkdir(parents=True, exist_ok=True)
        for split, ids in raw_order.items():
            docs = [projected_doc(mapping[rid], rid) for rid in ids]
            dump_json(target / f"{split}.json", docs)
            supports.setdefault(system, {})[split] = {
                "records": len(docs),
                "entities": sum(len(d.get("entities", [])) for d in docs),
                "relations": sum(len(d.get("relations", [])) for d in docs),
            }
        shutil.copy2(RAW / "avimaint_types.json", target / "avimaint_types.json")
        rewrite_config(raw_conf, target, system)
        model_dirs[system] = str(target.relative_to(ROOT))

    projection = split_projection_summary(raw_order)
    test_ref = raw_docs["test"]
    raw_test_support = {
        "entities": sum(len(d.get("entities", [])) for d in test_ref),
        "relations": sum(len(d.get("relations", [])) for d in test_ref),
    }
    for system in SYSTEMS:
        s = supports[system]["test"]
        if s["entities"] != raw_test_support["entities"] or s["relations"] != raw_test_support["relations"]:
            raise SystemExit(
                f"{system}: frozen TEST support changed ({s['entities']} entities/{s['relations']} relations) "
                f"vs raw ({raw_test_support['entities']}/{raw_test_support['relations']})."
            )

    manifest = {
        "experiment": "representation_matched_normalization_spert_full_9x11",
        "status": "prepared_no_training_yet",
        "raw_model_policy": "reuse already-frozen authoritative raw SpERT; do not retrain raw",
        "normalized_systems_to_train": SYSTEMS,
        "split_counts": EXPECTED_COUNTS,
        "split_membership_source": "legacy_import/maintenance-ie/outputs/splits.json",
        "ordering_source": "authoritative raw outputs/spert/{train,dev,test}.json orig_id order",
        "frozen_id_sha256": {
            split: sha256_text("\n".join(raw_order[split])) for split in raw_order
        },
        "schema": {"entities": n_ent, "relations": n_rel},
        "raw_test_support": raw_test_support,
        "variant_supports": supports,
        "projection_by_split": projection,
        "output_dirs": model_dirs,
        "protocol": {
            "matched_representation": "each normalized SpERT is trained/dev-evaluated/tested on the same normalization representation",
            "same_architecture_and_hyperparameters": True,
            "hyperparameter_source": "authoritative raw outputs/spert/avimaint_spert.conf; only label and filesystem paths rewritten",
            "test_policy": "train all four fixed conditions first; calculate comparative TEST metrics only after all four prediction artifacts exist",
            "retuning_after_test": "forbidden",
        },
    }
    dump_json(OUT / "PREP_MANIFEST.json", manifest)

    REPORT_OLD.mkdir(parents=True, exist_ok=True)
    (REPORT_OLD / "INTERPRETATION_NOTE.txt").write_text(
        "The earlier five-way frozen-RAW-model experiment is an inference-time representation-shift sensitivity test.\n"
        "It must NOT be used as the main answer to whether normalization improves a representation-matched IE pipeline.\n"
        "The main RQ1 downstream SpERT experiment is outputs/reports/normalization_spert_matched/.\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("MATCHED NORMALIZATION -> SpERT EXPORTS PREPARED")
    print("=" * 76)
    print("Frozen membership : train=1275 dev=100 test=225")
    print("Schema            : 9 entities / 11 relations")
    print(f"Raw TEST support  : entities={raw_test_support['entities']} relations={raw_test_support['relations']}")
    for system in SYSTEMS:
        p = projection[system]
        print(
            f"{system:18s}: train dE={p['train']['dropped_entities']} dR={p['train']['dropped_relations']} | "
            f"dev dE={p['dev']['dropped_entities']} dR={p['dev']['dropped_relations']} | "
            f"test dE={p['test']['dropped_entities']} dR={p['test']['dropped_relations']}"
        )
    print(f"Prepared -> {OUT}")
    print("RAW model is reused; four normalized SpERT models remain to train.")


if __name__ == "__main__":
    main()
