"""Freeze final Selective-ByT5 matched-SpERT extraction and KG."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
KGROOT = IE / "outputs" / "kg"
REPORT = IE / "outputs" / "reports" / "normalization_spert_matched_v2"
EXPORT = IE / "outputs" / "spert_normalized" / "selective_byt5"
FREEZE = ROOT / "outputs" / "frozen" / "final_operational_ie_kg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_required(source: Path, target: Path):
    if not source.exists():
        raise SystemExit(f"Required freeze artifact missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    extract_path = KGROOT / "FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
    kgcheck_path = KGROOT / "FINAL_KG_VERIFICATION.json"

    extract = json.loads(
        extract_path.read_text(encoding="utf-8-sig")
    ) if extract_path.exists() else {}
    kgcheck = json.loads(
        kgcheck_path.read_text(encoding="utf-8-sig")
    ) if kgcheck_path.exists() else {}

    if (
        extract.get("status") != "complete"
        or extract.get("representation") != "selective_byt5"
        or int(extract.get("records", 0)) != 6169
    ):
        raise SystemExit(
            "Final Selective-ByT5 extraction is not ready to freeze."
        )
    if kgcheck.get("status") != "pass":
        raise SystemExit("Final KG verification did not pass.")

    if FREEZE.exists():
        archive = FREEZE.parent / (
            "final_operational_ie_kg_previous_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.move(str(FREEZE), str(archive))
        print("[ARCHIVE] Existing final freeze ->", archive)

    for filename in (
        "full_corpus_spert.json",
        "full_index.jsonl",
        "full_corpus_manifest.json",
        "predictions_full.json",
        "FINAL_FULL_CORPUS_SPERT_MANIFEST.json",
        "FINAL_KG_VERIFICATION.json",
    ):
        copy_required(
            KGROOT / filename,
            FREEZE / "extraction" / filename,
        )

    shutil.copytree(
        KGROOT / "aviation",
        FREEZE / "kg" / "aviation",
    )

    for filename in (
        "matched_normalization_spert_ablation_v2.csv",
        "MODEL_REGISTRY_V2.json",
        "FINAL_MATCHED_NORMALIZATION_SPERT_MANIFEST_V2.json",
        "REPRESENTATION_AUDIT.json",
    ):
        copy_required(
            REPORT / filename,
            FREEZE / "rq1_provenance" / filename,
        )

    for filename in ("avimaint_spert.conf", "avimaint_types.json"):
        copy_required(
            EXPORT / filename,
            FREEZE / "model_provenance" / filename,
        )

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operational_representation": "selective_byt5",
        "operational_model_system": "selective_byt5",
        "raw_source_preserved": True,
        "selection_type": "engineering trade-off after controlled evaluation",
        "no_post_test_model_tuning": True,
        "reason": (
            "Selective ByT5 retains a normalized, human-readable representation "
            "and safety fallback while its matched SpERT performance remains "
            "close to raw on primary micro metrics and provides the strongest "
            "relation macro-F1 among the five evaluated conditions. The raw "
            "representation remains the controlled RQ1 micro-metric baseline "
            "and is preserved as source provenance."
        ),
        "corrected_matched_metrics": {
            "raw": {
                "entity_micro_f1": 0.9455,
                "relation_micro_f1": 0.8203,
                "relation_macro_f1": 0.7511,
            },
            "selective_byt5": {
                "entity_micro_f1": 0.9424,
                "relation_micro_f1": 0.8167,
                "relation_macro_f1": 0.7781,
            },
        },
    }
    (FREEZE / "OPERATIONAL_SELECTION_MANIFEST.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    freeze_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "records": 6169,
        "representation": "selective_byt5",
        "raw_source_retained": True,
        "model_weights_duplicated": False,
        "model_identity_source": (
            "extraction/FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
        ),
    }
    (FREEZE / "FREEZE_MANIFEST.json").write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for path in sorted(FREEZE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(
                f"{sha256(path)}  {path.relative_to(FREEZE).as_posix()}"
            )
    (FREEZE / "SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n", encoding="ascii"
    )

    bad = []
    for line in (FREEZE / "SHA256SUMS.txt").read_text(
        encoding="ascii"
    ).splitlines():
        expected, rel = line.split("  ", 1)
        path = FREEZE / rel
        if not path.exists():
            bad.append("MISSING " + rel)
        elif sha256(path) != expected:
            bad.append("MISMATCH " + rel)

    if bad:
        raise SystemExit(
            "Freeze checksum failure:\n" + "\n".join(bad)
        )

    print("=" * 88)
    print("FINAL OPERATIONAL IE + KG FREEZE COMPLETE")
    print("=" * 88)
    print("representation : selective_byt5")
    print("records        : 6169")
    print("raw provenance : retained")
    print("model weights  : not duplicated")
    print("checksums      : ALL VERIFIED")
    print("freeze         :", FREEZE)
    print("=" * 88)


if __name__ == "__main__":
    main()
