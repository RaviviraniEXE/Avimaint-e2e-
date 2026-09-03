"""Freeze the final 6,169-record true-raw IE extraction + lightweight KG.

Large model weights are NOT duplicated. Their exact path/hash is already in
FINAL_FULL_CORPUS_SPERT_MANIFEST.json and is copied into this freeze.

The freeze contains:
- final full-corpus input/index/predictions + manifests;
- corrected V2 model registry and RQ1 result table;
- final lightweight aviation KG and verification;
- raw SpERT config/types;
- SHA256SUMS.txt.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
KG = IE / "outputs" / "kg"
REPORT = IE / "outputs" / "reports" / "normalization_spert_matched_v2"
RAW_EXPORT = IE / "outputs" / "spert_normalized" / "raw"
FREEZE = ROOT / "outputs" / "frozen" / "final_true_raw_ie_kg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(source: Path, target: Path):
    if not source.exists():
        raise SystemExit(f"Required freeze artifact missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    extraction_manifest = KG / "FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
    kg_verification = KG / "FINAL_KG_VERIFICATION.json"
    kg_manifest = KG / "aviation" / "manifest.json"

    for path in (extraction_manifest, kg_verification, kg_manifest):
        if not path.exists():
            raise SystemExit(
                f"Final extraction/KG is not ready to freeze: missing {path}"
            )

    extraction = json.loads(
        extraction_manifest.read_text(encoding="utf-8-sig")
    )
    kgcheck = json.loads(
        kg_verification.read_text(encoding="utf-8-sig")
    )
    if (
        extraction.get("status") != "complete"
        or extraction.get("representation") != "raw"
        or int(extraction.get("records", 0)) != 6169
    ):
        raise SystemExit("Final extraction manifest is not complete/raw/6169.")
    if kgcheck.get("status") != "pass":
        raise SystemExit("Final KG verification did not pass.")

    if FREEZE.exists():
        archive = FREEZE.parent / (
            "final_true_raw_ie_kg_previous_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.move(str(FREEZE), str(archive))
        print(f"[ARCHIVE] Existing freeze -> {archive}")

    # IE extraction evidence
    for filename in (
        "full_corpus_spert.json",
        "full_index.jsonl",
        "full_corpus_manifest.json",
        "predictions_full.json",
        "FINAL_FULL_CORPUS_SPERT_MANIFEST.json",
        "FINAL_KG_VERIFICATION.json",
    ):
        copy_file(
            KG / filename,
            FREEZE / "extraction" / filename,
        )

    # Entire lightweight final KG
    shutil.copytree(
        KG / "aviation",
        FREEZE / "kg" / "aviation",
    )

    # Corrected RQ1 provenance needed to identify the exact raw model
    for filename in (
        "matched_normalization_spert_ablation_v2.csv",
        "MODEL_REGISTRY_V2.json",
        "FINAL_MATCHED_NORMALIZATION_SPERT_MANIFEST_V2.json",
        "REPRESENTATION_AUDIT.json",
    ):
        copy_file(
            REPORT / filename,
            FREEZE / "rq1_provenance" / filename,
        )

    for filename in (
        "avimaint_spert.conf",
        "avimaint_types.json",
    ):
        copy_file(
            RAW_EXPORT / filename,
            FREEZE / "model_provenance" / filename,
        )

    freeze_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "role": (
            "final true-System-A-raw full-corpus SpERT silver extraction "
            "and lightweight aviation KG"
        ),
        "records": 6169,
        "model_weights_duplicated": False,
        "model_identity_source": (
            "extraction/FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
        ),
        "rq1_model_registry": (
            "rq1_provenance/MODEL_REGISTRY_V2.json"
        ),
    }
    (FREEZE / "FREEZE_MANIFEST.json").write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for path in sorted(FREEZE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rel = path.relative_to(FREEZE)
            rows.append(f"{sha256(path)}  {rel.as_posix()}")

    (FREEZE / "SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n",
        encoding="ascii",
    )

    # Immediate self-verification
    bad = []
    for line in (FREEZE / "SHA256SUMS.txt").read_text(
        encoding="ascii"
    ).splitlines():
        expected, rel = line.split("  ", 1)
        path = FREEZE / Path(rel)
        if not path.exists():
            bad.append(f"MISSING {rel}")
        elif sha256(path) != expected:
            bad.append(f"MISMATCH {rel}")

    if bad:
        raise SystemExit(
            "Freeze checksum verification failed:\n" + "\n".join(bad)
        )

    print("=" * 84)
    print("FINAL TRUE-RAW IE + KG FREEZE COMPLETE")
    print("=" * 84)
    print("freeze :", FREEZE)
    print("records: 6169")
    print("model weights duplicated: False")
    print("checksums: ALL VERIFIED")
    print("=" * 84)


if __name__ == "__main__":
    main()
