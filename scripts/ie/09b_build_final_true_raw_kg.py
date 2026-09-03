"""Build and verify the final aviation KG from the final 6,169 true-raw extraction.

This wrapper:
- refuses any extraction manifest that is not complete/raw/6169;
- archives an existing outputs/kg/aviation directory rather than silently
  overwriting it;
- invokes the existing lightweight KG builder;
- checks final KG record count, invalid counts, and prediction provenance;
- writes FINAL_KG_VERIFICATION.json.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
KG = IE / "outputs" / "kg"
PRED = KG / "predictions_full.json"
INDEX = KG / "full_index.jsonl"
EXTRACT = KG / "FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
AVIATION = KG / "aviation"
ARCHIVE = KG / "_archive"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    for path in (PRED, INDEX, EXTRACT):
        if not path.exists():
            raise SystemExit(f"Missing required final extraction artifact: {path}")

    extraction = load_json(EXTRACT)
    if extraction.get("status") != "complete":
        raise SystemExit("Final extraction manifest status is not complete.")
    if extraction.get("representation") != "raw":
        raise SystemExit("Final extraction representation is not raw.")
    if int(extraction.get("records", 0)) != 6169:
        raise SystemExit("Final extraction does not contain 6169 records.")

    expected_pred_hash = (
        extraction.get("artifacts", {}).get("predictions_sha256")
    )
    actual_pred_hash = sha256(PRED)
    if expected_pred_hash != actual_pred_hash:
        raise SystemExit(
            "predictions_full.json hash differs from extraction manifest."
        )

    archived_to = None
    if AVIATION.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = ARCHIVE / f"aviation_before_final_true_raw_{stamp}"
        shutil.move(str(AVIATION), str(archived))
        archived_to = str(archived)
        print(f"[ARCHIVE] Previous aviation KG -> {archived}")

    cmd = [
        sys.executable,
        str(IE / "scripts" / "13_build_kg.py"),
        "--pred", str(PRED),
        "--tokens", str(INDEX),
        "--name", "aviation",
    ]
    print("[BUILD]", " ".join(cmd))
    subprocess.run(cmd, cwd=IE, check=True)

    manifest_path = AVIATION / "manifest.json"
    summary_path = AVIATION / "summary.txt"
    if not manifest_path.exists() or not summary_path.exists():
        raise SystemExit("Final KG builder did not create manifest/summary.")

    kg = load_json(manifest_path)
    counts = kg.get("counts", {})
    if int(counts.get("records", 0)) != 6169:
        raise SystemExit(
            f"Final KG records={counts.get('records')}, expected=6169"
        )
    if int(counts.get("invalid_entities_skipped", -1)) != 0:
        raise SystemExit(
            "Final KG skipped invalid entities; investigate before freezing."
        )
    if int(counts.get("invalid_relations_skipped", -1)) != 0:
        raise SystemExit(
            "Final KG skipped invalid relations; investigate before freezing."
        )

    kg_pred_hash = kg.get("input", {}).get("prediction_sha256")
    if kg_pred_hash != actual_pred_hash:
        raise SystemExit(
            "Final KG manifest prediction hash does not match final extraction."
        )

    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "records": 6169,
        "representation": "raw",
        "prediction_sha256": actual_pred_hash,
        "kg_manifest_sha256": sha256(manifest_path),
        "kg_summary_sha256": sha256(summary_path),
        "invalid_entities_skipped": 0,
        "invalid_relations_skipped": 0,
        "previous_aviation_kg_archived_to": archived_to,
        "checks": {
            "extraction_complete": True,
            "representation_raw": True,
            "records_6169": True,
            "prediction_hash_matches_extraction": True,
            "prediction_hash_matches_kg": True,
            "invalid_entities_zero": True,
            "invalid_relations_zero": True,
        },
    }
    out = KG / "FINAL_KG_VERIFICATION.json"
    out.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 84)
    print("FINAL TRUE-RAW AVIATION KG VERIFIED")
    print("=" * 84)
    print("records                  : 6169")
    print(f"nodes                    : {counts.get('nodes')}")
    print(f"aggregated edges         : {counts.get('edges')}")
    print(f"entity mentions          : {counts.get('entity_mentions')}")
    print(f"relation record-support  : {counts.get('relation_record_support')}")
    print("invalid entities skipped : 0")
    print("invalid relations skipped: 0")
    print("verification             :", out)
    print("=" * 84)


if __name__ == "__main__":
    main()
