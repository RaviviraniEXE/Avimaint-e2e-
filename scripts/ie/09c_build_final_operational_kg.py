"""Rebuild final lightweight KG from the verified operational extraction."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
KGROOT = IE / "outputs" / "kg"
PRED = KGROOT / "predictions_full.json"
INDEX = KGROOT / "full_index.jsonl"
EXTRACT = KGROOT / "FINAL_FULL_CORPUS_SPERT_MANIFEST.json"
AVIATION = KGROOT / "aviation"
ARCHIVE = KGROOT / "_archive"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for path in (PRED, INDEX, EXTRACT):
        if not path.exists():
            raise SystemExit(f"Missing final extraction artifact: {path}")

    extraction = json.loads(
        EXTRACT.read_text(encoding="utf-8-sig")
    )
    if (
        extraction.get("status") != "complete"
        or extraction.get("representation") != "selective_byt5"
        or int(extraction.get("records", 0)) != 6169
    ):
        raise SystemExit(
            "Extraction is not complete Selective-ByT5 / 6169."
        )

    expected_hash = (
        extraction.get("artifacts", {}).get("predictions_sha256")
    )
    actual_hash = sha256(PRED)
    if expected_hash != actual_hash:
        raise SystemExit(
            "Prediction SHA-256 differs from extraction manifest."
        )

    archived_to = None
    if AVIATION.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = ARCHIVE / (
            f"aviation_before_final_selective_byt5_{stamp}"
        )
        shutil.move(str(AVIATION), str(archived))
        archived_to = str(archived)
        print("[ARCHIVE]", archived)

    cmd = [
        sys.executable,
        str(IE / "scripts" / "13_build_kg.py"),
        "--pred", str(PRED),
        "--tokens", str(INDEX),
        "--name", "aviation",
    ]
    print("[BUILD]", " ".join(cmd))
    subprocess.run(cmd, cwd=IE, check=True)

    summary = AVIATION / "summary.txt"
    manifest = AVIATION / "manifest.json"
    if not summary.exists() or not manifest.exists():
        raise SystemExit("KG summary/manifest was not created.")

    text = summary.read_text(encoding="utf-8", errors="replace")
    rec = re.search(r"\brecords=(\d+)", text)
    inv_e = re.search(r"invalid entities skipped=(\d+)", text)
    inv_r = re.search(r"invalid relations skipped=(\d+)", text)

    if not rec or int(rec.group(1)) != 6169:
        raise SystemExit(
            "KG summary does not report records=6169."
        )
    if not inv_e or int(inv_e.group(1)) != 0:
        raise SystemExit(
            "KG reports skipped invalid entities; investigate."
        )
    if not inv_r or int(inv_r.group(1)) != 0:
        raise SystemExit(
            "KG reports skipped invalid relations; investigate."
        )

    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "records": 6169,
        "representation": "selective_byt5",
        "normalization_system": "Selective ByT5",
        "raw_source_retained_in_extraction_index": True,
        "prediction_sha256": actual_hash,
        "kg_summary_sha256": sha256(summary),
        "kg_manifest_sha256": sha256(manifest),
        "invalid_entities_skipped": 0,
        "invalid_relations_skipped": 0,
        "previous_aviation_kg_archived_to": archived_to,
    }
    out = KGROOT / "FINAL_KG_VERIFICATION.json"
    out.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("FINAL OPERATIONAL AVIATION KG VERIFIED")
    print("=" * 88)
    print("representation            : selective_byt5")
    print("records                   : 6169")
    print("invalid entities skipped  : 0")
    print("invalid relations skipped : 0")
    print("raw source retained       : True")
    print("verification              :", out)
    print("=" * 88)


if __name__ == "__main__":
    main()
