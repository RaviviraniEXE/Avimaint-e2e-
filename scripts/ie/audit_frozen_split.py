"""Audit the frozen aviation split for leakage, provenance, and label support."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import/maintenance-ie"


def main() -> None:
    split_path = IE / "outputs/splits.json"
    if not split_path.is_file():
        raise SystemExit("Frozen split is missing. Run scripts\\ie\\02_freeze_split.bat first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((IE / "outputs/gold").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[str(row["ident"])] = row

    all_ids = [ident for name in ("train", "dev", "test") for ident in split[name]]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Identifier leakage: a record appears in more than one split")
    if set(all_ids) != set(records):
        raise SystemExit("Frozen split does not cover the complete gold corpus")

    groups: dict[str, set[str]] = {}
    report: dict[str, Any] = {"seed": split.get("seed"), "splits": {}}
    for name in ("train", "dev", "test"):
        rows = [records[str(ident)] for ident in split[name]]
        groups[name] = {str(row["exact_group_id"]) for row in rows}
        report["splits"][name] = {
            "records": len(rows),
            "batches": dict(sorted(Counter(row["annotation_batch"] for row in rows).items())),
            "sampling_populations": dict(
                sorted(Counter(row["sampling_population"] for row in rows).items())
            ),
            "entities": dict(
                sorted(Counter(e["type"] for row in rows for e in row["entities"]).items())
            ),
            "relations": dict(
                sorted(Counter(r["type"] for row in rows for r in row["relations"]).items())
            ),
        }

    overlap = {
        "train_dev": len(groups["train"] & groups["dev"]),
        "train_test": len(groups["train"] & groups["test"]),
        "dev_test": len(groups["dev"] & groups["test"]),
    }
    report["exact_group_overlap"] = overlap
    report["rare_records_in_dev_or_test"] = sum(
        count
        for name in ("dev", "test")
        for population, count in report["splits"][name]["sampling_populations"].items()
        if population == "rare_enriched"
    )
    report["valid"] = not any(overlap.values()) and report["rare_records_in_dev_or_test"] == 0
    report["interpretation_warning"] = (
        "REFERENCE and ACTION_FOLLOWS_REFERENCE have very low unbiased test support. "
        "Report their support and uncertainty; do not make a standalone performance claim."
    )
    if not report["valid"]:
        raise SystemExit("Frozen split failed leakage/provenance validation")

    target = ROOT / "outputs/reports/annotation_audit/frozen_split_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Frozen split valid: "
        f"train={report['splits']['train']['records']}, "
        f"dev={report['splits']['dev']['records']}, "
        f"test={report['splits']['test']['records']}"
    )
    print("Exact-group overlap=0; rare-enriched records in dev/test=0")
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
