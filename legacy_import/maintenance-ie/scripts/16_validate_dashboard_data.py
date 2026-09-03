"""Validate the case library used by the AviMaint-DSS v2 dashboard."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.core.data import load_cases
from dashboard.core.schema import SchemaCatalog
from dashboard.core.validator import validate_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="aviation")
    parser.add_argument(
        "--cases",
        help="Case file or directory; defaults to outputs/dashboard/<name>",
    )
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any blocking record is present",
    )
    args = parser.parse_args()

    cases_path = (
        Path(args.cases).expanduser()
        if args.cases
        else ROOT / "outputs" / "dashboard" / args.name
    )
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path
    cases = load_cases(cases_path)
    schema = SchemaCatalog.from_yaml(args.schema)
    report = validate_cases(cases, schema)
    payload = report.as_dict()

    print(f"Records: {report.total_cases}")
    print(f"Recommendation-ready: {report.recommendation_ready_cases}")
    print(f"Excluded: {report.total_cases - report.recommendation_ready_cases}")
    print(f"Blocking findings: {report.blocking_count}")
    print(f"Warnings: {report.warning_count}")
    print("Finding counts:")
    for code, count in sorted(report.issue_counts().items()):
        print(f"  {code}: {count}")

    output = (
        Path(args.report).expanduser()
        if args.report
        else (
            cases_path
            if cases_path.is_dir()
            else cases_path.parent
        )
        / "validation.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Report: {output.resolve()}")

    if report.recommendation_ready_cases == 0:
        print(
            "FAILED: the dashboard has no recommendation-ready case.",
            file=sys.stderr,
        )
        return 2
    if args.strict and report.blocking_count:
        print(
            "FAILED STRICT CHECK: blocking records remain.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

