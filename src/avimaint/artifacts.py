"""Artifact and run-manifest helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from avimaint.contracts import MaintenanceRecord


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, records: Iterable[MaintenanceRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def base_run_manifest(
    experiment_id: str,
    run_id: str,
    schema_id: str,
    dataset_id: str,
    seed: int,
    project_root: str | Path,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "schema_id": schema_id,
        "seed": seed,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(project_root),
        "test_set_used_for_tuning": False,
    }

