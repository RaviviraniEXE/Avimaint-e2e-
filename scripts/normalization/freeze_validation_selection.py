"""Freeze the normalization choice before any final-test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = ROOT / "outputs/normalization/predictions/byt5_gold_v1"
OUTPUT = ROOT / "outputs/normalization/selection/normalization_selection_manifest.json"
SYSTEMS = (
    "raw",
    "most_frequent_replacement",
    "rules",
    "byt5",
    "selective_byt5",
    "rules_then_byt5",
)
SELECTED = "selective_byt5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_manifest() -> None:
    if not OUTPUT.exists():
        raise SystemExit("Frozen validation selection manifest is missing.")
    manifest = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if (
        manifest.get("selected_system") != SELECTED
        or manifest.get("test_results_consulted") is not False
    ):
        raise SystemExit("Frozen selection manifest has an unexpected selection contract.")
    for system, artifact in manifest.get("artifacts", {}).items():
        for kind in ("prediction", "metrics"):
            path = ROOT / artifact[f"{kind}_path"]
            if not path.exists() or sha256(path) != artifact[f"{kind}_sha256"]:
                raise SystemExit(
                    f"Validation artifact changed after selection freeze: {system} {kind}"
                )
    print("Frozen validation selection and artifact hashes verified.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_frozen_manifest()
        return
    metrics: dict[str, dict[str, float]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for system in SYSTEMS:
        metrics_path = PREDICTION_DIR / f"validation_{system}_metrics.json"
        prediction_path = PREDICTION_DIR / f"validation_{system}.csv"
        if not metrics_path.exists() or not prediction_path.exists():
            raise SystemExit(
                f"Missing validation artifact for {system}. Run the validation launchers first."
            )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("split") != "validation" or payload.get("system") != system:
            raise SystemExit(f"Unexpected metric identity in {metrics_path}")
        metrics[system] = payload["metrics"]
        artifacts[system] = {
            "prediction_path": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": sha256(prediction_path),
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256(metrics_path),
        }

    selected_metrics = metrics[SELECTED]
    if selected_metrics["protected_token_accuracy"] != 1.0:
        raise SystemExit(
            "Selected system is not fully grounded after fallback; selection not frozen."
        )
    if selected_metrics["wer"] > metrics["byt5"]["wer"]:
        raise SystemExit(
            "Selective fallback degraded WER relative to ByT5; review before freezing."
        )

    selected_predictions = pd.read_csv(
        PREDICTION_DIR / f"validation_{SELECTED}.csv", dtype=str
    ).fillna("")
    if len(selected_predictions) != 926:
        raise SystemExit(
            f"Expected 926 validation predictions; found {len(selected_predictions)}"
        )
    fallback_counts = {
        str(key): int(value)
        for key, value in selected_predictions["fallback"].value_counts().to_dict().items()
    }
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_population": "validation_only",
        "test_results_consulted": False,
        "selected_system": SELECTED,
        "deployment_policy": "ByT5; fallback to rules, then raw, on grounding failure",
        "selection_reason": (
            "Best validation text-agreement metrics with deterministic fallback for "
            "unsupported identifiers or numeric additions."
        ),
        "validation_records": 926,
        "fallback_counts": fallback_counts,
        "validation_metrics": metrics,
        "artifacts": artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
