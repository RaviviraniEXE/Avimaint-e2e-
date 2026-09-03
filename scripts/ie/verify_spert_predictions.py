from __future__ import annotations
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--predictions", required=True)
a = p.parse_args()

dataset_path = Path(a.dataset)
pred_path = Path(a.predictions)
if not dataset_path.exists():
    raise SystemExit(f"Dataset not found: {dataset_path}")
if not pred_path.exists():
    raise SystemExit(f"Prediction file was NOT created: {pred_path}")
try:
    gold = json.loads(dataset_path.read_text(encoding="utf-8"))
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
except Exception as e:
    raise SystemExit(f"Could not parse dataset/predictions JSON: {e}")
if not isinstance(gold, list) or not isinstance(pred, list):
    raise SystemExit("SpERT dataset and predictions must both be JSON lists.")
if len(pred) != len(gold):
    raise SystemExit(f"Prediction count mismatch: predictions={len(pred)}, expected={len(gold)}")
if len(pred) == 0:
    raise SystemExit("Prediction file is empty.")
print(f"SPERT PREDICTION ARTIFACT VERIFIED: {len(pred)}/{len(gold)} records")
