# Normalization outputs

Generated artifacts are intentionally excluded from the scaffold archive.

- `models/`: new checkpoints trained from scratch.
- `predictions/`: immutable per-example predictions for every condition.
- `metrics/`: aggregate metrics, confidence intervals and error analysis.
- `manifests/`: data/config/code hashes and runtime versions.

Do not copy model weights from the legacy project into this directory.
