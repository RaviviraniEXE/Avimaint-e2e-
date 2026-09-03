"""RQ5 historical-agreement calibration for live DSS queries.

This module deliberately re-fits the already-defined RQ5 calibrator from the
frozen DEV predictions every time the cached dashboard engine is built.

Inputs are exactly the three features used by final_rq5_planning_support.py:
    top_score, margin, support_clusters

The probability means:
    P(predicted action family agrees with the recorded historical action family)

It does NOT estimate technical correctness, safety, airworthiness, regulatory
compliance, or aircraft-specific applicability.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["top_score", "margin", "support_clusters"]


class RQ5AgreementCalibrator:
    def __init__(self, dev_predictions_path: str | Path | None):
        self.path = Path(dev_predictions_path).resolve() if dev_predictions_path else None
        self.model = None
        self.dev_rows = 0
        self.error = ""
        self._fit()

    def _fit(self) -> None:
        if self.path is None or not self.path.is_file():
            self.error = f"DEV calibration predictions not found: {self.path}"
            return
        try:
            frame = pd.read_csv(self.path)
            missing = [c for c in FEATURES + ["top1_correct"] if c not in frame.columns]
            if missing:
                raise ValueError(f"missing calibration columns: {missing}")
            X = frame[FEATURES].astype(float).to_numpy()
            y = frame["top1_correct"].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                raise ValueError("DEV correctness has one class; calibration is not identifiable")
            self.model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=42),
            ).fit(X, y)
            self.dev_rows = int(len(frame))
        except Exception as exc:
            self.model = None
            self.error = f"{type(exc).__name__}: {exc}"

    def available(self) -> bool:
        return self.model is not None

    def predict(self, top_score: float, margin: float, support_clusters: int) -> float | None:
        if self.model is None:
            return None
        X = np.asarray([[float(top_score), float(margin), float(support_clusters)]], dtype=float)
        return float(self.model.predict_proba(X)[0, 1])

    def status(self) -> str:
        if self.available():
            return f"DEV-only RQ5 calibrator · n={self.dev_rows}"
        return self.error or "RQ5 calibrator unavailable"
