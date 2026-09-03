"""Linear-chain CRF named-entity baseline with calibrated uncertainty support."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from src.evaluate import entity_scores
from src.features import sent_features
from src.progress import LiveProgress, trace_event


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert BIO tags to half-open entity spans, repairing invalid I-tags."""
    entities: list[dict] = []
    start: int | None = None
    current: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        prefix, _, label = tag.partition("-")
        begins = prefix == "B" or (prefix == "I" and label != current)
        ends = current is not None and (prefix == "O" or begins or label != current)
        if ends:
            entities.append({"type": current, "start": start, "end": index})
            start, current = None, None
        if begins:
            start, current = index, label
    return entities


class CRFTagger:
    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 200):
        import sklearn_crfsuite

        self.params = {"c1": float(c1), "c2": float(c2), "max_iterations": max_iterations}
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=float(c1), c2=float(c2),
            max_iterations=max_iterations, all_possible_transitions=True,
        )

    def _fit_xy(self, x: list[list[dict]], y: list[list[str]]) -> "CRFTagger":
        """Fit already-computed features. Used by tuning to avoid rebuilding the
        identical feature matrix for every c1/c2 configuration."""
        self.model.fit(x, y)
        return self

    def fit(self, records: list[dict]) -> "CRFTagger":
        if not records:
            raise ValueError("CRF training requires at least one record")
        x = [sent_features(record["tokens"]) for record in records]
        y = [record["bio"] for record in records]
        trace_event("crf_fit_data", records=len(records), sequences=len(x),
                    tokens=sum(len(row["tokens"]) for row in records), **self.params)
        return self._fit_xy(x, y)

    def predict(self, records: Iterable[dict]) -> list[list[str]]:
        return self.model.predict([sent_features(record["tokens"]) for record in records])

    def predict_bio(self, tokens: list[str]) -> list[str]:
        return self.model.predict_single(sent_features(tokens))

    def uncertainty(self, tokens: list[str]) -> float:
        marginals = self.model.predict_marginals_single(sent_features(tokens))
        if not marginals:
            return 1.0
        confidence = sum(max(distribution.values()) for distribution in marginals) / len(marginals)
        return float(1.0 - confidence)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "params": self.params}, target)

    @classmethod
    def load(cls, path: str) -> "CRFTagger":
        payload = joblib.load(path)
        instance = cls(**payload.get("params", {}))
        instance.model = payload["model"]
        return instance

    @classmethod
    def tuned(
        cls,
        train: list[dict],
        dev: list[dict],
        grid: list[tuple[float, float]] | None = None,
    ) -> tuple["CRFTagger", dict]:
        grid = grid or [(0.01, 0.1), (0.1, 0.1), (0.1, 0.5), (0.5, 0.1)]
        if not train or not dev:
            raise ValueError("CRF tuning requires non-empty train and development sets")

        # Feature extraction is deterministic and independent of c1/c2. Compute it
        # once instead of repeating it for every grid point (20x in the thesis grid).
        train_x = [sent_features(row["tokens"]) for row in train]
        train_y = [row["bio"] for row in train]
        dev_x = [sent_features(row["tokens"]) for row in dev]
        gold = [
            {"tokens": row["tokens"], "entities": bio_to_entities(row["tokens"], row["bio"]), "relations": []}
            for row in dev
        ]
        trace_event(
            "crf_tuning_setup",
            train_records=len(train),
            dev_records=len(dev),
            train_tokens=sum(len(row["tokens"]) for row in train),
            configurations=len(grid),
        )

        best_model: CRFTagger | None = None
        best = {"micro_f1": -1.0}
        history: list[dict] = []
        progress = LiveProgress("CRF NER DEV tuning", len(grid))
        for index, (c1, c2) in enumerate(grid, start=1):
            detail = f"config {index}/{len(grid)} c1={c1:g} c2={c2:g}"
            progress.begin(detail)
            try:
                candidate = cls(c1=c1, c2=c2)._fit_xy(train_x, train_y)
                tags_all = candidate.model.predict(dev_x)
                pred = [
                    {"tokens": row["tokens"], "entities": bio_to_entities(row["tokens"], tags), "relations": []}
                    for row, tags in zip(dev, tags_all)
                ]
                score = entity_scores(gold, pred)["micro_f1"]
                is_best = score > best["micro_f1"]
                history.append({"c1": float(c1), "c2": float(c2), "micro_f1": float(score)})
                if is_best:
                    best_model = candidate
                    best = {"c1": c1, "c2": c2, "micro_f1": score}
                progress.finish(metric=f"DEV entity micro-F1={score:.4f}", is_best=is_best)
            except Exception as exc:
                progress.fail(exc)
                raise
        progress.close(
            f"selected c1={best['c1']:g}, c2={best['c2']:g}, DEV F1={best['micro_f1']:.4f}"
        )
        best["history"] = history
        return best_model, best
