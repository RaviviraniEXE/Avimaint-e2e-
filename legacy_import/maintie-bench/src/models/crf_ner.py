"""Linear-chain CRF named-entity baseline with calibrated uncertainty support."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from src.evaluate import entity_scores
from src.features import sent_features


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

    def fit(self, records: list[dict]) -> "CRFTagger":
        if not records:
            raise ValueError("CRF training requires at least one record")
        x = [sent_features(record["tokens"]) for record in records]
        y = [record["bio"] for record in records]
        self.model.fit(x, y)
        return self

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
        best_model: CRFTagger | None = None
        best = {"micro_f1": -1.0}
        gold = [
            {"tokens": row["tokens"], "entities": bio_to_entities(row["tokens"], row["bio"]), "relations": []}
            for row in dev
        ]
        for c1, c2 in grid:
            candidate = cls(c1=c1, c2=c2).fit(train)
            pred = [
                {"tokens": row["tokens"], "entities": bio_to_entities(row["tokens"], tags), "relations": []}
                for row, tags in zip(dev, candidate.predict(dev))
            ]
            score = entity_scores(gold, pred)["micro_f1"]
            if score > best["micro_f1"]:
                best_model, best = candidate, {"c1": c1, "c2": c2, "micro_f1": score}
        if best_model is None:
            raise ValueError("CRF tuning requires a non-empty development set")
        return best_model, best
