"""Schema-constrained logistic-regression relation-extraction baseline."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluate import relation_scores
from src.features import relation_features


NONE = "__NONE__"


class RelationClassifier:
    def __init__(self, schema: dict, C: float = 1.0):
        self.schema = schema
        self.C = float(C)
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            C=self.C, max_iter=1500, class_weight="balanced", solver="liblinear",
            random_state=42,
        )
        self.constant_label: str | None = None

    def _allowed(self, head_type: str, tail_type: str) -> list[str]:
        return [
            name for name, spec in self.schema.get("relations", {}).items()
            if head_type in spec.get("head", []) and tail_type in spec.get("tail", [])
        ]

    def _examples(self, records: list[dict], with_labels: bool) -> tuple[list[dict], list[str], list[tuple[int, int, int]]]:
        features: list[dict] = []
        labels: list[str] = []
        refs: list[tuple[int, int, int]] = []
        for doc_index, record in enumerate(records):
            entities = record.get("entities", [])
            gold = {(rel["head"], rel["tail"]): rel["type"] for rel in record.get("relations", [])}
            for head_index, head in enumerate(entities):
                for tail_index, tail in enumerate(entities):
                    if head_index == tail_index or not self._allowed(head["type"], tail["type"]):
                        continue
                    feat = relation_features(record["tokens"], head, tail)
                    feat["allowed"] = "|".join(self._allowed(head["type"], tail["type"]))
                    features.append(feat)
                    labels.append(gold.get((head_index, tail_index), NONE) if with_labels else NONE)
                    refs.append((doc_index, head_index, tail_index))
        return features, labels, refs

    def fit(self, records: list[dict]) -> "RelationClassifier":
        features, labels, _ = self._examples(records, True)
        if not features:
            self.constant_label = NONE
            return self
        matrix = self.vectorizer.fit_transform(features)
        unique = sorted(set(labels))
        if len(unique) == 1:
            self.constant_label = unique[0]
        else:
            self.model.fit(matrix, labels)
        return self

    def predict(self, records: list[dict]) -> list[list[dict]]:
        output = [[] for _ in records]
        features, _, refs = self._examples(records, False)
        if not features:
            return output
        labels = (
            [self.constant_label] * len(features)
            if self.constant_label is not None
            else self.model.predict(self.vectorizer.transform(features))
        )
        for label, (doc, head, tail) in zip(labels, refs):
            if label != NONE:
                output[doc].append({"type": str(label), "head": head, "tail": tail})
        return output

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def tuned(
        cls,
        schema: dict,
        train: list[dict],
        dev: list[dict],
        grid: list[float] | None = None,
    ) -> tuple["RelationClassifier", dict]:
        grid = grid or [0.25, 0.5, 1.0, 2.0, 4.0]
        best_model: RelationClassifier | None = None
        best = {"micro_f1": -1.0}
        gold = [{"tokens": row["tokens"], "entities": row["entities"], "relations": row.get("relations", [])} for row in dev]
        for c_value in grid:
            candidate = cls(schema, C=c_value).fit(train)
            pred = [
                {"tokens": row["tokens"], "entities": row["entities"], "relations": rels}
                for row, rels in zip(dev, candidate.predict(dev))
            ]
            score = relation_scores(gold, pred)["micro_f1"]
            if score > best["micro_f1"]:
                best_model, best = candidate, {"C": c_value, "micro_f1": score}
        if best_model is None:
            raise ValueError("Relation tuning requires a non-empty development set")
        return best_model, best

