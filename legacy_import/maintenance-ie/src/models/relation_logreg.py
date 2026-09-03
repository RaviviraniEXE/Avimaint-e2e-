"""Schema-constrained logistic-regression relation-extraction baseline.

The baseline keeps every schema-valid candidate pair and uses balanced class
weights.  Tuning is DEV-only.  Feature/vectorizer work is cached across C values
so the live grid search does not repeat identical preprocessing.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluate import relation_scores
from src.features import relation_features
from src.progress import LiveProgress, trace_event


NONE = "__NONE__"


class RelationClassifier:
    def __init__(self, schema: dict, C: float = 1.0):
        self.schema = schema
        self.C = float(C)
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            C=self.C,
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        self.constant_label: str | None = None
        self.fit_stats: dict = {}

    def _allowed(self, head_type: str, tail_type: str) -> list[str]:
        return [
            name
            for name, spec in self.schema.get("relations", {}).items()
            if head_type in spec.get("head", []) and tail_type in spec.get("tail", [])
        ]

    def _examples(
        self, records: list[dict], with_labels: bool
    ) -> tuple[list[dict], list[str], list[tuple[int, int, int]]]:
        features: list[dict] = []
        labels: list[str] = []
        refs: list[tuple[int, int, int]] = []
        for doc_index, record in enumerate(records):
            entities = record.get("entities", [])
            gold = {
                (rel["head"], rel["tail"]): rel["type"]
                for rel in record.get("relations", [])
            }
            for head_index, head in enumerate(entities):
                for tail_index, tail in enumerate(entities):
                    allowed = self._allowed(head["type"], tail["type"])
                    if head_index == tail_index or not allowed:
                        continue
                    feat = relation_features(record["tokens"], head, tail)
                    feat["allowed"] = "|".join(allowed)
                    features.append(feat)
                    labels.append(
                        gold.get((head_index, tail_index), NONE) if with_labels else NONE
                    )
                    refs.append((doc_index, head_index, tail_index))
        return features, labels, refs

    @staticmethod
    def _stats(labels: list[str], records: int, features: int) -> dict:
        counts = Counter(labels)
        positives = sum(v for k, v in counts.items() if k != NONE)
        negatives = counts.get(NONE, 0)
        return {
            "records": int(records),
            "candidate_pairs": int(features),
            "positive_pairs": int(positives),
            "negative_pairs": int(negatives),
            "negative_to_positive_ratio": (
                round(negatives / positives, 4) if positives else None
            ),
            "label_counts": dict(sorted(counts.items())),
            "negative_policy": "all schema-valid negatives retained",
            "class_weight": "balanced",
            "random_state": 42,
        }

    def _fit_matrix(self, matrix, labels: list[str]) -> "RelationClassifier":
        unique = sorted(set(labels))
        if len(unique) == 1:
            self.constant_label = unique[0]
        else:
            self.model.fit(matrix, labels)
        return self

    @staticmethod
    def _labels_to_output(
        labels: list[str], refs: list[tuple[int, int, int]], n_records: int
    ) -> list[list[dict]]:
        output: list[list[dict]] = [[] for _ in range(n_records)]
        for label, (doc, head, tail) in zip(labels, refs):
            if label != NONE:
                output[doc].append(
                    {"type": str(label), "head": int(head), "tail": int(tail)}
                )
        return output

    def fit(self, records: list[dict]) -> "RelationClassifier":
        features, labels, _ = self._examples(records, True)
        self.fit_stats = self._stats(labels, len(records), len(features))
        trace_event("logreg_fit_data", C=self.C, **self.fit_stats)
        if not features:
            self.constant_label = NONE
            return self
        matrix = self.vectorizer.fit_transform(features)
        return self._fit_matrix(matrix, labels)

    def predict(self, records: list[dict]) -> list[list[dict]]:
        features, _, refs = self._examples(records, False)
        if not features:
            return [[] for _ in records]
        labels = (
            [self.constant_label] * len(features)
            if self.constant_label is not None
            else self.model.predict(self.vectorizer.transform(features)).tolist()
        )
        return self._labels_to_output(labels, refs, len(records))

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
        if not train or not dev:
            raise ValueError("Relation tuning requires non-empty train and development sets")

        # Candidate generation and DictVectorizer fitting do not depend on C.
        # Build them once and train only the logistic-regression estimator per C.
        template = cls(schema, C=float(grid[0]))
        train_features, train_labels, _ = template._examples(train, True)
        stats = template._stats(train_labels, len(train), len(train_features))
        if train_features:
            vectorizer = DictVectorizer(sparse=True)
            train_matrix = vectorizer.fit_transform(train_features)
        else:
            vectorizer = DictVectorizer(sparse=True)
            train_matrix = None

        dev_features, _, dev_refs = template._examples(dev, False)
        dev_matrix = vectorizer.transform(dev_features) if dev_features and train_features else None
        gold = [
            {
                "tokens": row["tokens"],
                "entities": row["entities"],
                "relations": row.get("relations", []),
            }
            for row in dev
        ]
        trace_event(
            "logreg_tuning_setup",
            configurations=len(grid),
            dev_records=len(dev),
            **stats,
        )

        best_model: RelationClassifier | None = None
        best = {"micro_f1": -1.0}
        history: list[dict] = []
        progress = LiveProgress("LogReg RE DEV tuning", len(grid))

        for index, c_value in enumerate(grid, start=1):
            detail = f"config {index}/{len(grid)} C={float(c_value):g}"
            progress.begin(detail)
            try:
                candidate = cls(schema, C=float(c_value))
                candidate.vectorizer = vectorizer
                candidate.fit_stats = dict(stats)
                if not train_features:
                    candidate.constant_label = NONE
                else:
                    candidate._fit_matrix(train_matrix, train_labels)

                if not dev_features:
                    rels = [[] for _ in dev]
                else:
                    labels = (
                        [candidate.constant_label] * len(dev_features)
                        if candidate.constant_label is not None
                        else candidate.model.predict(dev_matrix).tolist()
                    )
                    rels = candidate._labels_to_output(labels, dev_refs, len(dev))
                pred = [
                    {
                        "tokens": row["tokens"],
                        "entities": row["entities"],
                        "relations": doc_rels,
                    }
                    for row, doc_rels in zip(dev, rels)
                ]
                score = relation_scores(gold, pred)["micro_f1"]
                is_best = score > best["micro_f1"]
                history.append({"C": float(c_value), "micro_f1": float(score)})
                if is_best:
                    best_model = candidate
                    best = {"C": float(c_value), "micro_f1": score}
                progress.finish(
                    metric=f"DEV gold-entity relation micro-F1={score:.4f}",
                    is_best=is_best,
                    extra=(
                        f"pairs={stats['candidate_pairs']} (+{stats['positive_pairs']}/-{stats['negative_pairs']})"
                    ),
                )
            except Exception as exc:
                progress.fail(exc)
                raise

        if best_model is None:
            raise ValueError("Relation tuning could not select a model")
        progress.close(
            f"selected C={best['C']:g}, DEV F1={best['micro_f1']:.4f}"
        )
        best["history"] = history
        best["training_stats"] = stats
        return best_model, best
