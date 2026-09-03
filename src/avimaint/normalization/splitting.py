"""Exact/near-duplicate clustering and deterministic group-aware splits."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from avimaint.normalization.io import read_table, require_columns, sha256_file, write_json, write_table


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def assign_clusters(
    texts: list[str], threshold: float = 0.90, ngram_range: tuple[int, int] = (3, 5)
) -> list[str]:
    groups = DisjointSet(len(texts))
    exact: dict[str, int] = {}
    normalized = [fingerprint(text) for text in texts]
    for index, value in enumerate(normalized):
        if value in exact:
            groups.union(index, exact[value])
        else:
            exact[value] = index

    if len(texts) > 1 and threshold < 1.0:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1)
        matrix = vectorizer.fit_transform(normalized)
        neighbors = NearestNeighbors(metric="cosine", radius=max(0.0, 1.0 - threshold), n_jobs=-1)
        neighbors.fit(matrix)
        graph = neighbors.radius_neighbors_graph(matrix, mode="distance")
        rows, columns = graph.nonzero()
        for left, right in zip(rows.tolist(), columns.tolist(), strict=False):
            if left < right and graph[left, right] <= 1.0 - threshold + 1e-12:
                groups.union(left, right)

    root_to_label: dict[int, str] = {}
    labels: list[str] = []
    for index in range(len(texts)):
        root = groups.find(index)
        if root not in root_to_label:
            value = hashlib.sha256(normalized[root].encode("utf-8")).hexdigest()[:12]
            root_to_label[root] = f"normgrp-{value}"
        labels.append(root_to_label[root])
    return labels


def merge_clusters_by_record(cluster_ids: list[str], record_ids: list[str]) -> list[str]:
    """Keep every field from one maintenance record in one final cluster."""
    groups = DisjointSet(len(cluster_ids))
    first_cluster: dict[str, int] = {}
    first_record: dict[str, int] = {}
    for index, (cluster_id, record_id) in enumerate(
        zip(cluster_ids, record_ids, strict=False)
    ):
        if cluster_id in first_cluster:
            groups.union(index, first_cluster[cluster_id])
        else:
            first_cluster[cluster_id] = index
        if record_id in first_record:
            groups.union(index, first_record[record_id])
        else:
            first_record[record_id] = index
    root_to_label: dict[int, str] = {}
    labels: list[str] = []
    for index in range(len(cluster_ids)):
        root = groups.find(index)
        if root not in root_to_label:
            signature = f"{cluster_ids[root]}::{record_ids[root]}"
            root_to_label[root] = f"normgrp-{hashlib.sha256(signature.encode()).hexdigest()[:12]}"
        labels.append(root_to_label[root])
    return labels


def split_groups(
    cluster_ids: list[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    sizes = pd.Series(cluster_ids).value_counts().to_dict()
    items = list(sizes.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    total = sum(sizes.values())
    target = {name: total * ratio for name, ratio in ratios.items()}
    current = {name: 0 for name in ratios}
    assignment: dict[str, str] = {}
    for cluster_id, size in items:
        destination = min(ratios, key=lambda name: current[name] / max(target[name], 1.0))
        assignment[cluster_id] = destination
        current[destination] += size
    return assignment


def assert_no_cluster_leakage(frame: pd.DataFrame) -> None:
    leaking = frame.groupby("cluster_id")["split"].nunique()
    leaking = leaking[leaking > 1]
    if not leaking.empty:
        raise AssertionError(f"Clusters cross splits: {leaking.index[:5].tolist()}")


def run_split(config: dict[str, Any]) -> Path:
    frame = read_table(config["input_path"])
    require_columns(
        frame,
        [config["id_column"], config["record_id_column"], config["source_column"]],
        "normalization pairs",
    )
    near = config["near_duplicate"]
    threshold = float(near["similarity_threshold"]) if near.get("enabled", True) else 1.0
    ngram = tuple(int(value) for value in near.get("char_ngram_range", [3, 5]))
    frame = frame.copy()
    text_clusters = assign_clusters(
        frame[config["source_column"]].fillna("").astype(str).tolist(), threshold, ngram
    )
    frame["cluster_id"] = merge_clusters_by_record(
        text_clusters,
        frame[config["record_id_column"]].astype(str).tolist(),
    )
    assignment = split_groups(frame["cluster_id"].tolist(), config["ratios"], int(config["seed"]))
    frame["split"] = frame["cluster_id"].map(assignment)
    assert_no_cluster_leakage(frame)
    output = Path(config["output_path"])
    write_table(frame, output)
    counts = frame["split"].value_counts().sort_index().to_dict()
    manifest = {
        "source_path": str(config["input_path"]),
        "source_sha256": sha256_file(config["input_path"]),
        "output_path": str(output),
        "seed": int(config["seed"]),
        "similarity_threshold": threshold,
        "records": int(len(frame)),
        "clusters": int(frame["cluster_id"].nunique()),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
    }
    write_json(manifest, config["manifest_path"])
    return output
