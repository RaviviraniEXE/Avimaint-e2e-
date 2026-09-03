import pandas as pd

from avimaint.normalization.splitting import (
    assert_no_cluster_leakage,
    assign_clusters,
    merge_clusters_by_record,
    split_groups,
)


def test_exact_duplicates_share_cluster() -> None:
    clusters = assign_clusters(["FUEL PUMP FAILED", "fuel pump failed", "BRAKE WORN"])
    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]


def test_group_assignment_has_no_leakage() -> None:
    clusters = ["a", "a", "b", "c", "d", "e"]
    assignment = split_groups(clusters, {"train": 0.7, "validation": 0.15, "test": 0.15}, 42)
    frame = pd.DataFrame({"cluster_id": clusters})
    frame["split"] = frame["cluster_id"].map(assignment)
    assert_no_cluster_leakage(frame)


def test_problem_and_action_from_same_record_share_cluster() -> None:
    merged = merge_clusters_by_record(["p1", "a1", "p2"], ["record-1", "record-1", "record-2"])
    assert merged[0] == merged[1]
    assert merged[0] != merged[2]
