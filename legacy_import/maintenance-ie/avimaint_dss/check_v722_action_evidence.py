"""Deterministic regression for V7.2.2 cluster action expansion."""
import importlib.util
import sys
import types
from types import SimpleNamespace

import pandas as pd

if importlib.util.find_spec("rank_bm25") is None:
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object
    sys.modules["rank_bm25"] = rank_bm25

import core.recommend as recommendation_module
from core.evidence_policy import classify_evidence
from core.recommend import Recommender
from core.retrieval import Hit


class StaticRetriever:
    def search(self, *args, **kwargs):
        # One representative for cluster A and one for cluster B.
        return [
            Hit(0, 0.80, 0.80, 1.0, {"struct": 1.0}),
            Hit(3, 0.70, 0.70, 1.0, {"struct": 1.0}),
        ]


class OneClusterRetriever:
    def search(self, *args, **kwargs):
        return [Hit(0, 0.80, 0.80, 1.0, {"struct": 1.0})]


class CalibratorProbe:
    def __init__(self):
        self.calls = 0

    def available(self):
        return True

    def predict(self, *args):
        self.calls += 1
        return 0.75

    def status(self):
        return "synthetic-calibrator"


def structure(*args, **kwargs):
    return SimpleNamespace(
        source="spert",
        components=["intake"],
        faults=["leak"],
        locations=[],
        entities=[{"type": "MAINT_ITEM", "text": "intake"}, {"type": "FAULT", "text": "leak"}],
        relations=[{"type": "ISSUE_ON_ITEM"}],
    )


df = pd.DataFrame(
    [
        # The representative-only pipeline sees Replace for cluster A.
        dict(ident="A1", problem="INTAKE LEAKING", problem_norm="intake leaking", action="REPLACED GASKET", action_family="Replace", outcome="positive", cluster_id="A"),
        # This legitimate sibling action must not disappear.
        dict(ident="A2", problem="INTAKE LEAKING", problem_norm="intake leaking", action="INSPECTED INTAKE", action_family="Diagnose", outcome="unknown", cluster_id="A"),
        # Negative sibling must remain visible but must not vote.
        dict(ident="A3", problem="INTAKE LEAKING", problem_norm="intake leaking", action="REPAIR UNSUCCESSFUL", action_family="Repair", outcome="negative", cluster_id="A"),
        dict(ident="B1", problem="INTAKE GASKET LEAK", problem_norm="intake gasket leak", action="INSPECTED AND CONFIRMED LEAK", action_family="Diagnose", outcome="positive", cluster_id="B"),
        # Duplicate Replace in A must not become a second independent vote.
        dict(ident="A4", problem="INTAKE LEAKING", problem_norm="intake leaking", action="INSTALLED NEW GASKET", action_family="Replace", outcome="positive", cluster_id="A"),
    ]
)
for column, value in {
    "components": [["intake"]] * len(df),
    "faults": [["leak"]] * len(df),
    "locations": [[] for _ in range(len(df))],
    "problem_entity_types": [["FAULT", "MAINT_ITEM"]] * len(df),
    "problem_relation_types": [["ISSUE_ON_ITEM"]] * len(df),
}.items():
    df[column] = value

original_extract = recommendation_module.extract_structure
recommendation_module.extract_structure = structure
try:
    calibrator = CalibratorProbe()
    recommender = Recommender(
        df,
        StaticRetriever(),
        spert_client=object(),
        calibrator=calibrator,
        enable_compound_decomposition=False,
        require_anchor_for_action=True,
        abstain_on_single_cluster=False,
    )
    base = StaticRetriever().search()
    expanded = recommender._expand_retrieved_clusters(base)
    assert [hit.idx for hit in expanded] == [0, 1, 2, 4, 3]

    ranked = dict(recommender._family_evidence(expanded))
    assert set(ranked) == {"Replace", "Diagnose"}
    assert set(ranked["Replace"]["clusters"]) == {"A"}
    assert set(ranked["Diagnose"]["clusters"]) == {"A", "B"}

    result = recommender.recommend("INTAKE LEAKING")
    assert result.evidence_family == "Diagnose"
    assert result.support_clusters == 2
    assert {strategy.family for strategy in result.strategies} == {"Replace", "Diagnose"}
    assert {case.ident for case in result.negative_evidence} == {"A3"}
    assert len(result.nearest_cases) == 2, "Problem-side nearest cases must remain cluster-distinct"
    assert result.historical_agreement_probability is None
    assert calibrator.calls == 0, "Mismatched frozen calibrator must not be called"

    one_cluster_df = df.iloc[[0, 1]].copy().reset_index(drop=True)
    one_cluster_df.loc[1, "outcome"] = "positive"
    one_cluster = Recommender(
        one_cluster_df,
        OneClusterRetriever(),
        spert_client=object(),
        calibrator=CalibratorProbe(),
        enable_compound_decomposition=False,
        require_anchor_for_action=True,
        abstain_on_single_cluster=False,
    ).recommend("INTAKE LEAKING")
    assert one_cluster.badge == "limited" and not one_cluster.abstain
    assert one_cluster.evidence_family == "Replace"
    assert {strategy.family for strategy in one_cluster.strategies} == {"Replace", "Diagnose"}
finally:
    recommendation_module.extract_structure = original_extract

limited = classify_evidence(
    evidence_family="Inspect",
    support=1,
    family_margin=0.0,
    coverage=0.0,
    has_anchor=True,
    require_anchor=True,
    limited_min_coverage=0.50,
    allow_single_cluster=True,
)
assert limited.badge == "limited" and not limited.abstain
assert "weak anchor coverage" in limited.note

unanchored = classify_evidence(
    evidence_family="Inspect",
    support=1,
    family_margin=0.0,
    coverage=0.0,
    has_anchor=False,
    require_anchor=True,
    allow_single_cluster=True,
)
assert unanchored.abstain and unanchored.tier == "unanchored"

print("V7_2_2_ACTION_EVIDENCE_OK")
print("CLUSTER_A_FAMILIES Replace Diagnose")
print("DUPLICATE_CLUSTER_VOTES_SUPPRESSED")
print("SINGLE_CLUSTER_LIMITED_VISIBLE")
print("SINGLE_CLUSTER_ALTERNATIVES_VISIBLE")
print("RQ5_MISMATCHED_CALIBRATOR_SUPPRESSED")
