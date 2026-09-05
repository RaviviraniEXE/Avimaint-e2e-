"""Operational evidence-presentation policy.

This does not change the frozen RQ4 ranking or DEV-fitted RQ5 calibration.
Phase 3 only changes whether already-retrieved historical evidence is surfaced
as a primary planning-support suggestion.

A one-cluster result can be shown as LIMITED historical evidence when it is
query-anchored and has sufficient anchor coverage. RQ5 probability is never used
to force a recommendation.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceDecision:
    badge: str
    abstain: bool
    tier: str
    note: str


def classify_evidence(
    *,
    evidence_family: str,
    support: int,
    family_margin: float,
    coverage: float,
    has_anchor: bool,
    require_anchor: bool,
    strong_min_clusters: int = 3,
    moderate_min_clusters: int = 2,
    strong_min_margin: float = 0.08,
    moderate_min_margin: float = 0.03,
    limited_min_coverage: float = 0.50,
    allow_single_cluster: bool = True,
) -> EvidenceDecision:
    if not evidence_family:
        return EvidenceDecision("abstain", True, "none",
                                "No usable historical action family was retrieved.")
    if require_anchor and not has_anchor:
        return EvidenceDecision("abstain", True, "unanchored",
                                "No grounded component/fault anchor was available.")

    if (
        support >= int(strong_min_clusters)
        and family_margin >= float(strong_min_margin)
        and coverage >= 0.75
    ):
        return EvidenceDecision(
            "strong", False, "strong",
            "Corroborated by several independent historical problem clusters.",
        )

    if (
        support >= int(moderate_min_clusters)
        and family_margin >= float(moderate_min_margin)
        and coverage >= 0.50
    ):
        return EvidenceDecision(
            "moderate", False, "moderate",
            "Supported by multiple independent historical problem clusters.",
        )

    if (
        allow_single_cluster
        and int(support) == 1
        # A single grounded historical cluster remains visible as LIMITED
        # evidence.  Coverage controls the warning text, not whether the
        # traceable action disappears from Diagnose.  The anchor requirement
        # above and the validated raw-SpERT runtime gate remain in force.
    ):
        coverage_note = (
            " The selected action family has weak anchor coverage; verify the "
            "source record carefully."
            if coverage < float(limited_min_coverage)
            else ""
        )
        return EvidenceDecision(
            "limited", False, "limited",
            "One independent historical cluster supports this action family; "
            "show as traceable limited evidence, not as a recurring strategy."
            + coverage_note,
        )

    return EvidenceDecision(
        "exploratory", True, "exploratory",
        "Historical evidence exists, but the support/grounding gate is too weak "
        "for a primary suggestion.",
    )
