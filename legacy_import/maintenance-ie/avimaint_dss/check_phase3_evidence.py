from core.evidence_policy import classify_evidence

limited = classify_evidence(
    evidence_family="Replace", support=1, family_margin=0.0, coverage=1.0,
    has_anchor=True, require_anchor=True, allow_single_cluster=True,
)
assert limited.badge == "limited" and not limited.abstain

low_coverage = classify_evidence(
    evidence_family="Replace", support=1, family_margin=0.0, coverage=0.25,
    has_anchor=True, require_anchor=True, allow_single_cluster=True,
)
assert low_coverage.badge == "limited" and not low_coverage.abstain
assert "weak anchor coverage" in low_coverage.note

unanchored = classify_evidence(
    evidence_family="Replace", support=1, family_margin=0.0, coverage=1.0,
    has_anchor=False, require_anchor=True, allow_single_cluster=True,
)
assert unanchored.abstain

strong = classify_evidence(
    evidence_family="Replace", support=3, family_margin=0.10, coverage=0.9,
    has_anchor=True, require_anchor=True,
)
assert strong.badge == "strong" and not strong.abstain

moderate = classify_evidence(
    evidence_family="Repair", support=2, family_margin=0.04, coverage=0.6,
    has_anchor=True, require_anchor=True,
)
assert moderate.badge == "moderate" and not moderate.abstain

print("PHASE3_EVIDENCE_POLICY_OK")
