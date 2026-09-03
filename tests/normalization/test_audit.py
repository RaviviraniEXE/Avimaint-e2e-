from avimaint.normalization.audit import classify_pair


def test_classifies_case_only_change() -> None:
    category, _ = classify_pair("FUEL PUMP", "fuel pump")
    assert category == "formatting_only"


def test_flags_large_addition() -> None:
    category, _ = classify_pair("PUMP", "pump replaced and operational test satisfactory")
    assert category == "possible_unsupported_addition"
