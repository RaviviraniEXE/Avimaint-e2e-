from pathlib import Path

from avimaint.schemas.registry import load_schema


def test_frozen_aviation_schema_inventory() -> None:
    path = Path("schemas/aviation_compact_v1.yaml")
    schema = load_schema(path)
    assert len(schema.entity_types) == 9
    assert len(schema.relation_types) == 11
    assert schema.validate_relation("ISSUE_ON_ITEM", "FAULT", "MAINT_ITEM")
    assert not schema.validate_relation("ISSUE_ON_ITEM", "MAINT_ITEM", "FAULT")

