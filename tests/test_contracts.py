import pytest

from avimaint.contracts import Entity, MaintenanceRecord, Relation


def test_valid_record_round_trip() -> None:
    text = "PUMP LEAKING. REPLACED SEAL."
    record = MaintenanceRecord(
        dataset_id="demo",
        record_id="1",
        full_text=text,
        schema_id="aviation_compact_v1",
        entities=[
            Entity("e1", "MAINT_ITEM", 0, 4, "PUMP"),
            Entity("e2", "ABN_PROC", 5, 12, "LEAKING"),
        ],
        relations=[Relation("ISSUE_ON_ITEM", "e2", "e1")],
    )
    restored = MaintenanceRecord.from_dict(record.to_dict())
    assert restored.record_id == record.record_id


def test_unknown_relation_endpoint_is_rejected() -> None:
    record = MaintenanceRecord(
        dataset_id="demo",
        record_id="1",
        full_text="PUMP",
        schema_id="aviation_compact_v1",
        entities=[Entity("e1", "MAINT_ITEM", 0, 4, "PUMP")],
        relations=[Relation("HAS_PART", "e1", "missing")],
    )
    with pytest.raises(ValueError):
        record.validate()

