"""Shared, dependency-light data contracts used across isolated environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float | None = None

    def validate(self, full_text: str) -> None:
        if self.start < 0 or self.end <= self.start or self.end > len(full_text):
            raise ValueError(f"Invalid span for {self.entity_id}: {self.start}:{self.end}")
        if full_text[self.start : self.end] != self.text:
            raise ValueError(f"Span text mismatch for {self.entity_id}")


@dataclass(frozen=True)
class Relation:
    relation_type: str
    source_entity_id: str
    target_entity_id: str
    confidence: float | None = None


@dataclass
class MaintenanceRecord:
    dataset_id: str
    record_id: str
    full_text: str
    schema_id: str
    problem_text: str | None = None
    action_text: str | None = None
    representation: str = "raw"
    split: str | None = None
    cluster_id: str | None = None
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_id or not self.record_id or not self.schema_id:
            raise ValueError("dataset_id, record_id and schema_id are required")
        entity_ids: set[str] = set()
        for entity in self.entities:
            entity.validate(self.full_text)
            if entity.entity_id in entity_ids:
                raise ValueError(f"Duplicate entity ID: {entity.entity_id}")
            entity_ids.add(entity.entity_id)
        for relation in self.relations:
            if relation.source_entity_id not in entity_ids:
                raise ValueError(f"Unknown source entity: {relation.source_entity_id}")
            if relation.target_entity_id not in entity_ids:
                raise ValueError(f"Unknown target entity: {relation.target_entity_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaintenanceRecord":
        record = cls(
            dataset_id=str(payload["dataset_id"]),
            record_id=str(payload["record_id"]),
            full_text=str(payload["full_text"]),
            schema_id=str(payload["schema_id"]),
            problem_text=payload.get("problem_text"),
            action_text=payload.get("action_text"),
            representation=str(payload.get("representation", "raw")),
            split=payload.get("split"),
            cluster_id=payload.get("cluster_id"),
            entities=[Entity(**item) for item in payload.get("entities", [])],
            relations=[Relation(**item) for item in payload.get("relations", [])],
            provenance=dict(payload.get("provenance", {})),
        )
        record.validate()
        return record

