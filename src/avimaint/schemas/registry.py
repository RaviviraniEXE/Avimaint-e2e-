"""Schema loading and relation-signature validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avimaint.configuration import load_yaml


@dataclass(frozen=True)
class RelationSignature:
    sources: frozenset[str]
    targets: frozenset[str]


@dataclass(frozen=True)
class SchemaDefinition:
    schema_id: str
    entity_types: frozenset[str]
    relation_types: dict[str, RelationSignature]
    allow_dataset_defined_labels: bool = False

    def validate_relation(
        self,
        relation_type: str,
        source_type: str,
        target_type: str,
    ) -> bool:
        signature = self.relation_types.get(relation_type)
        if signature is None:
            return self.allow_dataset_defined_labels
        return source_type in signature.sources and target_type in signature.targets


def load_schema(path: str | Path) -> SchemaDefinition:
    payload = load_yaml(path)
    schema = payload["schema"]
    relation_types: dict[str, RelationSignature] = {}
    for name, definition in dict(schema.get("relation_types", {})).items():
        relation_types[name] = RelationSignature(
            sources=frozenset(definition.get("source", [])),
            targets=frozenset(definition.get("target", [])),
        )
    entity_types = schema.get("entity_types", schema.get("top_level_entity_types", []))
    return SchemaDefinition(
        schema_id=str(schema["id"]),
        entity_types=frozenset(entity_types),
        relation_types=relation_types,
        allow_dataset_defined_labels=bool(schema.get("allow_dataset_defined_labels", False)),
    )

