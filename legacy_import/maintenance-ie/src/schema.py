"""Load the schema and derive the BIO tag set and relation type constraints."""
from __future__ import annotations

import yaml


def load_schema(path: str = "config/schema.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def entity_types(schema: dict, include_hybrid: bool = True):
    return [k for k, v in schema["entities"].items()
            if include_hybrid or v.get("tier") != "hybrid"]


def bio_tags(schema: dict, include_hybrid: bool = True):
    """['O', 'B-MAINT_ITEM', 'I-MAINT_ITEM', ...]"""
    tags = ["O"]
    for e in entity_types(schema, include_hybrid):
        tags += [f"B-{e}", f"I-{e}"]
    return tags


def relation_types(schema: dict, include_hybrid: bool = True):
    return [k for k, v in schema["relations"].items()
            if include_hybrid or v.get("tier") != "hybrid"]


def allowed_pair(schema: dict, rel: str, head_type: str, tail_type: str) -> bool:
    r = schema["relations"][rel]
    return head_type in r["head"] and tail_type in r["tail"]


def candidate_relations(schema: dict, head_type: str, tail_type: str):
    """All relation labels whose type constraints admit (head_type, tail_type)."""
    return [rel for rel in schema["relations"]
            if allowed_pair(schema, rel, head_type, tail_type)]

