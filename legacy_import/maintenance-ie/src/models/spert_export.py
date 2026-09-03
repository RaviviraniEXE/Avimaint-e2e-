"""Export frozen AviMaint/MaintIE splits to the official SpERT JSON format."""
from __future__ import annotations

import json
from pathlib import Path

from src.schema import load_schema


def export_spert(records: list[dict], output_dir: str, split_name: str) -> str:
    target = Path(output_dir) / f"{split_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    documents = []
    for record in records:
        entities = [
            {"type": entity["type"], "start": int(entity["start"]), "end": int(entity["end"])}
            for entity in record.get("entities", [])
        ]
        relations = [
            {"type": relation["type"], "head": int(relation["head"]), "tail": int(relation["tail"])}
            for relation in record.get("relations", [])
        ]
        documents.append({
            "tokens": record["tokens"], "entities": entities, "relations": relations,
            "orig_id": str(record.get("ident", "")),
        })
    target.write_text(json.dumps(documents, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(target)


def write_types(output_dir: str, schema_path: str = "config/schema.yaml") -> str:
    schema = load_schema(schema_path)
    entity_types = {
        name: {"short": name, "verbose": spec.get("canonical", name)}
        for name, spec in schema.get("entities", {}).items()
    }
    relation_types = {
        name: {"short": name, "verbose": name, "symmetric": False}
        for name in schema.get("relations", {})
    }
    target = Path(output_dir) / "avimaint_types.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entities": entity_types, "relations": relation_types}, indent=2), encoding="utf-8")
    return str(target)


def write_config(output_dir: str) -> str:
    root = Path(output_dir).resolve()
    target = root / "avimaint_spert.conf"
    text = f"""[1]
label = avimaint_spert
model_type = spert
model_path = bert-base-cased
tokenizer_path = bert-base-cased
train_path = {root.as_posix()}/train.json
valid_path = {root.as_posix()}/dev.json
types_path = {root.as_posix()}/avimaint_types.json
save_path = {root.as_posix()}/save
log_path = {root.as_posix()}/log
train_batch_size = 2
eval_batch_size = 1
neg_entity_count = 100
neg_relation_count = 100
epochs = 20
lr = 5e-5
lr_warmup = 0.1
weight_decay = 0.01
max_grad_norm = 1.0
max_span_size = 10
rel_filter_threshold = 0.4
size_embedding = 25
prop_drop = 0.1
final_eval = true
store_predictions = true
store_examples = false
seed = 42
"""
    target.write_text(text, encoding="utf-8")
    return str(target)

