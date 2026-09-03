"""Build a complete, lightweight maintenance knowledge graph.

The builder keeps the full 9-entity/11-relation SpERT extraction as compact
CSV/JSONL tables. It aggregates repeated facts by distinct work-order support,
retains record-level provenance, and creates small planning views for
AviMaint-DSS. The dashboard should render only filtered/top-N subgraphs.

Default outputs under outputs/kg/<name>/:
  nodes.csv, edges.csv, edge_evidence.jsonl, records.csv
  dashboard_predictions.jsonl (AviMaint-DSS v5 prebuilt-prediction format)
  view_item_faults.csv, view_fault_actions.csv, view_fault_solutions.csv
  view_item_parts.csv, view_fault_contexts.csv, view_action_references.csv
  manifest.json, summary.txt

GraphML and NetworkX node-link JSON are optional because AviMaint-DSS v5 builds
small top-N/focused graphs directly from tabular data. Enable them only for
offline graph-tool analysis with --export-graph-files.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict


ENTITY_TYPES = {
    "MAINT_ITEM", "ACTION", "FAULT", "ABN_PROC", "LOC", "OP_CTX",
    "TECH_OBS", "OUTCOME", "REFERENCE",
}
RELATION_TYPES = {
    "HAS_LOCATION", "HAS_PART", "ISSUE_ON_ITEM", "OCCURS_UNDER_CONTEXT",
    "OBSERVATION_OF_ITEM", "ACTION_ON_ITEM", "ACTION_USES_ITEM",
    "ACTION_ADDRESSES_ISSUE", "ACTION_INVESTIGATES_ISSUE",
    "ACTION_RESULTS_IN_OUTCOME", "ACTION_FOLLOWS_REFERENCE",
}

# Optional, reviewed whole-mention aliases only. Heuristic singularisation is
# intentionally avoided because it corrupts valid aviation terms such as "bus".
ABBREV: dict[str, str] = {}


def canon(surface: str, _etype: str) -> str:
    """Conservatively canonicalise a surface without changing its meaning."""
    text = unicodedata.normalize("NFKC", str(surface)).casefold()
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:-")
    return ABBREV.get(text, text)


def _files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    missing = [path for path in files if not os.path.isfile(path)]
    if not files or missing:
        raise SystemExit(f"Prediction input not found: {missing or pattern}")
    return files


def load_records(pattern: str) -> list[dict]:
    records: list[dict] = []
    for filename in _files(pattern):
        if filename.lower().endswith(".jsonl"):
            with open(filename, encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise SystemExit(f"Invalid JSONL at {filename}:{line_number}: {exc}") from exc
        else:
            with open(filename, encoding="utf-8") as source:
                payload = json.load(source)
            if not isinstance(payload, list):
                raise SystemExit(f"Prediction JSON must contain a list: {filename}")
            records.extend(payload)
    return records


def load_index(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid index JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row.get("tokens"), list):
                raise SystemExit(f"Index row {line_number} has no token list: {path}")
            rows.append(row)
    return rows


def attach_index(records: list[dict], index_path: str | None) -> None:
    if index_path:
        index = load_index(index_path)
        if len(index) != len(records):
            raise SystemExit(
                f"Prediction/index count mismatch: predictions={len(records)}, index={len(index)}"
            )
        for position, (record, row) in enumerate(zip(records, index)):
            if not record.get("tokens"):
                record["tokens"] = row["tokens"]
            elif record["tokens"] != row["tokens"]:
                raise SystemExit(f"Token mismatch between prediction and index at row {position}")
            record.setdefault("ident", row.get("ident", position))

    seen_ids: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record.get("tokens"), list):
            raise SystemExit(f"Prediction row {position} has no tokens; provide --tokens")
        record.setdefault("ident", position)
        rid = str(record["ident"])
        if rid in seen_ids:
            raise SystemExit(f"Duplicate record ident in prediction/index data: {rid}")
        seen_ids.add(rid)


def entity_surface(record: dict, entity: dict) -> str:
    tokens = record["tokens"]
    start, end = entity.get("start"), entity.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(tokens)):
        raise ValueError(f"invalid entity span start={start}, end={end}, tokens={len(tokens)}")
    return " ".join(str(token) for token in tokens[start:end])


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise SystemExit("--name must contain at least one letter or number")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="predictions JSON/JSONL/glob")
    parser.add_argument("--name", default="aviation")
    parser.add_argument("--tokens", help="full_index.jsonl in prediction order")
    parser.add_argument(
        "--min-weight", type=int, default=1,
        help="minimum distinct-record support used only in lightweight planning views",
    )
    parser.add_argument(
        "--export-graph-files", action="store_true",
        help="also export GraphML and NetworkX node-link JSON for offline analysis",
    )
    parser.add_argument(
        "--allow-invalid", action="store_true",
        help="skip malformed/unknown predictions instead of failing the build",
    )
    args = parser.parse_args()
    if args.min_weight < 1:
        raise SystemExit("--min-weight must be at least 1")

    name = safe_name(args.name)
    output_dir = Path("outputs") / "kg" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.pred)
    attach_index(records, args.tokens)

    mention_count: Counter = Counter()
    node_records: defaultdict = defaultdict(set)
    edge_records: defaultdict = defaultdict(set)
    solution_records: defaultdict = defaultdict(set)
    record_rows: list[list] = []
    dashboard_rows: list[dict] = []
    invalid = Counter()

    for record_number, record in enumerate(records):
        rid = str(record["ident"])
        entities = record.get("entities", [])
        relations = record.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise SystemExit(f"Prediction row {record_number} has invalid entities/relations containers")

        canonical_nodes: list[tuple[str, str] | None] = []
        dashboard_entities: list[dict] = []
        entity_remap: dict[int, int] = {}
        by_type: defaultdict = defaultdict(list)
        for entity_number, entity in enumerate(entities):
            try:
                entity_type = entity.get("type")
                if entity_type not in ENTITY_TYPES:
                    raise ValueError(f"unknown entity type {entity_type!r}")
                label = canon(entity_surface(record, entity), entity_type)
                if not label:
                    raise ValueError("empty canonical entity surface")
            except (AttributeError, ValueError) as exc:
                invalid["entities"] += 1
                if not args.allow_invalid:
                    raise SystemExit(
                        f"Invalid entity at record={rid}, entity={entity_number}: {exc}"
                    ) from exc
                canonical_nodes.append(None)
                continue

            node = (label, entity_type)
            canonical_nodes.append(node)
            dashboard_entity = {
                "type": entity_type,
                "start": entity["start"],
                "end": entity["end"],
                "text": entity_surface(record, entity),
            }
            if "score" in entity:
                dashboard_entity["score"] = entity["score"]
            entity_remap[entity_number] = len(dashboard_entities)
            dashboard_entities.append(dashboard_entity)
            mention_count[node] += 1
            node_records[node].add(rid)
            by_type[entity_type].append(label)

        def unique_join(values: list[str]) -> str:
            return "; ".join(dict.fromkeys(values))

        record_rows.append([
            rid,
            " ".join(str(token) for token in record["tokens"]),
            unique_join(by_type["MAINT_ITEM"]),
            unique_join(by_type["FAULT"] + by_type["ABN_PROC"]),
            unique_join(by_type["ACTION"]),
        ])

        action_items: defaultdict = defaultdict(list)
        action_issues: list[tuple[int, str, str]] = []
        dashboard_relations: list[dict] = []
        dashboard_relation_signatures: set[tuple[str, int, int]] = set()
        record_edges: set[tuple] = set()
        for relation_number, relation in enumerate(relations):
            try:
                head, tail = relation.get("head"), relation.get("tail")
                relation_type = relation.get("type")
                if relation_type not in RELATION_TYPES:
                    raise ValueError(f"unknown relation type {relation_type!r}")
                if not isinstance(head, int) or not isinstance(tail, int):
                    raise ValueError("head/tail must be integer entity indexes")
                if not (0 <= head < len(canonical_nodes) and 0 <= tail < len(canonical_nodes)):
                    raise ValueError(f"endpoint out of range head={head}, tail={tail}")
                if canonical_nodes[head] is None or canonical_nodes[tail] is None:
                    raise ValueError("endpoint references an invalid entity")
            except (AttributeError, ValueError) as exc:
                invalid["relations"] += 1
                if not args.allow_invalid:
                    raise SystemExit(
                        f"Invalid relation at record={rid}, relation={relation_number}: {exc}"
                    ) from exc
                continue

            head_node, tail_node = canonical_nodes[head], canonical_nodes[tail]
            dashboard_relation = {
                "type": relation_type,
                "head": entity_remap[head],
                "tail": entity_remap[tail],
            }
            if "score" in relation:
                dashboard_relation["score"] = relation["score"]
            signature = (
                relation_type,
                dashboard_relation["head"],
                dashboard_relation["tail"],
            )
            if signature not in dashboard_relation_signatures:
                dashboard_relation_signatures.add(signature)
                dashboard_relations.append(dashboard_relation)
            edge = (
                head_node[0], head_node[1], relation_type,
                tail_node[0], tail_node[1],
            )
            record_edges.add(edge)

            if relation_type in {"ACTION_ON_ITEM", "ACTION_USES_ITEM"} and tail_node[1] == "MAINT_ITEM":
                action_items[head].append(tail_node[0])
            elif relation_type in {"ACTION_ADDRESSES_ISSUE", "ACTION_INVESTIGATES_ISSUE"}:
                kind = "addresses" if relation_type == "ACTION_ADDRESSES_ISSUE" else "investigates"
                action_issues.append((head, tail_node[0], kind))

        for edge in record_edges:
            edge_records[edge].add(rid)

        record_solutions: set[tuple] = set()
        for action_index, issue_label, kind in action_issues:
            action_node = canonical_nodes[action_index]
            if action_node is None:
                continue
            items = list(dict.fromkeys(action_items.get(action_index, []))) or [""]
            for item in items:
                record_solutions.add((issue_label, action_node[0], item, kind))
        for solution in record_solutions:
            solution_records[solution].add(rid)

        dashboard_rows.append({
            "ident": rid,
            "problem_pred": {
                "tokens": record["tokens"],
                "entities": dashboard_entities,
                "relations": dashboard_relations,
            },
        })

    node_rows = [
        [f"{entity_type}:{label}", label, entity_type, mention_count[(label, entity_type)], len(ids)]
        for (label, entity_type), ids in sorted(
            node_records.items(), key=lambda pair: (-len(pair[1]), pair[0][1], pair[0][0])
        )
    ]
    edge_rows = [
        [f"{source_type}:{source}", relation, f"{target_type}:{target}", len(ids)]
        for (source, source_type, relation, target, target_type), ids in sorted(
            edge_records.items(), key=lambda pair: (-len(pair[1]), pair[0][2], pair[0][0], pair[0][3])
        )
    ]
    write_csv(
        output_dir / "nodes.csv",
        ["node_id", "label", "type", "mention_count", "record_count"],
        node_rows,
    )
    write_csv(
        output_dir / "edges.csv",
        ["source", "relation", "target", "record_count"],
        edge_rows,
    )
    write_csv(
        output_dir / "records.csv",
        ["ident", "text", "items", "faults", "actions"],
        record_rows,
    )

    with (output_dir / "edge_evidence.jsonl").open("w", encoding="utf-8") as target:
        for edge, ids in sorted(edge_records.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            source, source_type, relation, destination, destination_type = edge
            target.write(json.dumps({
                "source": f"{source_type}:{source}",
                "relation": relation,
                "target": f"{destination_type}:{destination}",
                "record_ids": sorted(ids),
            }, ensure_ascii=False) + "\n")

    with (output_dir / "dashboard_predictions.jsonl").open("w", encoding="utf-8") as target:
        for row in dashboard_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    def relation_edges(relation: str):
        rows = []
        for (source, source_type, rel, target, target_type), ids in edge_records.items():
            if rel == relation and len(ids) >= args.min_weight:
                rows.append(((source, source_type), (target, target_type), len(ids)))
        return sorted(rows, key=lambda row: (-row[2], row[0][0], row[1][0]))

    write_csv(
        output_dir / "view_item_faults.csv",
        ["item", "issue", "issue_type", "record_count"],
        [[target[0], source[0], source[1], count]
         for source, target, count in relation_edges("ISSUE_ON_ITEM")],
    )

    fault_action_rows = []
    for relation, kind in (
        ("ACTION_ADDRESSES_ISSUE", "addresses"),
        ("ACTION_INVESTIGATES_ISSUE", "investigates"),
    ):
        for action, issue, count in relation_edges(relation):
            fault_action_rows.append([issue[0], action[0], kind, count])
    fault_action_rows.sort(key=lambda row: (-row[3], row[0], row[1]))
    write_csv(
        output_dir / "view_fault_actions.csv",
        ["issue", "action", "kind", "record_count"],
        fault_action_rows,
    )

    write_csv(
        output_dir / "view_fault_solutions.csv",
        ["issue", "action", "item", "kind", "record_count"],
        [[issue, action, item, kind, len(ids)]
         for (issue, action, item, kind), ids in sorted(
             solution_records.items(), key=lambda pair: (-len(pair[1]), pair[0])
         ) if len(ids) >= args.min_weight],
    )
    write_csv(
        output_dir / "view_item_parts.csv",
        ["whole", "part", "record_count"],
        [[source[0], target[0], count]
         for source, target, count in relation_edges("HAS_PART")],
    )
    write_csv(
        output_dir / "view_fault_contexts.csv",
        ["issue", "context", "record_count"],
        [[source[0], target[0], count]
         for source, target, count in relation_edges("OCCURS_UNDER_CONTEXT")],
    )
    write_csv(
        output_dir / "view_action_references.csv",
        ["action", "reference", "record_count"],
        [[source[0], target[0], count]
         for source, target, count in relation_edges("ACTION_FOLLOWS_REFERENCE")],
    )

    if args.export_graph_files:
        import networkx as nx

        graph = nx.MultiDiGraph(name=name)
        for node_id, label, entity_type, mentions, record_count in node_rows:
            graph.add_node(
                node_id, label=label, type=entity_type,
                mention_count=int(mentions), record_count=int(record_count),
            )
        for source, relation, target, record_count in edge_rows:
            graph.add_edge(
                source, target, key=relation,
                relation=relation, record_count=int(record_count),
            )
        nx.write_graphml(graph, output_dir / f"{name}_kg.graphml")
        with (output_dir / f"{name}_kg.json").open("w", encoding="utf-8") as target:
            json.dump(nx.node_link_data(graph), target, ensure_ascii=False)

    by_type = Counter(entity_type for _, entity_type in node_records)
    by_relation = Counter()
    for (_, _, relation, _, _), ids in edge_records.items():
        by_relation[relation] += len(ids)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "schema": {
            "entity_types": sorted(ENTITY_TYPES),
            "relation_types": sorted(RELATION_TYPES),
        },
        "input": {
            "predictions": args.pred,
            "prediction_sha256": sha256(args.pred) if not any(c in args.pred for c in "*?[") else None,
            "token_index": args.tokens,
            "token_index_sha256": sha256(args.tokens),
        },
        "counts": {
            "records": len(records),
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "entity_mentions": sum(mention_count.values()),
            "relation_record_support": sum(len(ids) for ids in edge_records.values()),
            "invalid_entities_skipped": invalid["entities"],
            "invalid_relations_skipped": invalid["relations"],
        },
        "edge_weight_semantics": "number of distinct supporting work orders",
        "min_weight_for_views": args.min_weight,
        "graph_files_exported": bool(args.export_graph_files),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as target:
        target.write(f"Knowledge Graph: {name}\n")
        target.write(
            f"records={len(records)}  nodes={len(node_rows)}  "
            f"edges(aggregated)={len(edge_rows)}\n"
        )
        target.write(
            f"entity mentions={sum(mention_count.values())}  "
            f"relation record-support={sum(len(ids) for ids in edge_records.values())}\n"
        )
        target.write(f"invalid entities skipped={invalid['entities']}\n")
        target.write(f"invalid relations skipped={invalid['relations']}\n\n")
        target.write("nodes by type:\n")
        for entity_type, count in by_type.most_common():
            target.write(f"  {entity_type}: {count} distinct\n")
        target.write("\nedges by relation (distinct-record support):\n")
        for relation, count in by_relation.most_common():
            target.write(f"  {relation}: {count}\n")

    print(
        f"KG '{name}': {len(node_rows)} nodes, {len(edge_rows)} aggregated edges "
        f"from {len(records)} records -> {output_dir}/"
    )
    print("Complete CSV/JSONL evidence + lightweight planning views written.")
    if not args.export_graph_files:
        print("GraphML/node-link JSON skipped (use --export-graph-files for offline analysis).")


if __name__ == "__main__":
    main()
