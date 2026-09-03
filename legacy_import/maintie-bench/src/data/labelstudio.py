"""Label Studio round-trip: our records <-> Label Studio NER+Relations format.

Task text = space-joined tokens (single-space), so tokenization is recoverable
as text.split(' ') and character offsets are deterministic. Pre-annotations are
attached as `predictions` so you correct rather than label from scratch.

Import  (to_tasks)   : records -> Label Studio tasks JSON (with predictions).
Config  (labeling_config) : the XML to paste into the LS project's labeling setup.
Export  (from_export): Label Studio export JSON -> internal records
                       {ident, tokens, bio, entities, relations, exact_group_id}.
"""
from __future__ import annotations

import json
from typing import List

from src.schema import entity_types, load_schema, relation_types

_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
            "#D55E00", "#F0E442", "#999999", "#8C564B"]


def char_offsets(tokens):
    """(start,end) char span of each token in ' '.join(tokens)."""
    offs, pos = [], 0
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    return offs


def to_tasks(records: List[dict]) -> List[dict]:
    tasks = []
    for r in records:
        toks = r["tokens"]
        text = " ".join(toks)
        offs = char_offsets(toks)
        results = []
        eid = {}
        for i, e in enumerate(r.get("entities", [])):
            rid = f"e{i}"
            eid[i] = rid
            s = offs[e["start"]][0]
            en = offs[e["end"] - 1][1]
            results.append({"id": rid, "from_name": "label", "to_name": "text",
                            "type": "labels",
                            "value": {"start": s, "end": en,
                                      "text": text[s:en], "labels": [e["type"]]}})
        for rel in r.get("relations", []):
            results.append({"from_id": eid.get(rel["head"]), "to_id": eid.get(rel["tail"]),
                            "type": "relation", "direction": "right",
                            "labels": [rel["type"]]})
        tasks.append({
            "data": {"text": text, "ident": str(r.get("ident", "")),
                     "exact_group_id": r.get("exact_group_id", ""),
                     "stratum": r.get("stratum", "")},
            "predictions": [{"model_version": "preannot", "result": results}],
        })
    return tasks


def labeling_config(schema_path="config/schema.yaml") -> str:
    schema = load_schema(schema_path)
    labels = "\n".join(
        f'    <Label value="{e}" background="{_PALETTE[i % len(_PALETTE)]}" hotkey="{i+1}"/>'
        for i, e in enumerate(entity_types(schema)))
    rels = "\n".join(f'    <Relation value="{r}"/>' for r in relation_types(schema))
    return (
        '<View>\n'
        '  <!-- AviMaint-DSS-IE v1.0 — NER + Relations labeling config for Label Studio -->\n'
        '  <Header value="Record: $ident"/>\n'
        '  <Labels name="label" toName="text" showInline="true">\n'
        f'{labels}\n'
        '  </Labels>\n'
        '  <Text name="text" value="$text"/>\n'
        '  <Relations>\n'
        f'{rels}\n'
        '  </Relations>\n'
        '</View>\n')


def _snap(offsets, start, end):
    """Snap a char span to (start_tok, end_tok) token indices."""
    s_tok = e_tok = None
    for i, (ts, te) in enumerate(offsets):
        if ts <= start < te or (start <= ts and te <= end):
            s_tok = i if s_tok is None else s_tok
        if ts < end <= te or (start <= ts and te <= end):
            e_tok = i
    if s_tok is None:
        s_tok = next((i for i, (ts, te) in enumerate(offsets) if te > start), 0)
    if e_tok is None:
        e_tok = next((i for i, (ts, te) in enumerate(offsets) if ts >= end), len(offsets) - 1)
    return s_tok, max(e_tok, s_tok) + 1


def from_export(ls_json) -> List[dict]:
    """Parse a Label Studio export (full JSON list of tasks) into internal records."""
    if isinstance(ls_json, str):
        ls_json = json.load(open(ls_json, encoding="utf-8"))
    out = []
    for task in ls_json:
        data = task.get("data", {})
        text = data.get("text", "")
        toks = text.split(" ")
        offs = char_offsets(toks)
        anns = task.get("annotations") or task.get("completions") or []
        result = anns[0]["result"] if anns else []
        regions = [r for r in result if r.get("type") == "labels"]
        rels = [r for r in result if r.get("type") == "relation"]
        entities, id2idx = [], {}
        for r in regions:
            v = r["value"]
            s_tok, e_tok = _snap(offs, v["start"], v["end"])
            id2idx[r.get("id")] = len(entities)
            entities.append({"type": v["labels"][0], "start": s_tok, "end": e_tok})
        relations = []
        for r in rels:
            h, t = id2idx.get(r.get("from_id")), id2idx.get(r.get("to_id"))
            if h is not None and t is not None and r.get("labels"):
                relations.append({"type": r["labels"][0], "head": h, "tail": t})
        bio = ["O"] * len(toks)
        for e in entities:
            bio[e["start"]] = "B-" + e["type"]
            for i in range(e["start"] + 1, e["end"]):
                bio[i] = "I-" + e["type"]
        out.append({"ident": str(data.get("ident", "")), "tokens": toks, "bio": bio,
                    "entities": entities, "relations": relations,
                    "exact_group_id": data.get("exact_group_id", "")})
    return out

