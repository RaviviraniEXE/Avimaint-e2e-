"""Audit a corrected gold JSONL for structural errors and likely annotation mistakes.

  python scripts/audit_gold.py <file.jsonl>

STRUCTURAL (hard errors): invalid entity/relation types, out-of-range spans,
overlapping spans, BIO<->entity mismatch, relation index errors, and relation
type-constraint violations (head/tail types not allowed by the schema).

QUALITY (review flags): high-precision missed cues, state/process and
action/outcome type swaps, and under-linking (entities that usually get a
relation but have none).
"""
import _bootstrap  # noqa: F401
import json
import sys
from collections import Counter

from src.data import lexicons as L
from src.models.crf_ner import bio_to_entities
from src.schema import allowed_pair, entity_types, load_schema, relation_types

PROC = set(L.ABN_PROC) | {"leaking", "vibrating", "seeping", "sputtering", "chafing", "overheating"}
FAULTW = set(L.FAULTS)
ITEMS = L.SINGLE["MAINT_ITEM"]
ACTS = L.SINGLE["ACTION"]


def audit_record(schema, r, ETYPES, RTYPES):
    errs, flags = [], []
    toks, ents, rels = r["tokens"], r.get("entities", []), r.get("relations", [])
    n = len(toks)
    ident = r.get("ident", "?")

    # --- structural: spans ---
    occ = []
    for e in ents:
        if e["type"] not in ETYPES:
            errs.append(f"invalid entity type '{e['type']}'")
        if not (0 <= e["start"] < e["end"] <= n):
            errs.append(f"bad span {e} (n={n})")
            continue
        for (s2, e2) in occ:
            if not (e["end"] <= s2 or e["start"] >= e2):
                errs.append(f"overlapping spans: {' '.join(toks[e['start']:e['end']])!r}")
        occ.append((e["start"], e["end"]))

    # --- structural: BIO <-> entities ---
    if "bio" in r and len(r["bio"]) == n:
        from_bio = {(e["start"], e["end"], e["type"]) for e in bio_to_entities(toks, r["bio"])}
        from_ent = {(e["start"], e["end"], e["type"]) for e in ents}
        if from_bio != from_ent:
            errs.append("BIO tags and entities list disagree")

    # --- structural: relations ---
    for rel in rels:
        if rel["type"] not in RTYPES:
            errs.append(f"invalid relation type '{rel['type']}'")
            continue
        if not (0 <= rel["head"] < len(ents)) or not (0 <= rel["tail"] < len(ents)):
            errs.append(f"relation index out of range: {rel}")
            continue
        ht, tt = ents[rel["head"]]["type"], ents[rel["tail"]]["type"]
        if not allowed_pair(schema, rel["type"], ht, tt):
            errs.append(f"{rel['type']} not allowed for {ht}->{tt}")

    # --- quality: state/process & action/outcome swaps ---
    for e in ents:
        txt = " ".join(toks[e["start"]:e["end"]]).lower()
        head = toks[e["start"]].lower()
        if e["type"] == "FAULT" and head in PROC:
            flags.append(f"'{txt}' typed FAULT but looks like ABN_PROC (ongoing behaviour)")
        if e["type"] == "ABN_PROC" and head in FAULTW:
            flags.append(f"'{txt}' typed ABN_PROC but looks like FAULT (static condition)")
        if head in ("good", "serviceable") and e["type"] != "OUTCOME":
            flags.append(f"'{txt}' typed {e['type']} but 'good/serviceable' is usually OUTCOME")

    # --- quality: high-precision missed cues (token O but strong cue) ---
    covered = set()
    for e in ents:
        covered.update(range(e["start"], e["end"]))
    for i, t in enumerate(toks):
        if i in covered:
            continue
        tl = t.lower()
        if tl in PROC:
            flags.append(f"'{t}' (tok {i}) untagged — likely ABN_PROC")
        elif tl in FAULTW:
            flags.append(f"'{t}' (tok {i}) untagged — likely FAULT")
        elif tl in ITEMS:
            flags.append(f"'{t}' (tok {i}) untagged — likely MAINT_ITEM")
        elif tl in ACTS:
            flags.append(f"'{t}' (tok {i}) untagged — likely ACTION")

    # --- quality: under-linking ---
    types = Counter(e["type"] for e in ents)
    have = {(rel["head"], rel["tail"], rel["type"]) for rel in rels}
    if types["ACTION"] and types["MAINT_ITEM"] and not any(r["type"] == "ACTION_ON_ITEM" for r in rels):
        flags.append("has ACTION + MAINT_ITEM but no ACTION_ON_ITEM relation")
    if (types["FAULT"] or types["ABN_PROC"]) and types["MAINT_ITEM"] and not any(r["type"] == "ISSUE_ON_ITEM" for r in rels):
        flags.append("has issue + MAINT_ITEM but no ISSUE_ON_ITEM relation")
    if types["LOC"] and types["MAINT_ITEM"] and not any(r["type"] == "HAS_LOCATION" for r in rels):
        flags.append("has LOC + MAINT_ITEM but no HAS_LOCATION relation")

    return errs, flags


def main():
    path = sys.argv[1]
    schema = load_schema()
    ET, RT = set(entity_types(schema)), set(relation_types(schema))
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    tot_e = tot_f = 0
    n_rel = sum(len(r.get("relations", [])) for r in recs)
    no_rel = sum(1 for r in recs if not r.get("relations"))
    print(f"Auditing {len(recs)} records — {sum(len(r.get('entities',[])) for r in recs)} entities, {n_rel} relations")
    print(f"records with NO relations: {no_rel}/{len(recs)}\n")

    for r in recs:
        errs, flags = audit_record(schema, r, ET, RT)
        tot_e += len(errs); tot_f += len(flags)
        if errs or flags:
            print(f"── IDENT {r.get('ident','?')} : {' '.join(r['tokens'])[:80]}")
            for e in errs:
                print(f"    ERROR  {e}")
            for f in flags:
                print(f"    flag   {f}")
    print(f"\nSUMMARY: {tot_e} structural errors, {tot_f} quality flags across {len(recs)} records")


if __name__ == "__main__":
    main()

