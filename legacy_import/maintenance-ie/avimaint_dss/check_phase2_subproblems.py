from core.subproblems import decompose_structure

entities = [
    {"type": "LOC", "text": "#3"},
    {"type": "MAINT_ITEM", "text": "ROCKER COVER"},
    {"type": "ABN_PROC", "text": "LEAKING"},
    {"type": "LOC", "text": "#4"},
    {"type": "MAINT_ITEM", "text": "INTAKE GASKET"},
    {"type": "ABN_PROC", "text": "LEAKING"},
]

indexed_relations = [
    {"type": "HAS_LOCATION", "head": 1, "tail": 0, "head_type": "MAINT_ITEM", "tail_type": "LOC",
     "head_text": "ROCKER COVER", "tail_text": "#3", "score": 0.9976},
    {"type": "ISSUE_ON_ITEM", "head": 2, "tail": 1, "head_type": "ABN_PROC", "tail_type": "MAINT_ITEM",
     "head_text": "LEAKING", "tail_text": "ROCKER COVER", "score": 0.9911},
    {"type": "ISSUE_ON_ITEM", "head": 2, "tail": 4, "head_type": "ABN_PROC", "tail_type": "MAINT_ITEM",
     "head_text": "LEAKING", "tail_text": "INTAKE GASKET", "score": 0.4584},
    {"type": "HAS_LOCATION", "head": 4, "tail": 3, "head_type": "MAINT_ITEM", "tail_type": "LOC",
     "head_text": "INTAKE GASKET", "tail_text": "#4", "score": 0.9822},
    {"type": "ISSUE_ON_ITEM", "head": 5, "tail": 4, "head_type": "ABN_PROC", "tail_type": "MAINT_ITEM",
     "head_text": "LEAKING", "tail_text": "INTAKE GASKET", "score": 0.9677},
]

subs = decompose_structure(entities, indexed_relations)
assert len(subs) == 2, subs
assert subs[0].component == "rocker cover" and subs[0].location == "#3", subs
assert subs[1].component == "intake gasket" and subs[1].location == "#4", subs

# This is the relation shape returned by older core.extraction.Structure:
# endpoint indices are absent. It must STILL preserve both leaking issues.
legacy_relations = [
    {k: v for k, v in r.items() if k not in {"head", "tail"}}
    for r in indexed_relations
]
subs_legacy = decompose_structure(entities, legacy_relations)
assert len(subs_legacy) == 2, subs_legacy
assert {s.component for s in subs_legacy} == {"rocker cover", "intake gasket"}, subs_legacy

print("PHASE2_SUBPROBLEM_BINDING_V2_OK")
for s in subs:
    print(s)
