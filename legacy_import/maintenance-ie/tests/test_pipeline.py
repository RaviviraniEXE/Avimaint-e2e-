import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dedup import add_duplicate_groups
from src.data.preannotate import preannotate
from src.evaluate import entity_scores, relation_scores
from src.models.crf_ner import bio_to_entities
from src.schema import bio_tags, candidate_relations, load_schema
import pandas as pd

schema = load_schema()


def test_schema_bio_tags():
    tags = bio_tags(schema)
    assert tags[0] == "O"
    assert "B-MAINT_ITEM" in tags and "I-ACTION" in tags
    # 8 core + 1 hybrid entity -> 1 + 2*9 = 19 tags
    assert len(tags) == 1 + 2 * 9


def test_type_constraints():
    # ISSUE_ON_ITEM only from FAULT/ABN_PROC to MAINT_ITEM
    assert "ISSUE_ON_ITEM" in candidate_relations(schema, "ABN_PROC", "MAINT_ITEM")
    assert "ISSUE_ON_ITEM" not in candidate_relations(schema, "MAINT_ITEM", "MAINT_ITEM")
    assert "ACTION_ON_ITEM" in candidate_relations(schema, "ACTION", "MAINT_ITEM")


def test_dedup():
    df = pd.DataFrame({"IDENT": ["1", "2", "3"],
                       "PROBLEM": ["a leak", "a leak", "b crack"],
                       "ACTION": ["fix", "fix", "weld"]})
    df, stats = add_duplicate_groups(df)
    assert stats["duplicate_rows"] == 1
    assert df.loc[0, "exact_group_id"] == df.loc[1, "exact_group_id"]


def test_preannotate_and_bio_roundtrip():
    pa = preannotate("number 2 cylinder gasket leaking. replaced gasket.")
    assert pa["tokens"]
    types = {e["type"] for e in pa["entities"]}
    assert "MAINT_ITEM" in types and "ABN_PROC" in types
    ents = bio_to_entities(pa["tokens"], pa["bio"])
    assert len(ents) == len(pa["entities"])  # BIO <-> spans consistent


def test_entity_scoring():
    g = [{"tokens": ["a", "b"], "entities": [{"type": "X", "start": 0, "end": 1}], "relations": []}]
    p = [{"tokens": ["a", "b"], "entities": [{"type": "X", "start": 0, "end": 1}], "relations": []}]
    assert entity_scores(g, p)["micro_f1"] == 1.0
    p2 = [{"tokens": ["a", "b"], "entities": [], "relations": []}]
    assert entity_scores(g, p2)["micro_f1"] == 0.0


def test_relation_scoring():
    ents = [{"type": "ACTION", "start": 0, "end": 1}, {"type": "MAINT_ITEM", "start": 1, "end": 2}]
    g = [{"tokens": ["x", "y"], "entities": ents, "relations": [{"type": "ACTION_ON_ITEM", "head": 0, "tail": 1}]}]
    assert relation_scores(g, g)["micro_f1"] == 1.0


def test_labelstudio_roundtrip():
    from src.data.labelstudio import from_export, to_tasks
    rec = {"ident": "1", "tokens": "number 2 cylinder gasket leaking".split(),
           "entities": [{"type": "LOC", "start": 0, "end": 2},
                        {"type": "MAINT_ITEM", "start": 2, "end": 4},
                        {"type": "ABN_PROC", "start": 4, "end": 5}],
           "relations": [{"type": "ISSUE_ON_ITEM", "head": 2, "tail": 1},
                         {"type": "HAS_LOCATION", "head": 1, "tail": 0}],
           "exact_group_id": "g1"}
    tasks = to_tasks([rec])
    export = [{"data": t["data"], "annotations": [{"result": t["predictions"][0]["result"]}]}
              for t in tasks]
    back = from_export(export)[0]
    assert back["tokens"] == rec["tokens"]
    assert {(e["start"], e["end"], e["type"]) for e in back["entities"]} == \
           {(e["start"], e["end"], e["type"]) for e in rec["entities"]}
    assert len(back["relations"]) == 2

