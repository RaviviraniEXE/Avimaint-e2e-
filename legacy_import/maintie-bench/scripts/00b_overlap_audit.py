"""MaintIE overlap/nesting audit. NO TRAINING.

Quantifies the structural ceiling imposed by the BIO flattening policy used by
Tier1/Tier2/Tier3A while preserving full spans for span-NER/SpERT evaluation.
Writes outputs/reports/maintie_overlap_audit.json.
"""
import _bootstrap  # noqa: F401
import json
import os
from collections import Counter

from src.data.gold import load_gold
from src.data.split import assign, load_splits


def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def _kept_indices(d):
    taken = [False] * len(d["tokens"])
    kept = set()
    for idx, e in sorted(enumerate(d.get("entities", [])),
                         key=lambda x: (x[1]["start"], -(x[1]["end"] - x[1]["start"]))):
        s, en = e["start"], e["end"]
        if any(taken[s:en]):
            continue
        kept.add(idx)
        for k in range(s, en):
            taken[k] = True
    return kept


def _stats(docs):
    any_overlap_docs = 0
    total_entities = nested_entities = kept_entities = 0
    total_relations = representable_relations = 0
    nested_by_type = Counter(); dropped_by_type = Counter()
    relation_total = Counter(); relation_lost = Counter()

    for d in docs:
        es = d.get("entities", [])
        total_entities += len(es)
        nested = set()
        for i, a in enumerate(es):
            if any(i != j and _overlap(a, b) for j, b in enumerate(es)):
                nested.add(i)
        if nested:
            any_overlap_docs += 1
        nested_entities += len(nested)
        nested_by_type.update(es[i]["type"] for i in nested)

        kept = _kept_indices(d)
        kept_entities += len(kept)
        dropped_by_type.update(es[i]["type"] for i in range(len(es)) if i not in kept)

        for r in d.get("relations", []):
            total_relations += 1
            relation_total[r["type"]] += 1
            if r["head"] in kept and r["tail"] in kept:
                representable_relations += 1
            else:
                relation_lost[r["type"]] += 1

    return {
        "documents": len(docs),
        "documents_with_overlap": any_overlap_docs,
        "documents_with_overlap_pct": round(any_overlap_docs / len(docs), 6) if docs else 0,
        "entities_total": total_entities,
        "entities_nested_or_overlapping": nested_entities,
        "entities_nested_or_overlapping_pct": round(nested_entities / total_entities, 6) if total_entities else 0,
        "bio_entities_retained": kept_entities,
        "bio_entities_dropped": total_entities - kept_entities,
        "bio_entity_recall_ceiling": round(kept_entities / total_entities, 6) if total_entities else 0,
        "relations_total": total_relations,
        "bio_relations_representable": representable_relations,
        "bio_relations_unrepresentable": total_relations - representable_relations,
        "bio_relation_recall_ceiling": round(representable_relations / total_relations, 6) if total_relations else 0,
        "nested_entities_by_type": dict(nested_by_type),
        "bio_dropped_entities_by_type": dict(dropped_by_type),
        "relations_total_by_type": dict(relation_total),
        "bio_unrepresentable_relations_by_type": dict(relation_lost),
    }


def main():
    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No MaintIE gold found. Run scripts\\maintie\\01_prepare.bat first.")
    if not load_splits():
        raise SystemExit("Frozen MaintIE split missing.")
    tr, dv, te = assign(gold)
    payload = {
        "policy": "BIO keeps earliest-then-longest non-overlapping spans; full spans remain authoritative gold",
        "training_performed": False,
        "all": _stats(gold),
        "train": _stats(tr),
        "dev": _stats(dv),
        "test": _stats(te),
    }
    os.makedirs("outputs/reports", exist_ok=True)
    out = "outputs/reports/maintie_overlap_audit.json"
    json.dump(payload, open(out, "w", encoding="utf-8"), indent=2)

    print("=" * 72)
    print("  MAINTIE OVERLAP / BIO REPRESENTATION AUDIT - NO TRAINING")
    print("=" * 72)
    for name in ["all", "train", "dev", "test"]:
        s = payload[name]
        print(f"{name.upper():5} docs={s['documents']:4d}  overlap-docs={s['documents_with_overlap']:3d} "
              f"({100*s['documents_with_overlap_pct']:.2f}%)")
        print(f"      entities={s['entities_total']:4d}  BIO dropped={s['bio_entities_dropped']:3d}  "
              f"entity recall ceiling={s['bio_entity_recall_ceiling']:.4f}")
        print(f"      relations={s['relations_total']:4d} BIO unrepresentable={s['bio_relations_unrepresentable']:3d}  "
              f"relation recall ceiling={s['bio_relation_recall_ceiling']:.4f}")
    print("TEST unrepresentable relations by type:", payload["test"]["bio_unrepresentable_relations_by_type"])
    print("Saved ->", out)


if __name__ == "__main__":
    main()
