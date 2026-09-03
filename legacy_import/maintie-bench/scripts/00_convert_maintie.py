"""Step 0 (MaintIE) — convert the official MaintIE gold corpus into this
pipeline's gold format, at COARSE (level-1) entity granularity, preserving
MaintIE's official 80/10/10 file-order split (860/108/108 = their published
108-text test set).

  python scripts/00_convert_maintie.py

Reads : data/raw/gold_release.json   (from github.com/nlp-tlp/maintie)
Writes: outputs/gold/maintie.jsonl   ({ident,tokens,bio,entities,relations})
        outputs/splits.json          (frozen train/dev/test by ident)

Notes
-----
- Entity type -> top level of the hierarchy (PhysicalObject/Activity/State/
  Process/Property). Relation type -> short name (hasParticipant/hasPatient ->
  hasPatient), matching MaintIE's own create_datasets.py.
- ~30% of records have OVERLAPPING (nested) entity spans. BIO cannot represent
  overlaps, so the BIO layer (CRF / BiLSTM / transformer-NER) keeps the
  earliest+longest entity per token and drops the overlapping one; the full
  entity list is preserved for span-based RE and for SpERT (which handles
  overlaps). This asymmetry is itself a benchmark finding.
"""
import _bootstrap  # noqa: F401
import json
import os

RAW = "data/raw/gold_release.json"
OUT_GOLD = "outputs/gold/maintie.jsonl"
SPLITS = "outputs/splits.json"


def coarse_entity(t):
    return t.split("/")[0]


def short_relation(t):
    return t.split("/")[1] if "/" in t else t


def make_bio(tokens, entities):
    """BIO tags; on overlap keep earliest-then-longest, drop the rest."""
    bio = ["O"] * len(tokens)
    taken = [False] * len(tokens)
    for e in sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        s, en, ty = e["start"], e["end"], e["type"]
        if any(taken[s:en]):
            continue
        bio[s] = f"B-{ty}"
        for k in range(s + 1, en):
            bio[k] = f"I-{ty}"
        for k in range(s, en):
            taken[k] = True
    return bio


def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"Missing {RAW}. Download gold_release.json from "
                         "github.com/nlp-tlp/maintie -> data/raw/")
    data = json.load(open(RAW, encoding="utf-8"))
    os.makedirs("outputs/gold", exist_ok=True)

    recs = []
    for i, item in enumerate(data):
        ents = [{"start": e["start"], "end": e["end"], "type": coarse_entity(e["type"])}
                for e in item["entities"]]
        rels = [{"head": r["head"], "tail": r["tail"], "type": short_relation(r["type"])}
                for r in item["relations"]]
        ident = f"maintie_{i:04d}"
        recs.append({"ident": ident, "exact_group_id": ident,
                     "tokens": item["tokens"], "bio": make_bio(item["tokens"], ents),
                     "entities": ents, "relations": rels})

    with open(OUT_GOLD, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    tr, dv = int(0.8 * n), int(0.9 * n)
    idents = [r["ident"] for r in recs]
    splits = {"train": idents[:tr], "dev": idents[tr:dv], "test": idents[dv:], "seed": 1337}
    json.dump(splits, open(SPLITS, "w"), indent=1)

    print(f"MaintIE -> {OUT_GOLD}: {n} records "
          f"(train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])})")
    from collections import Counter
    ec = Counter(e["type"] for r in recs for e in r["entities"])
    rc = Counter(r2["type"] for r in recs for r2 in r["relations"])
    print("entities:", dict(ec))
    print("relations:", dict(rc))
    print(f"wrote frozen split -> {SPLITS}")


if __name__ == "__main__":
    main()

