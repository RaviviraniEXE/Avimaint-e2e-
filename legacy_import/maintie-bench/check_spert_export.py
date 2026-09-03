import json
from pathlib import Path

for split in ["train", "dev", "test"]:
    path = Path("outputs/spert") / f"{split}.json"
    text = path.read_text(encoding="utf-8").strip()

    try:
        obj = json.loads(text)
        docs = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]

    n_entities = sum(len(d.get("entities", [])) for d in docs)
    n_relations = sum(len(d.get("relations", [])) for d in docs)

    overlap_docs = 0

    for doc in docs:
        entities = doc.get("entities", [])
        has_overlap = False

        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                a_tokens = set(range(a["start"], a["end"]))
                b_tokens = set(range(b["start"], b["end"]))

                if a_tokens.intersection(b_tokens):
                    has_overlap = True
                    break

            if has_overlap:
                break

        if has_overlap:
            overlap_docs += 1

    print(
        split,
        "docs=", len(docs),
        "entities=", n_entities,
        "relations=", n_relations,
        "overlap_docs=", overlap_docs,
    )