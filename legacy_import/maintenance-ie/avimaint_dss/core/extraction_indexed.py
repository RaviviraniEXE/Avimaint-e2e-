"""SpERT extraction wrapper that preserves relation endpoint indices.

The existing core.extraction.Structure representation keeps relation text/type,
but older versions discard the original `head`/`tail` entity indices.  Compound
queries can contain repeated surfaces such as two separate `LEAKING` entities.
Without endpoint identity those mentions can be accidentally collapsed.

This wrapper preserves the indices without changing any entity/relation types or
scores used by RQ4.
"""
from __future__ import annotations

from .extraction import extract_structure as _base_extract
from .extraction import spert_to_structure


def _indexed_relations(pred: dict | None) -> list[dict]:
    if not pred:
        return []
    ents = pred.get("entities", []) or []
    out = []
    for r in pred.get("relations", []) or []:
        try:
            hi = int(r["head"])
            ti = int(r["tail"])
            h = ents[hi]
            t = ents[ti]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        out.append({
            "type": r.get("type"),
            "head": hi,
            "tail": ti,
            "head_text": h.get("text", ""),
            "head_type": h.get("type"),
            "tail_text": t.get("text", ""),
            "tail_type": t.get("type"),
            "score": round(float(r.get("score", 0.0)), 3),
        })
    return out


def extract_structure_indexed(problem_text: str, action_text: str, client=None):
    if client is not None and client.health():
        pred = client.predict(problem_text)
        if pred is not None:
            action_pred = client.predict(action_text) if action_text else None
            st = spert_to_structure(pred, action_text, action_pred)
            st.relations = _indexed_relations(pred)
            return st
    return _base_extract(problem_text, action_text, client)
