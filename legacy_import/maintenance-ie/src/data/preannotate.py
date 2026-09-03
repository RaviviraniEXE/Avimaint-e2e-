"""Weak pre-annotation: produce first-draft entity spans for human correction.

High-precision dictionary + pattern matching over the normalized text. Output is
flat, non-overlapping spans (required for the CRF/BiLSTM-CRF BIO layer). Relations
are left empty for the annotator to add using the schema's type constraints.

Emits three views per record so correction is easy:
  * SpERT-format JSON  (tokens, entities[type,start,end], relations)
  * CoNLL BIO          (token <TAB> tag)  for CRF / BiLSTM-CRF
  * human review lines (readable span list)
"""
from __future__ import annotations

import re
from typing import List, Tuple

from src.data import lexicons as L

_TOK = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+|[^\sA-Za-z0-9]")

# priority for overlap resolution (higher wins on tie)
PRIORITY = {"TECH_OBS": 6, "REFERENCE": 5, "LOC": 4, "OP_CTX": 3,
            "ABN_PROC": 2, "FAULT": 2, "OUTCOME": 2, "MAINT_ITEM": 1, "ACTION": 1}


def tokenize(text: str):
    return [(m.group(), m.start(), m.end()) for m in _TOK.finditer(text)]


def _candidates(text: str) -> List[Tuple[int, int, str]]:
    low = text.lower()
    cands = []
    # multiword dictionary phrases
    for ent, phrases in L.MULTIWORD.items():
        for ph in phrases:
            for m in re.finditer(r"\b" + re.escape(ph) + r"\b", low):
                cands.append((m.start(), m.end(), ent))
    # pattern entities (LOC / TECH_OBS / OP_CTX / REFERENCE)
    for ent, pats in L.PATTERNS.items():
        for pat, _ in pats:
            for m in re.finditer(pat, low):
                cands.append((m.start(), m.end(), ent))
    # single-token dictionaries
    for ent, words in L.SINGLE.items():
        for m in re.finditer(r"\b[a-z]+\b", low):
            if m.group() in words:
                cands.append((m.start(), m.end(), ent))
    return cands


def _select(cands):
    """Greedy non-overlapping selection: longer spans first, then priority."""
    cands = sorted(cands, key=lambda c: (-(c[1] - c[0]), -PRIORITY.get(c[2], 0)))
    taken = []
    occupied = []
    for s, e, t in cands:
        if any(not (e <= os or s >= oe) for os, oe in occupied):
            continue
        taken.append((s, e, t)); occupied.append((s, e))
    return sorted(taken)


def preannotate(text: str):
    """Return dict with tokens, bio tags, and SpERT-style entities."""
    toks = tokenize(text)
    spans = _select(_candidates(text))
    n = len(toks)
    bio = ["O"] * n
    entities = []
    for s, e, t in spans:
        tok_idx = [i for i, (_, ts, te) in enumerate(toks) if ts >= s and te <= e]
        if not tok_idx:
            continue
        entities.append({"type": t, "start": tok_idx[0], "end": tok_idx[-1] + 1})
        bio[tok_idx[0]] = f"B-{t}"
        for i in tok_idx[1:]:
            bio[i] = f"I-{t}"
    return {"tokens": [w for w, _, _ in toks], "bio": bio,
            "entities": entities, "relations": []}


def coverage(records) -> dict:
    from collections import Counter
    c = Counter()
    for r in records:
        for e in r["entities"]:
            c[e["type"]] += 1
    return dict(c)

