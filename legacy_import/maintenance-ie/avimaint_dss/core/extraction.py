"""Structure extraction: SpERT client (preferred) + deterministic fallback.

`extract_structure` returns a canonical, derived structure for a work order.
It never changes source text or the frozen schema; it only maps predictions (or
lexicon matches) into the comparison concepts the dashboard reasons over.

SpERT is reached over the same local HTTP service the existing system ships
(`services/spert_query_service.py`). Point `spert_url` at it (default
http://127.0.0.1:8765). If it is unreachable, the lexicon fallback is used and
clearly labelled, so the dashboard always runs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import normalize as N

# SpERT entity labels
_ITEM = {"MAINT_ITEM"}
_ISSUE = {"FAULT", "ABN_PROC"}
_LOC = {"LOC"}
_ACTION = {"ACTION"}
_OUTCOME = {"OUTCOME"}
_REF = {"REFERENCE"}


@dataclass
class Structure:
    components: list[str] = field(default_factory=list)      # canonical MAINT_ITEM
    faults: list[str] = field(default_factory=list)          # canonical issue families
    fault_surfaces: list[str] = field(default_factory=list)  # raw issue text
    locations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)      # cited manuals/procedures
    action_family: str = "Other"
    action_role: str = "corrective"
    outcome: str = "unknown"
    source: str = "rule"                                     # "spert" | "rule"
    # raw extracted structure for display / knowledge graph
    entities: list[dict] = field(default_factory=list)       # {type, text, score}
    relations: list[dict] = field(default_factory=list)      # {type, head_text, head_type, tail_text, tail_type, score}

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Deterministic lexicon extraction (no SpERT needed).
# --------------------------------------------------------------------------- #
def rule_structure(problem_text: str, action_text: str) -> Structure:
    comps = N.find_components(problem_text) or N.find_components(action_text)
    fault_surfaces: list[str] = []
    faults: list[str] = []
    fam = N.issue_family(problem_text)
    if fam:
        faults.append(fam)
        fault_surfaces.append(problem_text)
    # secondary issue families from individual keywords
    for token in (problem_text or "").lower().replace(",", " ").split():
        f = N.issue_family(token)
        if f and f not in faults:
            faults.append(f)
    family = N.action_family(action_text)
    locs = N.find_locations(problem_text)
    # lexicon-derived entities + inferred relations (no learned relations)
    entities = ([{"type": "MAINT_ITEM", "text": c, "score": 1.0} for c in comps]
                + [{"type": "FAULT", "text": f, "score": 1.0} for f in faults]
                + [{"type": "LOC", "text": l, "score": 1.0} for l in locs])
    relations = []
    if comps:
        for f in faults:
            relations.append({"type": "ISSUE_ON_ITEM", "head_text": f, "head_type": "FAULT",
                              "tail_text": comps[0], "tail_type": "MAINT_ITEM", "score": 1.0})
        for l in locs:
            relations.append({"type": "HAS_LOCATION", "head_text": comps[0], "head_type": "MAINT_ITEM",
                              "tail_text": l, "tail_type": "LOC", "score": 1.0})
    return Structure(
        components=comps, faults=faults, fault_surfaces=fault_surfaces, locations=locs,
        action_family=family, action_role=N.action_role(family),
        outcome=N.outcome_polarity(action_text), source="rule",
        entities=entities, relations=relations,
    )


# --------------------------------------------------------------------------- #
# SpERT prediction -> Structure.
# --------------------------------------------------------------------------- #
def spert_to_structure(problem_pred: dict, action_text: str,
                       action_pred: dict | None = None) -> Structure:
    """Map SpERT output {entities:[{type,text,...}], relations:[...]} to structure."""
    ents = problem_pred.get("entities", []) if problem_pred else []
    comps: list[str] = []
    faults: list[str] = []
    fault_surfaces: list[str] = []
    locs: list[str] = []
    refs: list[str] = []
    for e in ents:
        t = e.get("type")
        surf = e.get("text", "")
        if t in _ITEM:
            comps.append(N.canonical_component(surf))
        elif t in _ISSUE:
            fault_surfaces.append(surf)
            fam = N.issue_family(surf) or surf.strip().lower()
            faults.append(fam)
        elif t in _LOC:
            locs.append(surf.strip().lower())
    # references / action outcome from the action-side prediction when present
    if action_pred:
        for e in action_pred.get("entities", []):
            if e.get("type") in _REF:
                refs.append(e.get("text", "").strip())
    fam = N.action_family(action_text)
    # raw entities + relations for display / KG (resolve relation endpoints)
    entities = [{"type": e.get("type"), "text": e.get("text", ""),
                 "score": round(float(e.get("score", 0.0)), 3)} for e in ents]
    relations = []
    for r in (problem_pred.get("relations", []) if problem_pred else []):
        try:
            h, t = ents[r["head"]], ents[r["tail"]]
        except (IndexError, KeyError, TypeError):
            continue
        relations.append({"type": r.get("type"),
                          "head_text": h.get("text", ""), "head_type": h.get("type"),
                          "tail_text": t.get("text", ""), "tail_type": t.get("type"),
                          "score": round(float(r.get("score", 0.0)), 3)})
    return Structure(
        components=list(dict.fromkeys(comps)),
        faults=list(dict.fromkeys(faults)),
        fault_surfaces=fault_surfaces,
        locations=list(dict.fromkeys(locs)),
        references=list(dict.fromkeys(refs)),
        action_family=fam,
        action_role=N.action_role(fam),
        outcome=N.outcome_polarity(action_text),
        source="spert",
        entities=entities, relations=relations,
    )


# --------------------------------------------------------------------------- #
# SpERT HTTP client.
# --------------------------------------------------------------------------- #
class SpERTClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", token: str = "", timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._ok: bool | None = None

    def health(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            req = urllib.request.Request(self.url + "/health")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self._ok = r.status == 200
        except Exception:
            self._ok = False
        return self._ok

    def predict(self, text: str) -> dict | None:
        try:
            body = json.dumps({"text": text}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            req = urllib.request.Request(self.url + "/predict", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None


def extract_structure(problem_text: str, action_text: str,
                      client: SpERTClient | None = None) -> Structure:
    """Best-available structure: SpERT if the client is live, else lexicon."""
    if client is not None and client.health():
        pred = client.predict(problem_text)
        if pred is not None:
            act_pred = client.predict(action_text) if action_text else None
            return spert_to_structure(pred, action_text, act_pred)
    return rule_structure(problem_text, action_text)

