"""Relation-grounded compound-problem decomposition.

Works with both:
- new indexed relations (`head`/`tail` preserved), and
- legacy relation dictionaries that contain only endpoint text/type.

Rules:
- ISSUE_ON_ITEM defines issue -> component.
- HAS_LOCATION defines component -> location.
- with entity indices, each repeated issue/component mention stays distinct;
- for one issue mention with several ISSUE_ON_ITEM links, only its strongest link
  is used (suppresses weak compound cross-binding);
- with legacy relations lacking indices, the strongest relation for each
  component+issue pair is retained instead of collapsing identical issue text;
- if a legacy component surface has conflicting locations, location is left
  blank rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from . import normalize as N

ISSUE_TYPES = {"FAULT", "ABN_PROC"}


@dataclass(frozen=True)
class SubproblemSpec:
    index: int
    component: str
    component_surface: str
    location: str
    issue: str
    issue_surface: str
    issue_type: str
    query: str
    relation_score: float


def _entity(entities: list[dict], idx):
    try:
        return entities[int(idx)]
    except (TypeError, ValueError, IndexError):
        return None


def _endpoints(rel: dict, entities: list[dict]):
    hi, ti = rel.get("head"), rel.get("tail")
    h = _entity(entities, hi)
    t = _entity(entities, ti)
    if h is None:
        h = {"type": rel.get("head_type"), "text": rel.get("head_text", "")}
    if t is None:
        t = {"type": rel.get("tail_type"), "text": rel.get("tail_text", "")}
    return h, t, hi, ti


def decompose_structure(
    entities: list[dict],
    relations: list[dict],
    *,
    max_subproblems: int = 4,
) -> list[SubproblemSpec]:
    entities = list(entities or [])
    relations = list(relations or [])

    # Location by exact item occurrence when indices exist.
    loc_by_item_id: dict[str, tuple[str, float]] = {}
    # Legacy fallback may see several locations for one identical component.
    locs_by_component: dict[str, dict[str, float]] = {}

    issue_candidates: list[dict] = []

    for rel in relations:
        rtype = str(rel.get("type", ""))
        score = float(rel.get("score", 0.0) or 0.0)
        h, t, hi, ti = _endpoints(rel, entities)
        ht, tt = str(h.get("type", "")), str(t.get("type", ""))
        hs, ts = str(h.get("text", "")).strip(), str(t.get("text", "")).strip()

        if rtype == "HAS_LOCATION":
            if ht == "MAINT_ITEM" and tt == "LOC":
                item, item_idx, loc = h, hi, ts
            elif tt == "MAINT_ITEM" and ht == "LOC":
                item, item_idx, loc = t, ti, hs
            else:
                continue

            component = N.canonical_component(str(item.get("text", "")))
            loc = loc.lower()
            if not component or not loc:
                continue

            if item_idx is not None:
                key = f"idx:{item_idx}"
                prev = loc_by_item_id.get(key)
                if prev is None or score > prev[1]:
                    loc_by_item_id[key] = (loc, score)

            cmap = locs_by_component.setdefault(component, {})
            cmap[loc] = max(cmap.get(loc, 0.0), score)

        elif rtype == "ISSUE_ON_ITEM":
            if ht in ISSUE_TYPES and tt == "MAINT_ITEM":
                issue, issue_idx, item, item_idx = h, hi, t, ti
            elif tt in ISSUE_TYPES and ht == "MAINT_ITEM":
                issue, issue_idx, item, item_idx = t, ti, h, hi
            else:
                continue

            component_surface = str(item.get("text", "")).strip()
            issue_surface = str(issue.get("text", "")).strip()
            if not component_surface or not issue_surface:
                continue

            component = N.canonical_component(component_surface)
            issue_family = N.issue_family(issue_surface) or issue_surface.lower()

            issue_candidates.append({
                "issue": issue,
                "issue_idx": issue_idx,
                "item": item,
                "item_idx": item_idx,
                "component": component,
                "component_surface": component_surface,
                "issue_family": issue_family,
                "issue_surface": issue_surface,
                "score": score,
            })

    # Indexed mode: strongest component link for each explicit issue occurrence.
    # Legacy mode: strongest link for each component+issue pair. This preserves
    # two identical "LEAKING" surfaces on two different components.
    best: dict[tuple, dict] = {}
    for row in issue_candidates:
        if row["issue_idx"] is not None:
            key = ("issue_idx", int(row["issue_idx"]))
        else:
            key = (
                "legacy_pair",
                row["component"],
                str(row["issue"].get("type", "")),
                row["issue_family"],
            )
        prev = best.get(key)
        if prev is None or row["score"] > prev["score"]:
            best[key] = row

    specs = []
    seen = set()
    for row in best.values():
        component = row["component"]

        location = ""
        if row["item_idx"] is not None:
            location = loc_by_item_id.get(
                f"idx:{row['item_idx']}", ("", 0.0)
            )[0]

        if not location:
            loc_map = locs_by_component.get(component, {})
            # Only use legacy canonical-component location if unambiguous.
            if len(loc_map) == 1:
                location = next(iter(loc_map))

        item_identity = (
            f"idx:{row['item_idx']}"
            if row["item_idx"] is not None
            else component
        )
        dedup = (item_identity, location, row["issue_family"])
        if dedup in seen:
            continue
        seen.add(dedup)

        query = " ".join(
            x for x in (
                location,
                row["component_surface"],
                row["issue_surface"],
            )
            if x
        ).strip()

        specs.append({
            **row,
            "location": location,
            "query": query,
        })

    # Stable entity-order sort when indices are available.
    def sort_key(row):
        try:
            return (int(row["item_idx"]), -float(row["score"]))
        except (TypeError, ValueError):
            surface = row["component_surface"]
            for i, e in enumerate(entities):
                if (
                    str(e.get("type")) == "MAINT_ITEM"
                    and str(e.get("text", "")).strip() == surface
                ):
                    return (i, -float(row["score"]))
            return (10**9, -float(row["score"]))

    specs.sort(key=sort_key)
    specs = specs[: int(max_subproblems)]

    return [
        SubproblemSpec(
            index=i + 1,
            component=row["component"],
            component_surface=row["component_surface"],
            location=row["location"],
            issue=row["issue_family"],
            issue_surface=row["issue_surface"],
            issue_type=str(row["issue"].get("type", "")),
            query=row["query"],
            relation_score=float(row["score"]),
        )
        for i, row in enumerate(specs)
    ]
