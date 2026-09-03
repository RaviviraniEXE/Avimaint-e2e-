"""Read-only presentation payloads for the Phase-5 frontend.

These functions adapt existing corpus analytics and planning helpers to JSON.
They do not perform retrieval, select action families, change evidence tiers,
or write to the corpus/frozen research outputs.
"""
from __future__ import annotations

import collections
from dataclasses import asdict

import pandas as pd

from . import insights as I
from . import watchlist as W


def insights_payload(
    df: pd.DataFrame,
    *,
    component: str = "",
    recurring_min: int = 5,
    top_components: int = 15,
    top_faults: int = 15,
) -> dict:
    """Return the count-only data used by the Insights page."""
    components = I.component_frequency(df, top_components)
    faults = I.fault_frequency(df, top_faults)
    matrix = I.component_fault_matrix(
        df,
        top_c=min(top_components, 12),
        top_f=min(top_faults, 12),
    )
    available_components = I.component_frequency(df, 1000)["component"].tolist()
    selected = str(component or "").strip()
    if selected and selected not in available_components:
        selected = ""

    return {
        "recurring": I.recurring_watchlist(df, recurring_min, 40).to_dict("records"),
        "components": components.to_dict("records"),
        "faults": faults.to_dict("records"),
        "actions": I.action_frequency(df).head(20).to_dict("records"),
        "outcomes": I.outcome_mix(df).to_dict("records"),
        "matrix": {
            "components": [str(value) for value in matrix.index.tolist()],
            "faults": [str(value) for value in matrix.columns.tolist()],
            "values": [[int(value) for value in row] for row in matrix.to_numpy().tolist()],
        },
        "component_options": [str(value) for value in available_components[:100]],
        "selected_component": selected,
        "component_actions": I.problem_to_action(
            df,
            by="component",
            key=selected or None,
            n=12,
        ).to_dict("records"),
        "note": "Observed work-order occurrence counts; not failure or reliability rates.",
    }


def _counter(df: pd.DataFrame, column: str) -> collections.Counter:
    return collections.Counter(
        value
        for values in df[column]
        for value in values
        if value and value != "(unspecified)"
    )


def knowledge_graph_payload(
    df: pd.DataFrame,
    *,
    top_components: int = 10,
    top_faults: int = 8,
    min_edge: int = 3,
    focus_component: str = "",
) -> dict:
    """Build a compact Component -> Fault -> Action graph as JSON."""
    component_frequency = _counter(df, "components")
    fault_frequency = _counter(df, "faults")
    focus_requested = str(focus_component or "").strip()
    focus = next(
        (
            value
            for value in component_frequency
            if value.casefold() == focus_requested.casefold()
        ),
        "",
    )

    if focus:
        top_component_names = {focus}
        focus_mask = df["components"].map(lambda values: focus in values)
        focus_fault_frequency = _counter(df[focus_mask], "faults")
        top_fault_names = {
            value for value, _ in focus_fault_frequency.most_common(max(top_faults, 12))
        }
        edge_floor = 1
    else:
        top_component_names = {
            value for value, _ in component_frequency.most_common(top_components)
        }
        top_fault_names = {value for value, _ in fault_frequency.most_common(top_faults)}
        edge_floor = int(min_edge)

    component_fault = collections.Counter()
    fault_action = collections.Counter()
    action_frequency = collections.Counter()
    for row in df.itertuples(index=False):
        row_components = set(row.components)
        row_faults = set(row.faults)
        components = row_components & top_component_names
        faults = row_faults & top_fault_names
        for component_name in components:
            for fault_name in faults:
                component_fault[(component_name, fault_name)] += 1

        action_family = str(row.action_family or "")
        if not action_family or action_family == "Other":
            continue
        if focus and focus not in row_components:
            continue
        for fault_name in faults:
            fault_action[(fault_name, action_family)] += 1
            action_frequency[action_family] += 1

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    used_faults: set[str] = set()
    for (component_name, fault_name), count in component_fault.items():
        if count < edge_floor:
            continue
        component_id = f"component:{component_name}"
        fault_id = f"fault:{fault_name}"
        nodes[component_id] = {
            "id": component_id,
            "label": component_name,
            "kind": "component",
            "count": int(component_frequency[component_name]),
            "focused": bool(component_name == focus),
        }
        nodes[fault_id] = {
            "id": fault_id,
            "label": fault_name,
            "kind": "fault",
            "count": int(fault_frequency[fault_name]),
            "focused": False,
        }
        used_faults.add(fault_name)
        edges.append(
            {
                "source": component_id,
                "target": fault_id,
                "kind": "component_fault",
                "count": int(count),
            }
        )

    for (fault_name, action_family), count in fault_action.items():
        if count < edge_floor or fault_name not in used_faults:
            continue
        fault_id = f"fault:{fault_name}"
        action_id = f"action:{action_family}"
        nodes[action_id] = {
            "id": action_id,
            "label": action_family,
            "kind": "action",
            "count": int(action_frequency[action_family]),
            "focused": False,
        }
        edges.append(
            {
                "source": fault_id,
                "target": action_id,
                "kind": "fault_action",
                "count": int(count),
            }
        )

    kind_order = {"component": 0, "fault": 1, "action": 2}
    ordered_nodes = sorted(
        nodes.values(),
        key=lambda node: (kind_order.get(node["kind"], 9), -node["count"], node["label"]),
    )
    edges.sort(key=lambda edge: (-edge["count"], edge["source"], edge["target"]))
    return {
        "nodes": ordered_nodes,
        "edges": edges,
        "focus_component": focus,
        "component_options": [
            str(value) for value, _ in component_frequency.most_common(100)
        ],
        "parameters": {
            "top_components": int(top_components),
            "top_faults": int(top_faults),
            "min_edge": edge_floor,
        },
        "note": "Observed work-order co-occurrences; not causal relationships.",
    }


def recurring_planning_payload(
    df: pd.DataFrame,
    *,
    min_support: int = 5,
    limit: int = 40,
) -> dict:
    return {
        "items": I.recurring_watchlist(df, min_support, limit).to_dict("records"),
        "min_support": int(min_support),
        "note": "Recurring historical problem clusters for planning prioritisation.",
    }


def job_card_payload(df: pd.DataFrame, cluster_id: str) -> dict:
    card = W.job_card_for_cluster(df, str(cluster_id))
    return {
        "card": asdict(card),
        "cluster_id": str(cluster_id),
        "warning": (
            "Every step is derived from recorded historical actions. Confirm against "
            "current approved maintenance data before use; this is not authorisation."
        ),
    }
