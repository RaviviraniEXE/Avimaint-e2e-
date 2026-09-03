"""Deterministic tests for Phase-5 read-only presentation adapters."""
from __future__ import annotations

import pandas as pd

from core.frontend_views import (
    insights_payload,
    job_card_payload,
    knowledge_graph_payload,
    recurring_planning_payload,
)


rows = []
for idx in range(6):
    rows.append(
        {
            "ident": f"WO-{idx + 1}",
            "problem": "#2 INTAKE LEAKING",
            "problem_norm": "#2 INTAKE LEAKING",
            "action": "REPLACED INTAKE GASKET" if idx < 5 else "CHECKED CLAMPS",
            "action_family": "Replace" if idx < 5 else "Inspect",
            "outcome": "positive" if idx < 5 else "negative",
            "cluster_id": "cluster-intake",
            "component": "INTAKE",
            "fault": "LEAKING",
            "components": ["INTAKE"],
            "faults": ["LEAKING"],
            "source": "spert",
        }
    )
df = pd.DataFrame(rows)

insights = insights_payload(df, recurring_min=5)
assert insights["recurring"][0]["cluster_id"] == "cluster-intake"
assert insights["matrix"]["values"] == [[6]]
assert insights["component_actions"][0]["action_family"] == "Replace"

graph = knowledge_graph_payload(df, min_edge=1)
node_ids = {node["id"] for node in graph["nodes"]}
assert "component:INTAKE" in node_ids
assert "fault:LEAKING" in node_ids
assert "action:Replace" in node_ids
assert all(edge["count"] >= 1 for edge in graph["edges"])

recurring = recurring_planning_payload(df, min_support=5)
assert recurring["items"][0]["work_orders"] == 6

card = job_card_payload(df, "cluster-intake")["card"]
assert card["work_orders"] == 6
assert card["steps"][0]["source_idents"]
assert all("WO-" in ident for ident in card["source_idents"])

print("PHASE5_READ_ONLY_VIEWS_OK")
