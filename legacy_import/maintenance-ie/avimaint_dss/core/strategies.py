"""Group the different recorded actions for one problem into meaningful strategies.

Semantically-similar problems often have several distinct recorded responses
(e.g. a gasket leak may be Replaced, Re-torqued, or Resealed). This module turns
that raw variation into a ranked set of STRATEGIES, each with a structured
recommendation sentence, a plain-language meaning, its support, and its recorded
outcomes — so the user sees *what the options are and what each means*, not a
list of log lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import compose as X


@dataclass
class Strategy:
    family: str
    sentence: str                 # structured imperative recommendation
    meaning: str                  # plain-language description of the strategy
    support_clusters: int         # distinct problem groups using it
    case_count: int               # total recorded work orders
    outcome_positive: int = 0
    outcome_negative: int = 0
    outcome_unknown: int = 0
    examples: list[dict] = field(default_factory=list)   # {ident, action}
    is_primary: bool = False
    tier: str = "corroborated"    # "corroborated" (>=2 groups) | "single_case"


def build_strategies(pool: pd.DataFrame, q_components: list[str], q_locations: list[str],
                     q_fault: str | None, max_corroborated: int = 4,
                     max_single_case: int = 3, corroborate_min_groups: int = 2) -> list[Strategy]:
    """`pool` = eligible cases relevant to the query (component/fault scoped).

    Returns corroborated strategies (>= corroborate_min_groups independent problem
    groups) first, then a bounded number of single-case options, each tier-labelled.
    The count is adaptive — not a fixed number.
    """
    if pool.empty:
        return []
    target = X.build_target(q_components, q_locations)
    grp = pool.groupby("action_family")
    rows = []
    for fam, sub in grp:
        if fam == "Other":
            continue
        rows.append((fam, sub))
    # rank by distinct clusters then case count
    rows.sort(key=lambda fs: (fs[1]["cluster_id"].nunique(), len(fs[1])), reverse=True)

    corroborated, single = [], []
    for fam, sub in rows:
        s = _make_strategy(fam, sub, target, q_fault)
        if s.support_clusters >= corroborate_min_groups:
            s.tier = "corroborated"
            corroborated.append(s)
        else:
            s.tier = "single_case"
            single.append(s)
    out = corroborated[:max_corroborated] + single[:max_single_case]
    if out:
        out[0].is_primary = True
    return out


def _make_strategy(fam, sub, target, q_fault) -> Strategy:
    acts = sub["action"].tolist()
    has_verif = X.cases_have_verification(acts)
    # prefer positive/known-outcome, distinct examples
    ex_sorted = sub.sort_values(
        by="outcome", key=lambda s: s.map({"positive": 0, "unknown": 1, "mixed": 2, "negative": 3}))
    seen, examples = set(), []
    for r in ex_sorted.itertuples(index=False):
        a = (getattr(r, "action", "") or "").strip()
        k = a.lower()
        if not a or k in seen:
            continue
        seen.add(k)
        examples.append({"ident": str(getattr(r, "ident")), "action": a,
                         "outcome": getattr(r, "outcome")})
        if len(examples) >= 3:
            break
    return Strategy(
        family=fam,
        sentence=X.compose_sentence(fam, target, q_fault, has_verif),
        meaning=X.family_meaning(fam),
        support_clusters=int(sub["cluster_id"].nunique()),
        case_count=int(len(sub)),
        outcome_positive=int((sub["outcome"] == "positive").sum()),
        outcome_negative=int(sub["outcome"].isin(["negative", "mixed"]).sum()),
        outcome_unknown=int((sub["outcome"] == "unknown").sum()),
        examples=examples,
    )

