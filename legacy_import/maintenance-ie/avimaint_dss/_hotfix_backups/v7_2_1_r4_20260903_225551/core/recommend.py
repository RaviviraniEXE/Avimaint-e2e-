"""AviMaint-DSS final evidence-grounded recommender.

This module intentionally combines the strongest parts of the repository
``maintenance-ie/avimaint_dss`` decision-support logic with the frozen RQ4/RQ5
research protocol:

Repository logic retained
-------------------------
* distinct historical action strategies instead of one generic family label;
* deterministic, source-grounded recommendation sentence composition;
* independent-cluster support rather than repeated-row support;
* negative/mixed outcomes shown separately;
* provenance through concrete historical work-order examples.

Final RQ4/RQ5 safeguards
------------------------
* query-side retrieval uses PROBLEM only;
* the RQ4 DEV-selected base mode remains ``structure``;
* candidate evidence is diversified by evidence cluster;
* action text is read only after problem-side retrieval;
* the optional cross-encoder may reorder displayed evidence, but it does NOT
  change the frozen base family selection or RQ5 calibration features;
* RQ5 uncertainty is a DEV-fitted probability of historical action-family
  agreement, never a probability of technical correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import normalize as N
from . import compose as X
from .extraction_indexed import extract_structure_indexed as extract_structure
from .retrieval import Retriever
from .strategies import build_strategies
from .subproblems import decompose_structure
from .evidence_policy import classify_evidence

# Compatibility names used by the repository's legacy exploratory evaluator.
# They are presentation/evidence thresholds only; the final thesis RQ5
# calibration is provided separately by RQ5AgreementCalibrator.
STRONG_SCORE = 0.42
MODERATE_SCORE = 0.22
MIN_SCORE = 0.06


@dataclass
class CaseEvidence:
    ident: str
    problem: str
    action: str
    action_family: str
    outcome: str
    cluster_id: str
    score: float
    text_score: float = 0.0
    structure_score: float = 0.0
    channels: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    query: str
    components: list[str]
    faults: list[str]
    badge: str
    lens: str
    headline_action: str
    headline_reason: str
    support_clusters: int

    structured_sentence: str = ""
    strategies: list = field(default_factory=list)
    recommended_cases: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    nearest_cases: list = field(default_factory=list)
    negative_evidence: list = field(default_factory=list)

    structure_source: str = "rule"
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)

    # Transparent evidence features.
    family_evidence_margin: float = 0.0
    retrieval_margin: float = 0.0
    anchor_coverage: float = 0.0
    abstain: bool = True
    evidence_family: str = ""
    base_top_score: float = 0.0
    evidence_tier: str = "none"
    evidence_note: str = ""

    # RQ5 live uncertainty.
    historical_agreement_probability: float | None = None
    calibration_source: str = ""
    reranker_used: bool = False
    candidate_split: str = "train"

    # Provenance: keep learned SpERT anchors separate from deterministic fallback.
    model_components: list[str] = field(default_factory=list)
    model_faults: list[str] = field(default_factory=list)
    derived_components: list[str] = field(default_factory=list)
    derived_faults: list[str] = field(default_factory=list)
    target_location: str = ""
    partial_structure_warning: str = ""
    model_input: str = ""
    input_adapter: str = "none"
    input_adapted: bool = False

    normalized_interpretation: str = ""
    normalization_model_input: str = ""
    normalization_method: str = "none"
    normalization_warning: str = ""
    normalization_model: str = ""

    semantic_branch_used: bool = False
    semantic_status: str = "raw_rq4_fallback"
    semantic_warning: str = ""
    semantic_entities: list = field(default_factory=list)
    semantic_relations: list = field(default_factory=list)
    semantic_components: list[str] = field(default_factory=list)
    semantic_faults: list[str] = field(default_factory=list)
    semantic_locations: list[str] = field(default_factory=list)

    rq4_entities: list = field(default_factory=list)
    rq4_relations: list = field(default_factory=list)
    rq4_components: list[str] = field(default_factory=list)
    rq4_faults: list[str] = field(default_factory=list)
    rq4_structure_source: str = "rule"
    rq4_live_validated: bool = False

    # Phase 2 compound-problem support.
    compound_detected: bool = False
    decomposition_source: str = ""
    subproblems: list = field(default_factory=list)



@dataclass
class SubproblemRecommendation:
    index: int
    title: str
    query: str
    component: str
    location: str
    issue: str
    issue_type: str
    relation_score: float
    recommendation: Recommendation


_BADGE_LABEL = {
    "strong": "Strong historical evidence",
    "moderate": "Moderate historical evidence",
    "limited": "Limited historical evidence",
    "exploratory": "Exploratory evidence",
    "abstain": "Nearest cases only",
}


def badge_label(badge: str) -> str:
    return _BADGE_LABEL.get(badge, badge)


class Recommender:
    def __init__(
        self,
        df,
        retriever: Retriever,
        spert_client=None,
        semantic_spert_client=None,
        normalizer=None,
        reranker=None,
        calibrator=None,
        query_case_adapter="ascii_uppercase",
        strong_min_clusters=3,
        moderate_min_clusters=2,
        strong_min_margin=0.08,
        moderate_min_margin=0.03,
        require_anchor_for_action=True,
        abstain_on_single_cluster=False,
        limited_min_coverage=0.50,
        enable_compound_decomposition=True,
        max_subproblems=4,
        retrieval_mode="structure",
        candidate_split="train",
    ):
        self.df = df.reset_index(drop=True)
        self.r = retriever
        self.client = spert_client
        self.semantic_client = semantic_spert_client
        self.normalizer = normalizer
        self.reranker = reranker
        self.calibrator = calibrator
        self.query_case_adapter = str(query_case_adapter or "none")

        self.strong_min_clusters = int(strong_min_clusters)
        self.moderate_min_clusters = int(moderate_min_clusters)
        self.strong_min_margin = float(strong_min_margin)
        self.moderate_min_margin = float(moderate_min_margin)
        self.require_anchor = bool(require_anchor_for_action)
        self.abstain_single = bool(abstain_on_single_cluster)
        self.limited_min_coverage = float(limited_min_coverage)
        self.enable_compound_decomposition = bool(enable_compound_decomposition)
        self.max_subproblems = int(max_subproblems)

        self.retrieval_mode = str(retrieval_mode)
        self.candidate_split = str(candidate_split)

    def _case(self, hit, display_score: float | None = None) -> CaseEvidence:
        row = self.df.iloc[hit.idx]
        score = float(hit.score if display_score is None else display_score)
        return CaseEvidence(
            ident=str(row.ident),
            problem=str(row.problem),
            action=str(row.action),
            action_family=str(row.action_family),
            outcome=str(row.outcome),
            cluster_id=str(row.cluster_id),
            score=round(score, 4),
            text_score=round(float(hit.text_sim), 4),
            structure_score=round(float(hit.struct), 4),
            channels=dict(hit.channels or {}),
        )

    def _family_evidence(self, hits):
        """Match the final RQ4 family aggregation: one score per evidence cluster."""
        fam = {}
        for hit in hits:
            row = self.df.iloc[hit.idx]
            family = str(row.action_family)
            if family == "Other" or str(row.outcome) in ("negative", "mixed"):
                continue
            meta = fam.setdefault(
                family,
                {"clusters": {}, "example_idx": int(hit.idx), "example_score": float(hit.score)},
            )
            cluster = str(row.cluster_id)
            meta["clusters"][cluster] = max(
                meta["clusters"].get(cluster, 0.0),
                float(hit.score),
            )
            if float(hit.score) > meta["example_score"]:
                meta["example_idx"] = int(hit.idx)
                meta["example_score"] = float(hit.score)

        ranked = sorted(
            fam.items(),
            key=lambda kv: (
                len(kv[1]["clusters"]),
                sum(kv[1]["clusters"].values()),
            ),
            reverse=True,
        )
        return ranked

    def _anchor_coverage(self, hits, q_components, q_faults) -> float:
        anchors = [
            *("c:" + x for x in q_components),
            *("f:" + x for x in q_faults),
        ]
        if not anchors:
            return 0.0

        found = set()
        for hit in hits[:10]:
            row = self.df.iloc[hit.idx]
            found.update("c:" + x for x in q_components if x in row.components)
            found.update("f:" + x for x in q_faults if x in row.faults)
        return float(len(found) / len(anchors))

    @staticmethod
    def _ascii_uppercase(text: str) -> str:
        """Case-only deployment adapter that preserves length and offsets.

        The frozen raw aviation corpus is predominantly uppercase while the
        matched SpERT encoder is ``bert-base-cased``. Live users naturally type
        lowercase. This adapter changes ASCII letter case only; punctuation,
        digits, abbreviations, spacing and character positions are untouched.
        """
        return str(text).translate(str.maketrans(
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ))

    @staticmethod
    def _component_covered(candidate: str, learned_components: list[str]) -> bool:
        """Suppress false fallback warnings for nested terms such as rocker/rocker cover."""
        c = N.canonical_component(candidate)
        ct = set(c.split())
        for learned in learned_components:
            m = N.canonical_component(learned)
            mt = set(m.split())
            if c == m or (ct and ct.issubset(mt)) or (mt and mt.issubset(ct)):
                return True
        return False

    @classmethod
    def _components_overlap(cls, left: list[str], right: list[str]) -> bool:
        if not left or not right:
            return True
        return any(cls._component_covered(x, right) for x in left)

    @classmethod
    def _semantic_consistency(cls, raw_st, semantic_st) -> tuple[bool, str]:
        if semantic_st is None or semantic_st.source != "spert":
            return False, "Verified normalized semantic SpERT is unavailable."
        if not semantic_st.entities:
            return False, "Verified normalized semantic SpERT returned no entities."
        raw_components=list(raw_st.components)
        semantic_components=list(semantic_st.components)
        if raw_components:
            missing=[c for c in raw_components if not cls._component_covered(c, semantic_components)]
            if missing:
                return False, ("Normalized semantic SpERT did not preserve validated raw component anchor(s): " + ", ".join(missing) + ". Diagnose used the validated raw representation.")
        return True, ""

    def _model_input(self, raw: str) -> tuple[str, str]:
        mode = self.query_case_adapter.strip().lower()
        if mode in ("ascii_uppercase", "raw_style_uppercase", "uppercase"):
            return self._ascii_uppercase(raw), "ascii_uppercase"
        if mode in ("none", "off", ""):
            return raw, "none"
        # Unknown settings fail closed to unchanged input rather than silently
        # performing a semantic transformation.
        return raw, f"unsupported:{self.query_case_adapter}"

    @staticmethod
    def _component_location(st, component: str) -> str:
        """Use learned HAS_LOCATION relation before positional location order.

        This prevents compound queries such as
        '#3 rocker cover ... and #4 intake gasket ...' from composing
        '#3 intake gasket' merely because #3 is the first LOC entity.
        """
        comp = N.canonical_component(component)
        for rel in st.relations:
            if rel.get("type") != "HAS_LOCATION":
                continue

            head_type = rel.get("head_type")
            tail_type = rel.get("tail_type")
            if head_type == "MAINT_ITEM" and tail_type == "LOC":
                item_text = rel.get("head_text", "")
                loc_text = rel.get("tail_text", "")
            elif tail_type == "MAINT_ITEM" and head_type == "LOC":
                item_text = rel.get("tail_text", "")
                loc_text = rel.get("head_text", "")
            else:
                continue

            if N.canonical_component(item_text) == comp and str(loc_text).strip():
                return str(loc_text).strip().lower()

        # Safe fallback only when there is a single location in the learned
        # structure. With multiple LOCs and no relation, do not guess.
        if len(st.locations) == 1:
            return str(st.locations[0]).strip().lower()
        return ""

    def _strategy_pool(self, base_hits, q_components, q_faults, min_fault_clusters=2):
        """Build strategies only from problem-retrieved, non-negative evidence.

        This is deliberately narrower than the old repository fallback that could
        expand to arbitrary corpus rows sharing only a component. The useful
        strategy concept is retained, while RQ4 leakage-safe retrieval remains the
        gatekeeper.
        """
        if not base_hits:
            return self.df.iloc[0:0].copy()

        idx = [h.idx for h in base_hits]
        pool = self.df.iloc[idx].copy()
        pool = pool[
            (pool["action_family"] != "Other")
            & (~pool["outcome"].isin(["negative", "mixed"]))
        ].copy()

        if q_components:
            component_match = pool["components"].map(
                lambda xs: any(c in xs for c in q_components)
            )
            if component_match.any():
                pool = pool[component_match].copy()

        if q_faults and len(pool):
            fault_match = pool["faults"].map(
                lambda xs: any(f in xs for f in q_faults)
            )
            # Keep the stronger component+fault scope when it still has enough
            # independent historical evidence; otherwise retain component scope.
            if pool.loc[fault_match, "cluster_id"].nunique() >= int(min_fault_clusters):
                pool = pool[fault_match].copy()

        return pool

    @staticmethod
    def _promote_headline_strategy(strategies, headline, abstain):
        if not strategies:
            return strategies

        ordered = sorted(
            strategies,
            key=lambda s: (
                0 if s.family == headline else 1,
                -int(s.support_clusters),
                -int(s.case_count),
            ),
        )
        for s in ordered:
            s.is_primary = bool((not abstain) and s.family == headline)
        return ordered

    def _reranked_display_hits(self, query_norm, base_hits, top_k):
        """Optional reranker affects display order only, never the frozen base decision."""
        if not (
            self.reranker
            and self.reranker.available()
            and base_hits
        ):
            return [(h, float(h.score)) for h in base_hits[:top_k]], False

        candidates = [
            (h.idx, str(self.df.iloc[h.idx].problem_norm), float(h.score))
            for h in base_hits
        ]
        order = self.reranker.rerank(query_norm, candidates)
        by_idx = {h.idx: h for h in base_hits}
        out = [
            (by_idx[idx], float(score))
            for idx, score in order
            if idx in by_idx
        ][:top_k]
        return out, True

    def recommend(self, query: str, top_k: int = 25, _decompose: bool = True) -> Recommendation:
        from .textnorm import normalize_text

        raw = str(query or "").strip()
        nq = normalize_text(raw)

        # VALIDATED RQ4/RQ5 BRANCH.
        model_input, input_adapter = self._model_input(raw)
        raw_st = extract_structure(model_input, "", self.client)
        rq4_live_validated = bool(raw_st.source == "spert")

        raw_model_components = list(raw_st.components)
        raw_model_faults = [x for x in raw_st.faults if x]

        rule_components = list(N.find_components(nq))
        rule_fault = N.issue_family(nq)
        rule_faults = [rule_fault] if rule_fault else []

        derived_components = [
            c for c in rule_components
            if not self._component_covered(c, raw_model_components)
        ]
        derived_faults = [
            f for f in rule_faults if f not in raw_model_faults
        ]

        q_components = list(raw_model_components or rule_components)
        q_faults = list(raw_model_faults or rule_faults)

        # OPERATIONAL SEMANTIC BRANCH: guarded ByT5 -> verified normalized semantic SpERT.
        norm_result = None
        semantic_st = None
        semantic_warning = ""
        if self.normalizer is not None:
            try:
                norm_result = self.normalizer.normalize(raw)
            except Exception as exc:
                semantic_warning = f"Normalization failed: {type(exc).__name__}"

        if (
            norm_result is not None
            and norm_result.accepted_for_semantic_spert
            and self.semantic_client is not None
        ):
            semantic_st = extract_structure(norm_result.normalized, "", self.semantic_client)

        semantic_ok, consistency_warning = self._semantic_consistency(raw_st, semantic_st)
        if not semantic_ok:
            semantic_warning = " ".join(x for x in (semantic_warning, consistency_warning) if x)

        primary_st = semantic_st if semantic_ok else raw_st
        display_components = list(primary_st.components or q_components)
        display_faults = [x for x in (primary_st.faults or q_faults) if x]

        primary_component = q_components[0] if q_components else ""
        target_st = raw_st
        if (
            semantic_ok
            and primary_component
            and self._component_covered(primary_component, list(semantic_st.components))
        ):
            target_st = semantic_st
        target_location = (
            self._component_location(target_st, primary_component)
            if primary_component else ""
        )

        warnings = []
        if raw_model_components and derived_components:
            warnings.append(
                "Validated raw SpERT missed additional fallback component anchor(s): "
                + ", ".join(derived_components)
                + ". They were not silently added to the frozen RQ4 decision."
            )
        if not raw_model_faults and derived_faults:
            warnings.append(
                "Validated raw SpERT did not identify an explicit issue entity; "
                "deterministic fault-family anchor(s) were used for RQ4 retrieval: "
                + ", ".join(derived_faults)
                + "."
            )
        partial_structure_warning = " ".join(warnings)

        # Final RQ4 evaluation used top 50 diversified candidates.
        base_hits = self.r.search(
            nq,
            q_components,
            q_faults,
            top_k=50,
            q_entity_types=[
                e.get("type") for e in raw_st.entities if e.get("type")
            ],
            q_relation_types=[
                r.get("type") for r in raw_st.relations if r.get("type")
            ],
            raw_query=raw,
            mode=self.retrieval_mode,
            diversify=True,
        )

        ranked = self._family_evidence(base_hits)
        evidence_family = ranked[0][0] if ranked else ""
        support = len(ranked[0][1]["clusters"]) if ranked else 0

        # RQ5 calibration features exactly match final_rq4_evaluate.py.
        base_top_score = float(base_hits[0].score) if base_hits else 0.0
        second_score = float(base_hits[1].score) if len(base_hits) > 1 else 0.0
        retrieval_margin = base_top_score - second_score

        first_family_score = (
            sum(ranked[0][1]["clusters"].values()) if ranked else 0.0
        )
        second_family_score = (
            sum(ranked[1][1]["clusters"].values()) if len(ranked) > 1 else 0.0
        )
        family_margin = first_family_score - second_family_score

        coverage = self._anchor_coverage(base_hits, q_components, q_faults)
        evidence_hits = []
        if evidence_family:
            evidence_hits = [
                h for h in base_hits
                if str(self.df.iloc[h.idx].action_family) == evidence_family
                and str(self.df.iloc[h.idx].outcome) not in ("negative", "mixed")
            ]
        family_coverage = self._anchor_coverage(evidence_hits, q_components, q_faults) if evidence_hits else 0.0
        gate_coverage = family_coverage if support == 1 else coverage
        has_anchor = bool(q_components or q_faults)

        agreement_probability = None
        calibration_source = ""
        if rq4_live_validated and evidence_family and self.calibrator and self.calibrator.available():
            agreement_probability = self.calibrator.predict(
                base_top_score,
                retrieval_margin,
                support,
            )
            calibration_source = self.calibrator.status()

        # Phase 3 operational presentation gate. It does not alter the RQ4
        # ranking or RQ5 calibration features above.
        decision = classify_evidence(
            evidence_family=evidence_family,
            support=support,
            family_margin=family_margin,
            coverage=gate_coverage,
            has_anchor=has_anchor,
            require_anchor=self.require_anchor,
            strong_min_clusters=self.strong_min_clusters,
            moderate_min_clusters=self.moderate_min_clusters,
            strong_min_margin=self.strong_min_margin,
            moderate_min_margin=self.moderate_min_margin,
            limited_min_coverage=self.limited_min_coverage,
            allow_single_cluster=(not self.abstain_single),
        )
        badge = decision.badge
        abstain = decision.abstain

        if not rq4_live_validated:
            badge = "exploratory"
            abstain = True
            agreement_probability = None
            calibration_source = ""
            semantic_warning = " ".join(x for x in (semantic_warning, "Validated TRUE-RAW SpERT was unavailable; primary recommendation was suppressed and nearest historical evidence is shown only.") if x)

        # Reranker is presentation-only. Family selection, support and RQ5
        # probability above are still from the frozen structure base.
        display_hits, reranker_used = self._reranked_display_hits(
            nq, base_hits, top_k
        )
        nearest_cases = [
            self._case(hit, display_score=display_score)
            for hit, display_score in display_hits
        ]

        negative = [
            c for c in nearest_cases
            if c.outcome in ("negative", "mixed")
        ][:3]

        strategy_pool = self._strategy_pool(
            base_hits,
            q_components,
            q_faults,
            min_fault_clusters=(1 if decision.tier == "limited" else 2),
        )
        strategy_locations = [target_location] if target_location else []
        strategies = build_strategies(
            strategy_pool,
            q_components,
            strategy_locations,
            q_faults[0] if q_faults else None,
            max_corroborated=4,
            max_single_case=3,
            corroborate_min_groups=self.moderate_min_clusters,
        )
        strategies = self._promote_headline_strategy(
            strategies, evidence_family, abstain
        )

        sentence = ""
        if evidence_family and not abstain:
            primary = next(
                (s for s in strategies if s.family == evidence_family),
                None,
            )
            if primary is not None:
                sentence = primary.sentence
            else:
                supporting_actions = [
                    str(self.df.iloc[h.idx].action)
                    for h in base_hits
                    if str(self.df.iloc[h.idx].action_family) == evidence_family
                    and str(self.df.iloc[h.idx].outcome)
                    not in ("negative", "mixed")
                ]
                sentence = X.compose_sentence(
                    evidence_family,
                    X.build_target(
                        q_components,
                        strategy_locations,
                    ),
                    q_faults[0] if q_faults else None,
                    X.cases_have_verification(supporting_actions),
                )

        alternatives = []
        for fam, meta in ranked[1:4]:
            row = self.df.iloc[meta["example_idx"]]
            alternatives.append(
                {
                    "action_family": fam,
                    "support_clusters": len(meta["clusters"]),
                    "example_action": str(row.action),
                    "example_ident": str(row.ident),
                }
            )

        recommended_cases = []
        if evidence_family:
            seen_support = set()
            for hit in base_hits:
                row = self.df.iloc[hit.idx]
                if (
                    str(row.action_family) != evidence_family
                    or str(row.outcome) in ("negative", "mixed")
                ):
                    continue
                cluster = str(row.cluster_id)
                if cluster in seen_support:
                    continue
                seen_support.add(cluster)
                recommended_cases.append(self._case(hit))
                if len(recommended_cases) >= 5:
                    break

        if evidence_family:
            prob_text = (
                f" RQ5 historical-agreement estimate={agreement_probability:.1%};"
                if agreement_probability is not None
                else ""
            )
            reason = (
                f"{support} independent retrieved problem cluster(s) support "
                f"'{evidence_family}'. Selected-family anchor coverage={gate_coverage:.0%}."
                f"{prob_text} base retrieval={self.retrieval_mode}; "
                f"live SpERT adapter={input_adapter}. "
                f"Evidence tier={decision.tier}."
            )
        else:
            reason = "No corroborated historical action family was found."

        result = Recommendation(
            query=raw,
            components=display_components,
            faults=display_faults,
            badge=badge,
            lens="problem" if base_hits else "none",
            headline_action=evidence_family if not abstain else "",
            headline_reason=reason,
            support_clusters=support,
            structured_sentence=sentence,
            strategies=strategies,
            recommended_cases=recommended_cases,
            alternatives=alternatives,
            nearest_cases=nearest_cases[:8],
            negative_evidence=negative,
            structure_source=("normalized_spert" if semantic_ok else raw_st.source),
            entities=list(primary_st.entities),
            relations=list(primary_st.relations),
            family_evidence_margin=float(family_margin),
            retrieval_margin=float(retrieval_margin),
            anchor_coverage=float(gate_coverage),
            abstain=bool(abstain),
            evidence_family=evidence_family,
            base_top_score=float(base_top_score),
            evidence_tier=decision.tier,
            evidence_note=decision.note,
            historical_agreement_probability=agreement_probability,
            calibration_source=calibration_source,
            reranker_used=reranker_used,
            candidate_split=self.candidate_split,
            model_components=list(primary_st.components),
            model_faults=[x for x in primary_st.faults if x],
            derived_components=derived_components,
            derived_faults=derived_faults,
            target_location=target_location,
            partial_structure_warning=partial_structure_warning,
            model_input=model_input,
            input_adapter=input_adapter,
            input_adapted=(model_input != raw),
            normalized_interpretation=(norm_result.normalized if norm_result is not None else raw),
            normalization_model_input=(norm_result.model_input if norm_result is not None else ""),
            normalization_method=(norm_result.method if norm_result is not None else "none"),
            normalization_warning=(norm_result.warning if norm_result is not None else ""),
            normalization_model=(norm_result.model if norm_result is not None else ""),
            semantic_branch_used=bool(semantic_ok),
            semantic_status=("verified_normalized_semantic_spert" if semantic_ok else "raw_rq4_fallback"),
            semantic_warning=semantic_warning,
            semantic_entities=(list(semantic_st.entities) if semantic_st is not None else []),
            semantic_relations=(list(semantic_st.relations) if semantic_st is not None else []),
            semantic_components=(list(semantic_st.components) if semantic_st is not None else []),
            semantic_faults=([x for x in semantic_st.faults if x] if semantic_st is not None else []),
            semantic_locations=(list(semantic_st.locations) if semantic_st is not None else []),
            rq4_entities=list(raw_st.entities),
            rq4_relations=list(raw_st.relations),
            rq4_components=list(q_components),
            rq4_faults=list(q_faults),
            rq4_structure_source=raw_st.source,
            rq4_live_validated=rq4_live_validated,
        )

        # Phase 2: decompose compound query structure and run the SAME validated
        # retrieval/evidence logic independently for each relation-grounded
        # issue/component subproblem. Action text is never used for decomposition.
        if _decompose and self.enable_compound_decomposition and primary_st.source == "spert":
            specs = decompose_structure(
                list(primary_st.entities),
                list(primary_st.relations),
                max_subproblems=self.max_subproblems,
            )
            if len(specs) >= 2:
                children = []
                for spec in specs:
                    child = self.recommend(
                        spec.query,
                        top_k=top_k,
                        _decompose=False,
                    )
                    title_parts = [
                        x for x in (
                            spec.location,
                            spec.component_surface,
                            spec.issue_surface,
                        ) if x
                    ]
                    children.append(
                        SubproblemRecommendation(
                            index=spec.index,
                            title=" · ".join(title_parts),
                            query=spec.query,
                            component=spec.component,
                            location=spec.location,
                            issue=spec.issue,
                            issue_type=spec.issue_type,
                            relation_score=spec.relation_score,
                            recommendation=child,
                        )
                    )
                result.compound_detected = True
                result.decomposition_source = (
                    "normalized_relation_graph"
                    if semantic_ok
                    else "raw_relation_graph_fallback"
                )
                result.subproblems = children

        return result
