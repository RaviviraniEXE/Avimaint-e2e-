"""AviMaint-DSS — explainable maintenance planning support.

Seven pages: Overview · Diagnose · Insights · Knowledge Graph · Planning · Evaluation · Guide.
Deterministic core; SpERT predictions enhance structure when the service is on.
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import json
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import yaml

from core.corpus import load_corpus
from core.extraction import SpERTClient
from core.retrieval import Retriever
from core.recommend import Recommender, badge_label
from core.reranker import CrossEncoderReranker
from core.calibration import RQ5AgreementCalibrator
from core.query_normalization import NormalizationClient
from core import insights as I
from core import watchlist as W
from core import kg as KG
import plotly.graph_objects as go
from ui import theme as T
from ui import charts as C

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def get_runtime_lock() -> dict:
    path = ROOT / "runtime_model_lock.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "byt5": {"enabled": False, "reason": "runtime model lock unavailable"},
            "normalized_spert": {
                "enabled": False,
                "verified_representation": False,
                "reason": "runtime model lock unavailable",
            },
        }


def service_metadata(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(str(url).rstrip("/") + "/health", timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Loaders (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource(show_spinner="Loading corpus and building structure…")
def get_corpus(cfg_key: str):
    cfg = get_config()
    csv = ROOT / cfg["data"]["csv_path"]
    pred = cfg["data"].get("problem_predictions_path") or None
    protocol = cfg["data"].get("protocol_path") or None
    corpus = load_corpus(csv, mode=cfg["extraction"]["mode"],
                         spert_url=cfg["extraction"]["spert_url"],
                         predictions_path=(ROOT / pred).resolve() if pred else None,
                         protocol_path=(ROOT / protocol).resolve() if protocol else None,
                         require_predictions=cfg["extraction"].get("require_problem_predictions", False),
                         normalize=False)
    return corpus


@st.cache_resource(show_spinner="Building validated evidence index…")
def get_engine(cfg_key: str):
    cfg = get_config()
    corpus = get_corpus(cfg_key)

    # Research-validated live mode uses the same candidate population used by
    # frozen RQ4/RQ5: TRAIN evidence only. Insights/overview still use all 6,169.
    research = cfg.get("research", {})
    candidate_split = str(research.get("validated_candidate_split", "train"))
    if candidate_split:
        if "frozen_split" not in corpus.df.columns:
            raise RuntimeError("Leakage-safe evidence split is unavailable: corpus has no 'frozen_split' column.")
        candidate_df = corpus.df[corpus.df["frozen_split"] == candidate_split].copy()
        if candidate_df.empty:
            raise RuntimeError(f"Leakage-safe evidence split '{candidate_split}' contains zero rows.")
    else:
        raise RuntimeError("research.validated_candidate_split must be configured; refusing corpus-wide evidence fallback.")

    dense = cfg["retrieval"].get("dense_model") or None
    retr = Retriever(
        candidate_df,
        weights=cfg["retrieval"].get("weights"),
        dense_model=dense,
        rrf_k=cfg["retrieval"].get("rrf_k", 60),
    )

    client = None
    if cfg["extraction"]["mode"] == "spert":
        raw_url = cfg["extraction"]["spert_url"]
        raw_meta = service_metadata(raw_url)
        raw_ok = bool(
            raw_meta and raw_meta.get("status") == "ready"
            and int(raw_meta.get("entity_types", 0)) == 9
            and int(raw_meta.get("relation_types", 0)) == 11
            and raw_meta.get("query_case_normalization") == "none_true_raw"
        )
        c = SpERTClient(raw_url)
        client = c if raw_ok and c.health() else None

    runtime_lock = get_runtime_lock()

    semantic_client = None
    scfg = cfg.get("semantic_extraction", {})
    semantic_locked = runtime_lock.get("normalized_spert", {})
    if (
        scfg.get("enabled", True)
        and semantic_locked.get("enabled", False)
        and semantic_locked.get("verified_representation", False)
    ):
        semantic_url = scfg.get("spert_url", "http://127.0.0.1:8767")
        semantic_meta = service_metadata(semantic_url)
        semantic_ok = bool(
            semantic_meta and semantic_meta.get("status") == "ready"
            and semantic_meta.get("role") == "rules_then_byt5_semantic_spert"
            and semantic_meta.get("representation") == "rules_then_byt5_guarded_operational"
            and semantic_meta.get("weights_sha256") == semantic_locked.get("weight_sha256")
        )
        sc = SpERTClient(semantic_url)
        semantic_client = sc if semantic_ok and sc.health() else None

    rr_model = cfg["retrieval"].get("reranker_model") or None
    reranker = (
        CrossEncoderReranker(
            rr_model,
            cfg["retrieval"].get("reranker_blend", 0.7),
        )
        if rr_model else None
    )

    cal_rel = research.get("rq5_dev_predictions_path")
    calibrator = RQ5AgreementCalibrator(
        (ROOT / cal_rel).resolve() if cal_rel else None
    )

    ncfg = cfg.get("normalization", {})
    byt5_locked = runtime_lock.get("byt5", {})
    live_normalizer = NormalizationClient(
        url=ncfg.get("service_url", "http://127.0.0.1:8766"),
        timeout=float(ncfg.get("timeout_seconds", 30)),
        enabled=bool(ncfg.get("enabled", True) and byt5_locked.get("enabled", False)),
    )

    rcfg = cfg.get("recommender", {})
    rec = Recommender(
        candidate_df,
        retr,
        spert_client=client,
        semantic_spert_client=semantic_client,
        normalizer=live_normalizer,
        reranker=reranker,
        calibrator=calibrator,
        query_case_adapter=cfg["extraction"].get("live_query_case_adapter", "ascii_uppercase"),
        strong_min_clusters=rcfg.get("strong_min_clusters", 3),
        moderate_min_clusters=rcfg.get("moderate_min_clusters", 2),
        strong_min_margin=rcfg.get("strong_min_margin", 0.08),
        moderate_min_margin=rcfg.get("moderate_min_margin", 0.03),
        require_anchor_for_action=rcfg.get("require_anchor_for_action", True),
        abstain_on_single_cluster=rcfg.get("abstain_on_single_cluster", False),
        limited_min_coverage=rcfg.get("limited_min_coverage", 0.50),
        enable_compound_decomposition=cfg.get("phase2", {}).get("enabled", True),
        max_subproblems=cfg.get("phase2", {}).get("max_subproblems", 4),
        retrieval_mode=cfg["retrieval"].get("default_mode", "structure"),
        candidate_split=candidate_split,
    )
    return retr, rec

@st.cache_data(show_spinner=False)
def get_frozen_evaluation() -> dict:
    """Read the immutable final RQ4/RQ5/manual-review summaries for presentation."""
    frozen_root = REPO / "outputs" / "frozen" / "final_rq4_rq5"
    paths = {
        "rq4": frozen_root / "rq4_case_retrieval" / "test" / "RQ4_FINAL_TEST.json",
        "rq5": frozen_root / "rq5_planning_support" / "RQ5_PLANNING_SUPPORT.json",
        "manual": frozen_root / "rq4_case_retrieval" / "manual_review" / "MANUAL_REVIEW_RESULTS.json",
        "lock": frozen_root / "rq4_case_retrieval" / "FINAL_TEST_LOCK.json",
    }
    out = {}
    for key, path in paths.items():
        try:
            out[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            out[key] = None
    return out


# ---- cached heavy computes (keyed by config signature) -------------------- #
@st.cache_data(show_spinner=False)
def c_kpis(key, recurring_min):
    return I.kpis(get_corpus(key).df, recurring_min)

@st.cache_data(show_spinner=False)
def c_recurring(key, min_support, n):
    return I.recurring_watchlist(get_corpus(key).df, min_support, n)

@st.cache_data(show_spinner=False)
def c_top_recurring(key, n):
    return I.top_recurring_problems(get_corpus(key).df, n)

@st.cache_data(show_spinner=False)
def c_component_freq(key, n):
    return I.component_frequency(get_corpus(key).df, n)

@st.cache_data(show_spinner=False)
def c_fault_freq(key, n):
    return I.fault_frequency(get_corpus(key).df, n)

@st.cache_data(show_spinner=False)
def c_action_freq(key):
    return I.action_frequency(get_corpus(key).df)

@st.cache_data(show_spinner=False)
def c_matrix(key, tc, tf):
    return I.component_fault_matrix(get_corpus(key).df, tc, tf)

@st.cache_data(show_spinner=False)
def c_outcome(key):
    return I.outcome_mix(get_corpus(key).df)

@st.cache_data(show_spinner=False)
def c_p2a(key, comp):
    return I.problem_to_action(get_corpus(key).df, "component", comp)

@st.cache_data(show_spinner=False)
def c_kg_html(key, tc, tf, min_edge, focus):
    return KG.corpus_graph(get_corpus(key).df, tc, tf, min_edge, focus or None)

@st.cache_data(show_spinner=False)
def c_component_list(key, n):
    return I.component_frequency(get_corpus(key).df, n)["component"].tolist()


def cfg_signature(cfg: dict) -> str:
    e = cfg["extraction"]
    r = cfg.get("research", {})
    n = cfg.get("normalization", {})
    s = cfg.get("semantic_extraction", {})
    return (f'{e["mode"]}|{e.get("normalize", True)}|{e.get("live_query_case_adapter", "none")}|{cfg["data"].get("csv_path")}'
            f'|{cfg["data"].get("problem_predictions_path")}|{cfg["retrieval"].get("dense_model")}'
            f'|{cfg["retrieval"].get("reranker_model")}|'
            f'{cfg["retrieval"].get("default_mode", "structure")}|'
            f'{r.get("validated_candidate_split", "train")}|'
            f'{r.get("rq5_dev_predictions_path", "")}|'
            f'{n.get("enabled", True)}|{n.get("mode", "")}|{n.get("service_url", "")}|'
            f'{s.get("enabled", True)}|{s.get("mode", "")}|{s.get("spert_url", "")}')


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(key: str, cfg: dict):
    st.markdown("### Overview")
    st.markdown('<div class="muted">The shape of your maintenance corpus at a glance. '
                'All figures are observed work-order counts — never failure rates.</div>',
                unsafe_allow_html=True)
    k = c_kpis(key, cfg["insights"]["recurring_min"])
    T.kpi_row([
        ("Work orders", f'{k["work_orders"]:,}', ""),
        ("Unique problems", f'{k["unique_problems"]:,}', ""),
        ("Problem clusters", f'{k["problem_clusters"]:,}', ""),
        ("Components tracked", f'{k["components_tracked"]:,}', ""),
        ("Recurring faults", f'{k["recurring_faults"]:,}', f'≥{cfg["insights"]["recurring_min"]} work orders'),
        ("Recorded outcomes", f'{k["recorded_outcomes_pct"]:.0f}%', "rest unknown"),
    ])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        T.section("Recurring problems (chronic faults)",
                  "The problems your fleet keeps seeing — where a standing plan pays off.")
        rec = c_top_recurring(key, 12)
        st.dataframe(rec.rename(columns={
            "problem": "Problem", "work_orders": "Work orders", "component": "Component",
            "fault": "Fault", "top_action": "Usual action"}).drop(columns=["cluster_id"]),
            hide_index=True, use_container_width=True, height=460)
    with c2:
        T.section("Most-involved components", "What shows up most in problems.")
        st.plotly_chart(C.hbar(c_component_freq(key, 10), "component", "work_orders"),
                        use_container_width=True, config={"displayModeBar": False})
        T.section("What you do most", "Recorded action families.")
        st.plotly_chart(C.action_bars(c_action_freq(key), "action_family", "work_orders", 260),
                        use_container_width=True, config={"displayModeBar": False})


def _render_cases(cases, key_prefix=""):
    for cse in cases:
        st.markdown(
            f'<div class="card"><span class="caseid">{cse.ident}</span> '
            f'<span class="pill">{cse.action_family}</span> '
            f'<span class="pill">outcome: {cse.outcome}</span><br>'
            f'<div style="margin-top:6px"><b>Problem:</b> {cse.problem}</div>'
            f'<div><b>Recorded action:</b> {cse.action}</div></div>',
            unsafe_allow_html=True)


def _render_strategy(s):
    star = '<span class="star">★ primary</span>' if s.is_primary else ""
    weak = ('<span class="pill" style="color:#eda100">single-case · weak evidence</span>'
            if s.tier == "single_case" else "")
    out = (f'<span class="pill" style="color:#0ca30c">✔ {s.outcome_positive} positive</span>'
           if s.outcome_positive else "")
    neg = (f'<span class="pill" style="color:#e66767">✕ {s.outcome_negative} neg</span>'
           if s.outcome_negative else "")
    ex = ""
    if s.examples:
        e = s.examples[0]
        ex = (f'<div class="muted" style="margin-top:6px">e.g. '
              f'<span class="caseid">{e["ident"]}</span> {e["action"]}</div>')
    cls = "primary" if s.is_primary else ("weak" if s.tier == "single_case" else "")
    st.markdown(
        f'<div class="strat {cls}">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'<span class="pill" style="font-weight:700">{s.family}</span>{star}{weak}'
        f'<span class="pill">{s.case_count} work orders · {s.support_clusters} groups</span>'
        f'{out}{neg}</div>'
        f'<div class="reco2">➜ {s.sentence}</div>'
        f'<div class="muted">{s.meaning}</div>{ex}</div>',
        unsafe_allow_html=True)


def _render_structure_payload(entities, relations, title: str, source: str, graph_label: str):
    if not entities and not relations:
        st.info(f"{title}: no structured prediction is available.")
        return
    st.markdown(f"#### {title}")
    chips = "".join(
        f'<span class="ent" style="background:{KG.TYPE_COLOR.get(e.get("type"),"#5a5a55")}22;'
        f'border:1px solid {KG.TYPE_COLOR.get(e.get("type"),"#5a5a55")}">'
        f'<b>{e.get("text","")}</b> <span class="etype">{e.get("type")}</span></span>'
        for e in entities
    )
    st.markdown(f'<div class="entrow">{chips}</div>', unsafe_allow_html=True)
    if relations:
        rels = "".join(
            f'<div class="rel"><span class="rt">{r.get("head_text")}</span>'
            f'<span class="rarrow"> —[{r.get("type")}]→ </span>'
            f'<span class="rt">{r.get("tail_text")}</span></div>'
            for r in relations
        )
        st.markdown(f'<div class="relbox">{rels}</div>', unsafe_allow_html=True)
    with st.expander(f"🕸 {graph_label} ({source})"):
        components.html(KG.query_graph(entities, relations), height=440, scrolling=False)


def _render_extraction(R):
    source = (
        "Expert rules → guarded ByT5 → matched semantic SpERT"
        if R.semantic_branch_used
        else "validated raw-matched SpERT fallback"
    )
    _render_structure_payload(
        R.entities, R.relations, "Primary structured interpretation", source,
        "Problem knowledge graph"
    )


def _render_rq4_extraction(R):
    _render_structure_payload(
        R.rq4_entities, R.rq4_relations,
        "Validated RQ4 retrieval representation",
        "true-raw matched SpERT", "RQ4 query graph"
    )



def _render_subproblem_result(sp, cfg: dict):
    r = sp.recommendation
    prob = r.historical_agreement_probability
    prob_text = f"{100 * prob:.1f}%" if prob is not None else "n/a"

    st.markdown(
        f'<div class="card" style="padding:14px 16px;margin:12px 0 8px">'
        f'<div class="muted" style="font-size:.82rem;font-weight:700">'
        f'ISSUE {sp.index}</div>'
        f'<div style="font-size:1.08rem;font-weight:750;margin-top:2px">'
        f'{sp.title}</div>'
        f'<div class="muted" style="margin-top:4px">'
        f'Relation-grounded subquery: {sp.query}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if r.structured_sentence:
        st.markdown(
            f'<div class="reco">➜ {r.structured_sentence}</div>',
            unsafe_allow_html=True,
        )
        if r.badge == "limited":
            st.warning(
                "Limited historical evidence: only one independent historical "
                "problem cluster supports this action family. It is shown as a "
                "traceable example, not as a recurring maintenance strategy."
            )
    else:
        fam = r.evidence_family or "none"
        st.info(
            "No primary action passed the current evidence gate for this issue. "
            f"Closest historical action family: {fam}."
        )

    st.markdown(
        '<div class="card" style="padding:9px 12px;margin:5px 0 10px">'
        '<div style="display:flex;gap:7px;align-items:center;flex-wrap:wrap">'
        f'<span class="pill"><b>Evidence</b> · {badge_label(r.badge)}</span>'
        f'<span class="pill"><b>Historical agreement</b> · {prob_text}</span>'
        f'<span class="pill"><b>Support</b> · {r.support_clusters} clusters</span>'
        f'<span class="pill"><b>Anchor coverage</b> · {100*r.anchor_coverage:.0f}%</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Distinct historical strategies are always visible, even if the current
    # primary evidence gate abstains. This does not turn weak evidence into an
    # approved recommendation; each strategy keeps its support tier.
    if r.strategies:
        st.markdown("**Recorded strategy alternatives**")
        shown_single = False
        for strategy in r.strategies:
            if strategy.tier == "single_case" and not shown_single:
                st.caption("Single-case historical options — weak evidence, review only.")
                shown_single = True
            _render_strategy(strategy)

    if r.recommended_cases:
        with st.expander(f"Supporting work orders · issue {sp.index}"):
            _render_cases(r.recommended_cases, key_prefix=f"sp{sp.index}_support")

    with st.expander(f"Nearest historical cases · issue {sp.index}"):
        _render_cases(r.nearest_cases, key_prefix=f"sp{sp.index}_nearest")

    if r.negative_evidence:
        with st.expander(f"Negative / unresolved evidence · issue {sp.index}"):
            _render_cases(r.negative_evidence, key_prefix=f"sp{sp.index}_negative")

    if r.structured_sentence and st.button(
        f"➕ Add issue {sp.index} to planning watchlist",
        key=f"add_subproblem_{sp.index}_{hash(sp.query)}",
    ):
        st.session_state.setdefault("plan_queue", [])
        st.session_state["plan_queue"].append(
            {
                "query": sp.query,
                "recommendation": r.structured_sentence,
                "action_family": r.headline_action,
                "component": sp.component,
                "evidence_grade": badge_label(r.badge),
                "rq5_historical_agreement": (
                    round(float(r.historical_agreement_probability), 4)
                    if r.historical_agreement_probability is not None
                    else None
                ),
                "support_clusters": r.support_clusters,
                "anchor_coverage": round(float(r.anchor_coverage), 4),
                "source": "phase2_relation_grounded_subproblem",
            }
        )
        st.success(f"Issue {sp.index} added to planning watchlist.")


def _render_compound_plan(R, cfg: dict):
    st.markdown("#### Multi-problem planning support")
    st.markdown(
        f'<div class="card" style="padding:12px 14px;margin-bottom:10px">'
        f'<b>{len(R.subproblems)} relation-grounded maintenance issues detected.</b>'
        f'<div class="muted" style="margin-top:5px">'
        f'Each issue is retrieved and ranked independently. Decomposition source: '
        f'{R.decomposition_source}. No synthetic combined maintenance action is invented.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    for sp in R.subproblems:
        _render_subproblem_result(sp, cfg)

    st.caption(
        "Per-issue RQ5 values estimate historical action-family agreement for each "
        "operational subquery. They are not probabilities of technical correctness "
        "and the compound decomposition itself was not the frozen RQ4 TEST target."
    )


def page_diagnose(rec_engine: Recommender, cfg: dict):
    # Keep the original AviMaint-DSS Diagnose interaction/layout from the repo:
    # input -> evidence badge/anchors -> extraction -> recommendation -> strategies
    # -> negative evidence -> nearest cases. RQ4/RQ5 are added as compact evidence
    # metadata rather than redesigning the page around research metrics.
    st.markdown("### Diagnose")
    st.markdown(
        '<div class="muted">Type a problem and press Enter. The system retrieves the closest '
        'historical work orders and grades its evidence — it never hides the supporting cases.</div>',
        unsafe_allow_html=True,
    )

    q = st.text_input(
        "Problem description",
        value="#2 intake gasket leaking",
        placeholder="e.g. left magneto rpm drop on run up",
        label_visibility="collapsed",
    )
    if not q.strip():
        st.info("Enter a problem description above to get historical planning support.")
        return

    R = rec_engine.recommend(q, top_k=cfg["recommender"]["top_k"])

    # Evidence badge remains from the validated RQ4/RQ5 branch.
    st.markdown(T.badge_html(R.badge, R.lens), unsafe_allow_html=True)

    normalized = R.normalized_interpretation or R.query
    semantic_label = (
        "verified hybrid semantic SpERT active"
        if R.semantic_branch_used
        else "safe raw-SpERT fallback"
    )
    st.markdown(
        '<div class="card" style="padding:12px 14px;margin:8px 0 14px">'
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
        '<b>Normalized interpretation</b>'
        f'<span class="pill">{R.normalization_method}</span>'
        f'<span class="pill">{semantic_label}</span>'
        '</div>'
        f'<div style="font-size:1.05rem;margin-top:8px">{normalized}</div>'
        '<div class="muted" style="margin-top:6px">'
        'Primary Diagnose semantics use the matched rules-then-ByT5 SpERT only when its '
        'checkpoint representation is verified and all protected-value/consistency guards pass.</div></div>',
        unsafe_allow_html=True,
    )
    if R.normalization_warning:
        st.caption("Normalization guard: " + R.normalization_warning)
    if R.semantic_warning:
        st.warning(R.semantic_warning)
    if not R.rq4_live_validated:
        st.error("Validated TRUE-RAW SpERT is not available for this query. The system will not issue a primary RQ4/RQ5 recommendation.")

    anchors = " ".join(
        f'<span class="pill">{c}</span>'
        for c in (R.model_components + R.model_faults)
    ) or '<span class="muted">no primary semantic anchor predicted</span>'
    st.markdown(
        f'<div style="margin:6px 0 10px"><b>Primary semantic anchors:</b> {anchors}</div>',
        unsafe_allow_html=True,
    )
    _render_extraction(R)

    with st.expander("Research provenance · validated RQ4/RQ5 branch"):
        st.markdown(
            '<div class="muted">The historical action-family selection and RQ5 probability '
            'are computed only from this true-raw representation.</div>',
            unsafe_allow_html=True,
        )
        _render_rq4_extraction(R)
        derived = R.derived_components + R.derived_faults
        if derived:
            st.markdown("**RQ4 deterministic fallback anchors:** " + ", ".join(derived))
        if R.partial_structure_warning:
            st.warning(R.partial_structure_warning)

    with st.expander("Model input provenance"):
        st.markdown("**User input**")
        st.code(R.query, language=None)
        if R.normalization_model_input:
            st.markdown("**ByT5 model input (expert-rule normalized)**")
            st.code(R.normalization_model_input, language=None)
        st.markdown("**Normalized interpretation**")
        st.code(R.normalized_interpretation or R.query, language=None)
        if R.normalization_model:
            st.caption("Locked ByT5 checkpoint: " + R.normalization_model)
        st.markdown("**Validated RQ4 SpERT input**")
        st.code(R.model_input, language=None)
        st.caption(
            "The optional verified rules-then-ByT5 semantic branch can improve Diagnose understanding. "
            "The frozen RQ4/RQ5 decision branch remains unchanged."
        )

    # Original repo-style recommendation block, now backed by final RQ4 logic.
    if R.compound_detected and R.subproblems:
        _render_compound_plan(R, cfg)
    else:
        if R.structured_sentence:
            st.markdown(
                f'<div class="reco">➜ {R.structured_sentence}</div>'
                f'<div class="muted" style="margin:2px 0 8px">{R.headline_reason}</div>',
                unsafe_allow_html=True,
            )
            if R.badge == "limited":
                st.warning(
                    "Limited historical evidence: one independent historical "
                    "problem cluster supports this action family. The result is "
                    "visible for planning awareness, but is not recurring evidence."
                )

            # RQ4/RQ5 added unobtrusively as an evidence strip, not a new page design.
            prob = R.historical_agreement_probability
            prob_text = f"{100 * prob:.1f}%" if prob is not None else "n/a"
            st.markdown(
                '<div class="card" style="padding:10px 13px;margin:6px 0 14px">'
                '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
                f'<span class="pill"><b>RQ4 base</b> · {cfg["retrieval"].get("default_mode","structure")}</span>'
                f'<span class="pill"><b>RQ5 historical agreement</b> · {prob_text}</span>'
                f'<span class="pill"><b>Independent support</b> · {R.support_clusters} clusters</span>'
                f'<span class="pill"><b>Anchor coverage</b> · {100*R.anchor_coverage:.0f}%</span>'
                f'<span class="pill"><b>Evidence library</b> · {R.candidate_split.upper()}</span>'
                f'<span class="pill"><b>Primary semantics</b> · {R.semantic_status}</span>'
                f'<span class="pill"><b>RQ4 raw adapter</b> · {R.input_adapter}</span>'
                '</div>'
                '<div class="muted" style="margin-top:7px">'
                'RQ5 estimates agreement with recorded historical action families only — '
                'not technical correctness, safety, or regulatory applicability. '
                'The optional reranker affects visible case ordering only.</div></div>',
                unsafe_allow_html=True,
            )

            if st.button("➕ Promote to planning watchlist"):
                st.session_state.setdefault("plan_queue", [])
                st.session_state["plan_queue"].append(
                    {
                        "query": R.query,
                        "recommendation": R.structured_sentence,
                        "action_family": R.headline_action,
                        "component": R.components[0] if R.components else "",
                        "evidence_grade": badge_label(R.badge),
                        "rq5_historical_agreement": (
                            round(float(R.historical_agreement_probability), 4)
                            if R.historical_agreement_probability is not None
                            else None
                        ),
                        "support_clusters": R.support_clusters,
                        "anchor_coverage": round(float(R.anchor_coverage), 4),
                    }
                )
                st.success("Added to planning watchlist.")

            # Restore the original repo's strong strategy presentation.
            if R.strategies:
                st.markdown("#### Recorded strategies for this problem")
                st.markdown(
                    '<div class="muted" style="margin-top:-6px;margin-bottom:8px">'
                    'Semantically similar historical problems were resolved in different ways. '
                    'Each strategy below is grouped from recorded actions and linked to source cases.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                shown_weak_header = False
                for strategy in R.strategies:
                    if strategy.tier == "single_case" and not shown_weak_header:
                        st.markdown(
                            '<div class="muted" style="margin:12px 0 6px;font-weight:600">'
                            'Single-case options — recorded once, weak evidence, review only'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        shown_weak_header = True
                    _render_strategy(strategy)
        else:
            family = R.evidence_family or "none"
            st.info(
                "Closest historical cases are shown, but the evidence gate did not surface "
                f"a positive action recommendation. Highest historical family: {family}."
            )
            prob = R.historical_agreement_probability
            if prob is not None:
                st.caption(
                    f"RQ5 historical action-family agreement estimate: {100*prob:.1f}% "
                    "(not technical correctness)."
                )

    if not R.compound_detected:
        if R.recommended_cases:
            with st.expander("Why this strategy? Supporting historical work orders"):
                _render_cases(R.recommended_cases)

        if R.negative_evidence:
            with st.expander("⚠ Negative / unresolved evidence (excluded from the recommendation)"):
                _render_cases(R.negative_evidence)

        with st.expander("All nearest cases (transparency)"):
            _render_cases(R.nearest_cases)
    else:
        with st.expander("Combined-query nearest cases · research transparency"):
            _render_cases(R.nearest_cases)

    st.caption(
        "Decision support only. Verify against current approved maintenance data and qualified "
        "maintenance authority before operational use."
    )



def page_insights(key: str, cfg: dict):
    st.markdown("### Insights")
    st.markdown('<div class="muted">How often things are recorded as failing, and how often '
                'each action is taken. Occurrence counts only — not reliability rates.</div>',
                unsafe_allow_html=True)
    tabs = st.tabs(["Recurring faults", "Component Pareto", "Fault modes",
                    "Actions", "Component × Fault", "Problem → Action", "Outcomes"])

    with tabs[0]:
        T.section("Chronic-defect register", "Recurring problem clusters, most frequent first.")
        wl = c_recurring(key, cfg["insights"]["recurring_min"], 40)
        st.dataframe(wl.drop(columns=["cluster_id"]).rename(columns={
            "problem": "Problem", "component": "Component", "fault": "Fault",
            "work_orders": "Work orders", "dominant_action": "Dominant action",
            "positive_outcomes": "Positive", "negative_outcomes": "Neg/unresolved",
            "outcome_unknown": "Unknown"}), hide_index=True, use_container_width=True, height=520)
    with tabs[1]:
        T.section("Component Pareto", "The vital few components behind most work orders.")
        st.plotly_chart(C.pareto(c_component_freq(key, 15), "component", "share_pct", "cumulative_pct"),
                        use_container_width=True, config={"displayModeBar": False})
    with tabs[2]:
        T.section("Fault modes", "Which failure conditions dominate.")
        st.plotly_chart(C.hbar(c_fault_freq(key, 15), "fault", "work_orders", C.SERIES[4], 460),
                        use_container_width=True, config={"displayModeBar": False})
    with tabs[3]:
        T.section("Action frequency", "What the fleet keeps doing.")
        st.plotly_chart(C.action_bars(c_action_freq(key), "action_family", "work_orders", 360),
                        use_container_width=True, config={"displayModeBar": False})
    with tabs[4]:
        T.section("Component × Fault", "Which components get which faults.")
        st.plotly_chart(C.heatmap(c_matrix(key, 8, 8)),
                        use_container_width=True, config={"displayModeBar": False})
    with tabs[5]:
        T.section("Problem → Action", "For a component, what actions were recorded.")
        sel = st.selectbox("Component", c_component_list(key, 25), label_visibility="collapsed")
        p2a = c_p2a(key, sel)
        if len(p2a):
            st.plotly_chart(C.action_bars(p2a, "action_family", "work_orders", 320),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No recorded actions for that component.")
    with tabs[6]:
        T.section("Outcome mix", "Recorded outcomes — mostly unknown, shown honestly.")
        st.plotly_chart(C.outcome_donut(c_outcome(key)), use_container_width=True,
                        config={"displayModeBar": False})


def page_kg(key: str, cfg: dict):
    st.markdown("### Knowledge Graph")
    st.markdown('<div class="muted">The corpus as a network — <b>components</b> (blue) carry '
                '<b>faults</b> (red) that <b>actions</b> (green) resolve. Click any node to light up '
                'its connections; pick a component below to focus on just its faults and actions.</div>',
                unsafe_allow_html=True)
    c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
    comp_options = ["— whole corpus —"] + c_component_list(key, 30)
    focus = c0.selectbox("Focus on a component", comp_options, label_visibility="collapsed")
    focus = "" if focus.startswith("—") else focus
    top_c = c1.slider("Components", 5, 20, 10, disabled=bool(focus))
    top_f = c2.slider("Faults", 4, 15, 8)
    min_edge = c3.slider("Min co-occurrence", 1, 20, 1 if focus else 4)
    st.markdown(
        '<div class="kglegend">'
        '<span><span class="dot" style="background:#3987e5"></span> Component</span>'
        '<span><span class="dot" style="background:#e66767"></span> Fault</span>'
        '<span><span class="sq" style="background:#199e70"></span> Action family</span>'
        '<span class="muted">· click a node to highlight its neighbours</span>'
        '</div>', unsafe_allow_html=True)
    html = c_kg_html(key, top_c, top_f, min_edge, focus)
    components.html(html, height=640, scrolling=False)
    st.caption("Observed co-occurrences in the work orders — not causal or reliability claims.")


def page_eval(cfg: dict, key: str):
    st.markdown("### Evaluation")
    st.markdown(
        '<div class="muted">Frozen thesis evidence for retrieval, uncertainty and semantic usefulness. '
        'These values describe agreement with recorded history and human/AI-assisted relevance judgments — '
        'not technical correctness or aircraft-specific approval.</div>',
        unsafe_allow_html=True,
    )

    frozen = get_frozen_evaluation()
    rq4, rq5, manual, lock = frozen.get("rq4"), frozen.get("rq5"), frozen.get("manual"), frozen.get("lock")
    if not (rq4 and rq5 and manual and lock and lock.get("locked")):
        st.warning("One or more frozen evaluation summaries are missing. Run FINAL_11_VERIFY_ALL.bat and check outputs/frozen.")
        return

    m = rq4["selected_mode_metrics"]
    T.kpi_row([
        ("RQ4 Hit@1", f'{100*m["hit_at_1"]:.1f}%', "locked TEST"),
        ("RQ4 Hit@3", f'{100*m["hit_at_3"]:.1f}%', "locked TEST"),
        ("MRR", f'{m["mrr"]:.3f}', "structured retrieval"),
        ("Macro family recall", f'{100*m["macro_action_family_recall"]:.1f}%', "balanced action families"),
        ("RQ5 ECE", f'{rq5["ece_10_bins"]:.3f}', "10 bins"),
        ("RQ5 Brier", f'{rq5["brier_score"]:.3f}', "historical agreement"),
    ])

    c1, c2 = st.columns([1.05, 1])
    with c1:
        T.section("Selective risk–coverage", "Higher-confidence subsets show stronger historical action-family agreement.")
        rc = pd.DataFrame(rq5["risk_coverage"])
        fig = go.Figure()
        fig.add_scatter(
            x=100*rc["coverage"], y=100*rc["agreement_accuracy"],
            mode="lines+markers", name="Agreement", line=dict(color=T.SERIES[0], width=3),
            marker=dict(size=7), hovertemplate="Coverage %{x:.0f}% · agreement %{y:.1f}%<extra></extra>",
        )
        fig.update_layout(
            height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=T.INK2, size=12), margin=dict(l=8, r=12, t=10, b=8),
            xaxis=dict(title="coverage (%)", range=[0, 105], gridcolor=T.HAIR),
            yaxis=dict(title="historical agreement (%)", range=[0, 105], gridcolor=T.HAIR),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="research-note"><b>Calibration boundary.</b> Probability estimates historical '
            'action-family agreement only. They are not probabilities of safety, technical correctness, '
            'regulatory applicability or aircraft release.</div>', unsafe_allow_html=True)

    with c2:
        T.section("Semantic review", "100 locked-TEST queries · 500 blinded problem pairs · 380 Phase-B action pairs.")
        pa, pb = manual["phase_a"], manual["phase_b"]
        depths = ["@1", "@3", "@5"]
        prob = [100*pa["problem_relevance_hit_at_1"], 100*pa["problem_relevance_hit_at_3"], 100*pa["problem_relevance_hit_at_5"]]
        act = [100*pb["usable_historical_action_evidence_hit_at_1"], 100*pb["usable_historical_action_evidence_hit_at_3"], 100*pb["usable_historical_action_evidence_hit_at_5"]]
        fig2 = go.Figure()
        fig2.add_bar(x=depths, y=prob, name="Relevant problem found", marker_color=T.SERIES[0], text=[f"{x:.0f}%" for x in prob], textposition="outside")
        fig2.add_bar(x=depths, y=act, name="Useful action evidence", marker_color=T.SERIES[2], text=[f"{x:.0f}%" for x in act], textposition="outside")
        fig2.update_layout(
            height=330, barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=T.INK2, size=12), margin=dict(l=8, r=8, t=20, b=8),
            legend=dict(orientation="h", y=1.13, bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="query-level hit rate (%)", range=[0, 105], gridcolor=T.HAIR),
            xaxis=dict(gridcolor=T.HAIR),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="research-note"><b>Action usefulness:</b> {100*pb["applicable_or_partial_rate_among_problem_relevant_pairs"]:.1f}% '
            f'of problem-relevant pairs were at least partially useful historical action evidence; '
            f'{100*pb["clearly_applicable_rate_among_problem_relevant_pairs"]:.1f}% were judged clearly useful. '
            'Single-reviewer/AI-assisted judgments do not provide inter-rater reliability.</div>', unsafe_allow_html=True)

    T.section("Frozen RQ4 protocol", "The base retrieval system was selected on DEV before the one-time TEST was exposed.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Selected mode", rq4["selected_mode"])
    p2.metric("Evaluable TEST queries", f'{m["queries"]:,}')
    p3.metric("nDCG@5", f'{m["ndcg_at_5"]:.3f}')
    p4.metric("Hit@5", f'{100*m["hit_at_5"]:.1f}%')
    st.caption("The optional dashboard cross-encoder reranker is an operational demonstration layer and is not the basis for the frozen RQ4 selection.")


def page_guide():
    st.markdown("## Guide")
    st.markdown("""
AviMaint-DSS is an **evidence-grounded maintenance planning-support prototype**.
It does not generate an untraceable maintenance procedure. The final dashboard
combines the strongest semantic-action logic from the repository with the frozen
RQ4/RQ5 research protocol.

**Diagnose pipeline**

1. The user's **problem text only** is sent to the matched true-Raw SpERT service.
2. The frozen **RQ4-selected `structure` retrieval mode** retrieves historical
   problem cases from the frozen TRAIN evidence library.
3. The action family is aggregated from **independent problem clusters**. Repeated
   rows from the same cluster cannot inflate support.
4. Historical ACTION text is opened **after retrieval**, then grouped into
   distinct recorded strategies such as Replace, Repair, Adjust, Service,
   Inspect or Diagnose.
5. A deterministic sentence composer turns the supported strategy and extracted
   problem anchors into a readable planning sentence. A verification step is
   added only when it is actually present in the supporting historical actions.
6. The optional cross-encoder reranker can improve the ordering of cases shown to
   the user, but it does **not** replace the frozen RQ4 base decision.
7. RQ5 adds a DEV-fitted **historical action-family agreement probability** using
   the same `top_score`, `margin` and `support_clusters` features used in the
   final calibration experiment.

**What the RQ5 probability means**

It estimates how likely the predicted action family is to agree with the action
family recorded in historical data. It is **not** a probability that an action is
technically correct, safe, approved, airworthy, or applicable to a particular
aircraft.

**Planning**

Items promoted from Diagnose carry their evidence grade, independent-cluster
support, anchor coverage and RQ5 historical-agreement estimate. Recurring-fault
job cards remain deterministic and source-grounded: every displayed action step
comes from a recorded historical work order.

**Evaluation**

The Evaluation page reads the immutable final RQ4 TEST, RQ5 calibration and
manual semantic-review summaries. It does not recompute or tune the frozen test
results from the dashboard.

Always verify planning support against current approved maintenance data and
qualified maintenance authority before operational use.
""")



def page_planning(df: pd.DataFrame, cfg: dict, key: str):
    st.markdown("### Planning & Support")
    st.markdown('<div class="muted">Turn recurring faults into standing plans and draft '
                'traceable job-card evidence from recorded historical actions.</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="research-note"><b>RQ5 planning-support rule:</b> a probability carried '
        'from Diagnose means expected agreement with the recorded historical action family. '
        'It is uncertainty information for prioritisation/abstention, not an approval score. '
        'Every action step below remains traceable to a historical work order.</div>',
        unsafe_allow_html=True,
    )

    queue = st.session_state.get("plan_queue", [])
    if queue:
        T.section("Your planning watchlist", "Promoted from Diagnose.")
        st.dataframe(pd.DataFrame(queue), hide_index=True, use_container_width=True)
        if st.button("Clear watchlist"):
            st.session_state["plan_queue"] = []

    T.section("Recurring-fault register", "Pick a chronic problem to draft a job card.")
    wl = c_recurring(key, cfg["insights"]["recurring_min"], 40)
    labels = [f'{r.work_orders:>3} × {r.problem[:55]}' for r in wl.itertuples(index=False)]
    if not labels:
        st.info("No recurring clusters at the current threshold.")
        return
    pick = st.selectbox("Recurring problem", range(len(labels)), format_func=lambda i: labels[i])
    cid = wl.iloc[pick]["cluster_id"]
    card = W.job_card_for_cluster(df, cid)

    st.markdown(f"### Job card · {card.title}")
    T.kpi_row([
        ("Work orders", f"{card.work_orders:,}", ""),
        ("Problem groups", f"{card.problem_groups:,}", ""),
        ("Dominant action", card.dominant_action or "—", ""),
        ("Positive", f"{card.outcome_positive}", ""),
        ("Neg/unresolved", f"{card.outcome_negative}", ""),
        ("Unknown", f"{card.outcome_unknown}", ""),
    ])
    st.markdown("#### Recorded action steps (grounded, deduplicated)")
    if card.steps:
        for i, s in enumerate(card.steps, 1):
            st.markdown(f'{i}. {s["text"]}  '
                        f'<span class="caseid">{",".join(s["source_idents"])}</span>',
                        unsafe_allow_html=True)
    else:
        st.info("No positive/known-outcome action steps recorded for this cluster.")
    if card.references:
        st.markdown("**Cited references:** " + ", ".join(card.references))
    st.caption("Every step is a recorded historical action. Confirm against current approved "
               "maintenance data before use — this is decision support, not authorisation.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    T.setup_page()
    cfg = get_config()
    key = cfg_signature(cfg)
    corpus = get_corpus(key)
    retr, rec = get_engine(key)
    df = corpus.df

    # Live SpERT auto-detection (independent of config mode). Starting the
    # service any time is picked up on the next interaction.
    spert = SpERTClient(cfg["extraction"]["spert_url"])
    spert_on = spert.health()
    rec.client = spert if spert_on else None

    rr_on = bool(getattr(rec, "reranker", None)) and rec.reranker.available()
    ds = os.path.basename(cfg["data"].get("csv_path", ""))

    # ---- professional product header + live status ------------------------- #
    hl, hr = st.columns([2.2, 3.8])
    with hl:
        st.markdown(
            '<div class="brand-shell"><span class="brand-mark">✈</span>'
            '<span class="brand-copy"><span class="brand-name">AviMaint-DSS</span>'
            '<span class="brandsub">Evidence-grounded maintenance planning support</span></span></div>',
            unsafe_allow_html=True,
        )
    with hr:
        spert_chip = (
            '<span class="chip on"><span class="sdot on"></span>SpERT on</span>' if spert_on
            else '<span class="chip off"><span class="sdot off"></span>SpERT off</span>'
        )
        rr_chip = (
            '<span class="chip on"><span class="sdot on"></span>Reranker on · optional</span>' if rr_on
            else '<span class="chip off"><span class="sdot off"></span>Reranker off</span>'
        )
        base_mode = getattr(rec, "retrieval_mode", cfg["retrieval"].get("default_mode", "structure"))
        st.markdown(
            f'<div class="chiprow">{spert_chip}{rr_chip}'
            f'<span class="chip mode">Base · {base_mode}</span>'
            f'<span class="chip data">Validated evidence · {len(rec.df):,} {rec.candidate_split.upper()}</span>'
            f'<span class="chip data">{corpus.n:,} work orders · {ds}</span></div>',
            unsafe_allow_html=True,
        )

    # ---- full-width horizontal navigation ---------------------------------- #
    pages = ["Overview", "Diagnose", "Insights", "Knowledge Graph", "Planning", "Evaluation", "Guide"]
    icons = ["speedometer2", "search", "bar-chart-line", "diagram-3", "clipboard-check",
             "clipboard-data", "info-circle"]
    page = option_menu(
        None, pages, icons=icons, orientation="horizontal", default_index=0,
        styles={
            "container": {
                "padding": "6px", "background-color": "#ffffff",
                "border": "1px solid #e2e8f0", "border-radius": "14px",
                "margin": "6px 0 22px", "width": "100%",
                "box-shadow": "0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04)",
            },
            "nav-link": {
                "font-size": "13px", "color": "#475569", "padding": "10px 12px",
                "margin": "0", "border-radius": "9px", "white-space": "nowrap",
                "font-weight": "600", "--hover-color": "#eff6ff", "text-align": "center",
                "transition": "all .15s ease",
            },
            "nav-link-selected": {
                "background-color": "#2563eb", "color": "#ffffff", "font-weight": "700",
                "box-shadow": "0 4px 12px rgba(37,99,235,.20)",
            },
            "icon": {"font-size": "14px", "margin-right": "7px"},
        },
    )

    if page == "Overview":
        page_overview(key, cfg)
    elif page == "Diagnose":
        page_diagnose(rec, cfg)
    elif page == "Insights":
        page_insights(key, cfg)
    elif page == "Knowledge Graph":
        page_kg(key, cfg)
    elif page == "Planning":
        page_planning(df, cfg, key)
    elif page == "Evaluation":
        page_eval(cfg, key)
    else:
        page_guide()

    T.footer()

    # reranker load error (if any) — small footer note, not a blocker
    if getattr(rec, "reranker", None) is not None and not rr_on:
        with st.expander("⚠ Reranker configured but not loaded — why?"):
            st.caption(rec.reranker.last_error())


if __name__ == "__main__":
    main()
