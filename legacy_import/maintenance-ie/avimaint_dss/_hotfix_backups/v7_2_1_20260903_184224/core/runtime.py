"""Streamlit-independent AviMaint runtime for the final frontend API."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .corpus import load_corpus
from .extraction import SpERTClient
from .retrieval import Retriever
from .recommend import Recommender
from .reranker import CrossEncoderReranker
from .calibration import RQ5AgreementCalibrator
from .query_normalization import NormalizationClient

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_runtime_lock() -> dict:
    path = ROOT / "runtime_model_lock.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "byt5": {"enabled": False},
            "normalized_spert": {
                "enabled": False,
                "verified_representation": False,
            },
        }


@lru_cache(maxsize=1)
def get_runtime():
    cfg = load_config()

    pred = cfg["data"].get("problem_predictions_path") or None
    protocol = cfg["data"].get("protocol_path") or None
    corpus = load_corpus(
        ROOT / cfg["data"]["csv_path"],
        mode=cfg["extraction"]["mode"],
        spert_url=cfg["extraction"]["spert_url"],
        predictions_path=(ROOT / pred).resolve() if pred else None,
        protocol_path=(ROOT / protocol).resolve() if protocol else None,
        require_predictions=cfg["extraction"].get("require_problem_predictions", False),
        normalize=False,
    )

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

    retriever = Retriever(
        candidate_df,
        weights=cfg["retrieval"].get("weights"),
        dense_model=cfg["retrieval"].get("dense_model") or None,
        rrf_k=cfg["retrieval"].get("rrf_k", 60),
    )

    raw_client = None
    if cfg["extraction"]["mode"] == "spert":
        c = SpERTClient(cfg["extraction"]["spert_url"])
        raw_client = c if c.health() else None

    runtime_lock = load_runtime_lock()

    semantic_client = None
    scfg = cfg.get("semantic_extraction", {})
    sem_lock = runtime_lock.get("normalized_spert", {})
    if (
        scfg.get("enabled", True)
        and sem_lock.get("enabled", False)
        and sem_lock.get("verified_representation", False)
    ):
        c = SpERTClient(scfg.get("spert_url", "http://127.0.0.1:8767"))
        semantic_client = c if c.health() else None

    ncfg = cfg.get("normalization", {})
    byt5_lock = runtime_lock.get("byt5", {})
    normalizer = NormalizationClient(
        url=ncfg.get("service_url", "http://127.0.0.1:8766"),
        timeout=float(ncfg.get("timeout_seconds", 45)),
        enabled=bool(ncfg.get("enabled", True) and byt5_lock.get("enabled", False)),
    )

    rr_model = cfg["retrieval"].get("reranker_model") or None
    reranker = (
        CrossEncoderReranker(rr_model, cfg["retrieval"].get("reranker_blend", 0.7))
        if rr_model else None
    )

    cal_rel = research.get("rq5_dev_predictions_path")
    calibrator = RQ5AgreementCalibrator((ROOT / cal_rel).resolve() if cal_rel else None)

    rcfg = cfg.get("recommender", {})
    recommender = Recommender(
        candidate_df,
        retriever,
        spert_client=raw_client,
        semantic_spert_client=semantic_client,
        normalizer=normalizer,
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

    return {
        "config": cfg,
        "corpus": corpus,
        "candidate_df": candidate_df,
        "candidate_split": candidate_split,
        "retriever": retriever,
        "recommender": recommender,
        "raw_client": raw_client,
        "semantic_client": semantic_client,
        "normalizer": normalizer,
        "reranker": reranker,
        "calibrator": calibrator,
        "runtime_lock": runtime_lock,
    }


def frozen_evaluation():
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


def jsonable(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
