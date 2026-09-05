"""Frontend-ready FastAPI service for AviMaint-DSS."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, HTMLResponse

from core.runtime import get_runtime, frozen_evaluation, jsonable
from core import insights as I
from core.frontend_views import (
    insights_payload,
    job_card_payload,
    knowledge_graph_payload,
    recurring_planning_payload,
)


ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
FRONTEND_VERSION = "5.0.0"

app = FastAPI(
    title="AviMaint-DSS API",
    version="1.0.1",
    description="Structured maintenance IE, historical retrieval and planning support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:3000", "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class DiagnoseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=25, ge=5, le=100)


class JobCardRequest(BaseModel):
    cluster_id: str = Field(min_length=1, max_length=500)


def frontend_health() -> dict:
    index = FRONTEND_DIST / "index.html"
    build_info = FRONTEND_DIST / "build-info.json"
    version = ""
    try:
        version = str(json.loads(build_info.read_text(encoding="utf-8")).get("version", ""))
    except Exception:
        version = ""
    ready = bool(index.is_file() and version == FRONTEND_VERSION)
    return {
        "ready": ready,
        "version": version,
        "expected_version": FRONTEND_VERSION,
        "mode": "fastapi_static",
        "url": "/",
    }


def service_health(client):
    if client is None:
        return {"ready": False}
    try:
        url = str(getattr(client, "url", "")).rstrip("/") + "/health"
        with urllib.request.urlopen(url, timeout=3.0) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        return {"ready": True, "metadata": metadata}
    except Exception as exc:
        return {"ready": False, "error": type(exc).__name__}


@app.get("/api/v1/health")
def health():
    rt = get_runtime()
    cfg = rt["config"]
    raw = service_health(rt["raw_client"])
    semantic = service_health(rt["semantic_client"])
    nh = rt["normalizer"].health() if rt["normalizer"] else None
    calibrator_ready = bool(rt["calibrator"] and rt["calibrator"].available())
    critical_ready = bool(raw["ready"] and calibrator_ready)

    return {
        "status": "ready" if critical_ready else "degraded",
        "api_version": "1.0.1",
        "rq4_base": cfg["retrieval"].get("default_mode", "structure"),
        "candidate_split": rt["candidate_split"],
        "phase2_compound_decomposition": bool(cfg.get("phase2", {}).get("enabled", False)),
        "phase3_limited_evidence": bool(cfg.get("phase3", {}).get("enabled", False)),
        "raw_spert": raw,
        "normalization": {"ready": bool(nh), "metadata": nh},
        "semantic_spert": semantic,
        "runtime_model_lock": rt.get("runtime_lock", {}),
        "rq5_calibrator": {
            "ready": calibrator_ready,
            "status": rt["calibrator"].status() if rt["calibrator"] else "",
        },
        "frontend": frontend_health(),
        "critical_ready": critical_ready,
    }


@app.post("/api/v1/diagnose")
def diagnose(req: DiagnoseRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")
    rt = get_runtime()
    try:
        result = rt["recommender"].recommend(req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnose pipeline failed: {type(exc).__name__}",
        ) from exc
    return {
        "result": jsonable(result),
        "meta": {
            "rq4_base": rt["config"]["retrieval"].get("default_mode", "structure"),
            "candidate_split": rt["candidate_split"],
            "reranker_role": "presentation_only",
            "rq5_meaning": "historical_action_family_agreement",
            "technical_correctness_probability": False,
        },
    }


@app.get("/api/v1/overview")
def overview():
    rt = get_runtime()
    cfg = rt["config"]
    df = rt["corpus"].df
    recurring_min = int(cfg.get("insights", {}).get("recurring_min", 5))
    return {
        "kpis": jsonable(I.kpis(df, recurring_min)),
        "top_recurring": jsonable(I.top_recurring_problems(df, 12).to_dict("records")),
        "component_frequency": jsonable(I.component_frequency(df, 15).to_dict("records")),
        "fault_frequency": jsonable(I.fault_frequency(df, 15).to_dict("records")),
        "action_frequency": jsonable(I.action_frequency(df).to_dict("records")),
        "outcome_mix": jsonable(I.outcome_mix(df).to_dict("records")),
        "note": "Observed work-order counts; not failure rates.",
    }


@app.get("/api/v1/components")
def components(limit: int = Query(default=100, ge=1, le=1000)):
    rt = get_runtime()
    return {
        "items": jsonable(
            I.component_frequency(rt["corpus"].df, limit).to_dict("records")
        )
    }


@app.get("/api/v1/evaluation")
def evaluation():
    return {
        "frozen": frozen_evaluation(),
        "warning": (
            "Frozen RQ4/RQ5 values describe the locked research protocol. "
            "Compound decomposition and limited-evidence presentation are "
            "operational extensions."
        ),
    }


@app.get("/api/v1/config/public")
def public_config():
    rt = get_runtime()
    cfg = rt["config"]
    return {
        "retrieval_mode": cfg["retrieval"].get("default_mode", "structure"),
        "candidate_split": rt["candidate_split"],
        "normalization_mode": cfg.get("normalization", {}).get("mode"),
        "semantic_mode": cfg.get("semantic_extraction", {}).get("mode"),
        "phase2": cfg.get("phase2", {}),
        "phase3": cfg.get("phase3", {}),
        "api": cfg.get("api", {}),
    }


@app.get("/api/v1/insights")
def insights(
    component: str = Query(default="", max_length=300),
    top_components: int = Query(default=15, ge=5, le=30),
    top_faults: int = Query(default=15, ge=4, le=30),
):
    rt = get_runtime()
    recurring_min = int(rt["config"].get("insights", {}).get("recurring_min", 5))
    return jsonable(
        insights_payload(
            rt["corpus"].df,
            component=component,
            recurring_min=recurring_min,
            top_components=top_components,
            top_faults=top_faults,
        )
    )


@app.get("/api/v1/knowledge-graph")
def knowledge_graph(
    top_components: int = Query(default=10, ge=5, le=20),
    top_faults: int = Query(default=8, ge=4, le=15),
    min_edge: int = Query(default=3, ge=1, le=50),
    focus_component: str = Query(default="", max_length=300),
):
    rt = get_runtime()
    return jsonable(
        knowledge_graph_payload(
            rt["corpus"].df,
            top_components=top_components,
            top_faults=top_faults,
            min_edge=min_edge,
            focus_component=focus_component,
        )
    )


@app.get("/api/v1/planning/recurring")
def planning_recurring(
    min_support: int = Query(default=5, ge=2, le=1000),
    limit: int = Query(default=40, ge=1, le=200),
):
    rt = get_runtime()
    return jsonable(
        recurring_planning_payload(
            rt["corpus"].df,
            min_support=min_support,
            limit=limit,
        )
    )


@app.post("/api/v1/planning/job-card")
def planning_job_card(req: JobCardRequest):
    rt = get_runtime()
    payload = job_card_payload(rt["corpus"].df, req.cluster_id)
    if not payload["card"]["work_orders"]:
        raise HTTPException(status_code=404, detail="Historical problem cluster not found")
    return jsonable(payload)


def _frontend_index():
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<h1>AviMaint-DSS frontend is not built</h1>"
            "<p>Run the Phase-5 installer or build frontend/dist.</p>",
            status_code=503,
        )
    response = FileResponse(index)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def frontend_index():
    return _frontend_index()


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_files(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    root = FRONTEND_DIST.resolve()
    candidate = (FRONTEND_DIST / full_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if candidate.is_file():
        response = FileResponse(candidate)
        if full_path.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
    return _frontend_index()
