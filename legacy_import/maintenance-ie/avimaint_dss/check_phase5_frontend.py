"""Static/API contract checks for the Phase-5 frontend integration."""
from __future__ import annotations

import json
from pathlib import Path

import api_server


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"


routes = {route.path for route in api_server.app.routes}
required = {
    "/api/v1/health",
    "/api/v1/diagnose",
    "/api/v1/overview",
    "/api/v1/components",
    "/api/v1/evaluation",
    "/api/v1/config/public",
    "/api/v1/insights",
    "/api/v1/knowledge-graph",
    "/api/v1/planning/recurring",
    "/api/v1/planning/job-card",
    "/",
    "/{full_path:path}",
}
missing = required - routes
assert not missing, f"missing Phase-5 routes: {sorted(missing)}"

index = DIST / "index.html"
build_info = DIST / "build-info.json"
assert index.is_file(), "frontend/dist/index.html is missing"
assert build_info.is_file(), "frontend/dist/build-info.json is missing"
info = json.loads(build_info.read_text(encoding="utf-8"))
assert info.get("version") == api_server.FRONTEND_VERSION == "5.0.1"
assert info.get("apiVersion") == "1.0.2"

launcher = (ROOT / "FINAL_12_START_DASHBOARD.bat").read_text(encoding="utf-8")
assert "FINAL_12_START_DASHBOARD.bat" not in launcher
assert "tools\\runtime_supervisor.py" in launcher
assert "start " not in launcher.lower(), "launcher must not open service console windows"
supervisor = (ROOT / "tools" / "runtime_supervisor.py").read_text(encoding="utf-8")
assert "http://127.0.0.1:8780/" in supervisor
assert '"5.0.1"' in supervisor
assert '"1.0.2"' in supervisor
assert "legacy_streamlit" in supervisor, "legacy comparison option was removed"

schema = api_server.app.openapi()
for path in required:
    if path.startswith("/api/"):
        assert path in schema["paths"], path

print("PHASE5_FRONTEND_CONTRACT_OK")
print("PHASE5_FRONTEND_VERSION=5.0.1")
print("PHASE4_API_VERSION=1.0.2")
