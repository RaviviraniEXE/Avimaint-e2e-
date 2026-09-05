import importlib.util

for name in ("fastapi", "uvicorn"):
    assert importlib.util.find_spec(name) is not None, f"{name} not installed"

import api_server

routes = {r.path for r in api_server.app.routes}
required = {
    "/api/v1/health",
    "/api/v1/diagnose",
    "/api/v1/overview",
    "/api/v1/components",
    "/api/v1/evaluation",
    "/api/v1/config/public",
}
missing = required - routes
assert not missing, missing

schema = api_server.app.openapi()
assert schema["info"]["version"] == "1.0.2"
assert "/api/v1/diagnose" in schema["paths"]
assert "post" in schema["paths"]["/api/v1/diagnose"]

print("PHASE4_API_CONTRACT_V3_OK")
for route in sorted(required):
    print(route)
