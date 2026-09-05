
from __future__ import annotations
import argparse, hashlib, json, sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))
from spert_runtime import SpERTRunner, SpERTRuntimeError, resolve_runtime_paths

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_lock(path: Path):
    lock = json.loads(path.read_text(encoding="utf-8"))
    info = lock["normalized_spert"]
    if not bool(info.get("enabled", False)) or not bool(info.get("verified_representation", False)):
        raise RuntimeError(
            "Normalized semantic SpERT is disabled because its representation was not "
            "verified. " + str(info.get("reason", ""))
        )
    model = Path(info["model_path"]).resolve()
    types = Path(info["types_path"]).resolve()
    wf = model / "model.safetensors"
    if not wf.is_file():
        wf = model / "pytorch_model.bin"
    actual = digest(wf)
    expected = str(info["weight_sha256"])
    if actual.lower() != expected.lower():
        raise RuntimeError(f"Normalized SpERT SHA mismatch: expected={expected}, actual={actual}")
    return lock, model, types, actual

def make_handler(runner, lock, model, actual):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, obj):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                meta = dict(runner.metadata)
                meta.update({
                    "role": "normalized_semantic_spert",
                    "representation": lock.get("semantic_representation", "normalized_operational"),
                    "weights_sha256": actual,
                    "locked_model_path": str(model),
                })
                self.send_json(HTTPStatus.OK, meta)
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str):
                    raise RuntimeError("Missing string field 'text'")
                self.send_json(HTTPStatus.OK, runner.predict(text))
            except Exception as exc:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}")
    return Handler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke-text", default="left-hand magneto excessive rpm drop during run up")
    args = ap.parse_args()

    lock, model, types, actual = read_lock(Path(args.lock).resolve())
    paths = resolve_runtime_paths(args.project_root, model_path=str(model), types_path=str(types))
    runner = SpERTRunner(
        paths, rel_filter_threshold=0.4, max_span_size=10, max_pairs=1000,
        force_cpu=args.cpu
    )
    meta = dict(runner.metadata)
    meta.update({
        "role": "normalized_semantic_spert",
        "representation": lock.get("semantic_representation", "normalized_operational"),
        "weights_sha256": actual,
        "locked_model_path": str(model),
    })
    print(json.dumps(meta, indent=2), flush=True)
    if args.check_only:
        pred = runner.predict(args.smoke_text)
        print(json.dumps(pred, indent=2), flush=True)
        etypes = {str(e.get("type")) for e in pred.get("entities", [])}
        rtypes = {str(r.get("type")) for r in pred.get("relations", [])}
        if "MAINT_ITEM" not in etypes:
            raise RuntimeError(
                "Semantic SpERT compatibility check failed: smoke text produced no MAINT_ITEM."
            )
        if not ({"FAULT", "ABN_PROC"} & etypes):
            raise RuntimeError(
                "Semantic SpERT compatibility check failed: smoke text produced no issue entity."
            )
        if "ISSUE_ON_ITEM" not in rtypes:
            raise RuntimeError(
                "Semantic SpERT compatibility check failed: smoke text produced no ISSUE_ON_ITEM."
            )
        print("NORMALIZED_SPERT_CHECK_OK")
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner, lock, model, actual))
    print(f"Normalized SpERT ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
