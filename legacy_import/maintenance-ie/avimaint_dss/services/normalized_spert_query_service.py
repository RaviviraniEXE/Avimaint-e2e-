
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
    representation = str(lock.get("semantic_representation", ""))
    if representation != "rules_then_byt5_guarded_operational":
        raise RuntimeError(f"Unsupported semantic representation: {representation or 'missing'}")
    if not bool(lock.get("normalization_rules", {}).get("enabled", False)):
        raise RuntimeError("Semantic SpERT cannot start without the locked expert-rule stage.")
    if not bool(lock.get("byt5", {}).get("enabled", False)):
        raise RuntimeError("Semantic SpERT cannot start without the locked ByT5 stage.")
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
    actual_types = digest(types)
    expected_types = str(info.get("types_sha256", ""))
    if not expected_types or actual_types.lower() != expected_types.lower():
        raise RuntimeError(
            f"Normalized SpERT type-definition SHA mismatch: expected={expected_types}, actual={actual_types}"
        )
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
                    "role": "rules_then_byt5_semantic_spert",
                    "representation": lock.get("semantic_representation"),
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


def prediction_contract_ok(prediction: dict) -> tuple[bool, str]:
    """Validate runtime shape/index integrity without inventing an accuracy test.

    Model quality is established by the frozen registry metrics. A startup
    smoke test should prove that the exact model can execute on its recorded
    representation, not require particular labels for hand-picked sentences.
    """
    if not isinstance(prediction, dict):
        return False, "prediction is not an object"
    entities = prediction.get("entities")
    relations = prediction.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return False, "entities/relations are not lists"
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or not isinstance(entity.get("type"), str):
            return False, f"entity {index} has invalid shape"
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict) or not isinstance(relation.get("type"), str):
            return False, f"relation {index} has invalid shape"
        head, tail = relation.get("head"), relation.get("tail")
        if not isinstance(head, int) or not isinstance(tail, int):
            return False, f"relation {index} endpoints are not integer indices"
        if not (0 <= head < len(entities) and 0 <= tail < len(entities)):
            return False, f"relation {index} endpoint is outside the entity list"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke-text", default="")
    ap.add_argument("--smoke-input", default="")
    args = ap.parse_args()

    lock, model, types, actual = read_lock(Path(args.lock).resolve())
    paths = resolve_runtime_paths(args.project_root, model_path=str(model), types_path=str(types))
    runner = SpERTRunner(
        paths, rel_filter_threshold=0.4, max_span_size=10, max_pairs=1000,
        force_cpu=args.cpu
    )
    meta = dict(runner.metadata)
    meta.update({
        "role": "rules_then_byt5_semantic_spert",
        "representation": lock.get("semantic_representation"),
        "weights_sha256": actual,
        "locked_model_path": str(model),
    })
    print(json.dumps(meta, indent=2), flush=True)
    if args.check_only:
        texts = []
        if args.smoke_input:
            smoke = json.loads(Path(args.smoke_input).resolve().read_text(encoding="utf-8"))
            if isinstance(smoke, dict):
                texts = [str(value).strip() for value in smoke.get("semantic_smoke_texts", []) if str(value).strip()]
            elif isinstance(smoke, list):
                # Backward-compatible reader for the R3 temporary smoke file.
                texts = [str(row.get("normalized", "")).strip() for row in smoke
                         if isinstance(row, dict) and row.get("accepted_for_semantic_spert")]
        elif args.smoke_text:
            texts = [args.smoke_text]
        else:
            texts = [
                "on run up, left-hand magneto dropped 350 rpm.",
                "number 2 intake leaking.",
                "right-hand engine number 4 cylinder has low compression 20/80 psi.",
            ]
        if not texts:
            raise RuntimeError("Semantic SpERT compatibility check received no accepted hybrid text.")
        reports = []
        meaningful = 0
        for text in texts:
            pred = runner.predict(text)
            ok, reason = prediction_contract_ok(pred)
            if not ok:
                raise RuntimeError(f"Semantic SpERT output contract failed for {text!r}: {reason}")
            meaningful += int(bool(pred.get("entities")))
            reports.append({"text": text, "contract_valid": True, "prediction": pred})
        print(json.dumps(reports, indent=2), flush=True)
        print(
            "RULES_THEN_BYT5_SPERT_CHECK_OK "
            f"runtime_contract={len(reports)}/{len(texts)} nonempty_entity_outputs={meaningful}/{len(texts)}"
        )
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner, lock, model, actual))
    print(f"Normalized SpERT ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
