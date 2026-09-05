
from __future__ import annotations
import argparse, json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

def ascii_upper(text: str) -> str:
    return str(text).translate(str.maketrans(
        "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ))

class ServiceError(RuntimeError):
    pass

class Runner:
    def __init__(self, lock_path: Path, device="cpu"):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise ServiceError("avimaint-normalization lacks torch/transformers") from exc

        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not bool(lock.get("byt5", {}).get("enabled", False)):
            raise ServiceError(
                "ByT5 runtime branch is disabled by runtime_model_lock.json: "
                + str(lock.get("byt5", {}).get("reason", "not resolved"))
            )
        self.model_path = Path(lock["byt5"]["model_path"]).expanduser().resolve()
        self.task_prefix = str(lock.get("byt5", {}).get("task_prefix", "") or "")
        if not self.model_path.is_dir():
            raise ServiceError(f"Locked ByT5 path missing: {self.model_path}")
        self.torch = torch
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        print(f"Loading LOCKED ByT5: {self.model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    @property
    def metadata(self):
        return {
            "status": "ready",
            "role": "operational_byt5_normalizer",
            "mode": "byt5_guarded_deployment",
            "model": str(self.model_path),
            "device": str(self.device),
            "model_input_case_adapter": "ascii_uppercase_raw_style",
            "task_prefix": self.task_prefix,
            "use_for_rq4": False,
            "use_for_rq5": False,
        }

    def normalize(self, text: str):
        original = str(text or "").strip()
        if not original:
            raise ServiceError("text is empty")
        model_input = ascii_upper(original)
        source_text = self.task_prefix + model_input
        batch = self.tokenizer(source_text, return_tensors="pt", truncation=True, max_length=512)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=256, num_beams=1, do_sample=False)
        normalized = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return {
            "original": original,
            "model_input": model_input,
            "task_prefix": self.task_prefix,
            "normalized": normalized,
            "method": "byt5_candidate",
            "model": str(self.model_path),
        }

def make_handler(runner):
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
                self.send_json(HTTPStatus.OK, runner.metadata)
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/normalize":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str):
                    raise ServiceError("Missing string field 'text'")
                self.send_json(HTTPStatus.OK, runner.normalize(text))
            except Exception as exc:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}")
    return Handler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke-text", default="L/H MAG EXC RPM DROP DURING RUN UP")
    args = ap.parse_args()
    runner = Runner(Path(args.lock).resolve(), args.device)
    print(json.dumps(runner.metadata, indent=2), flush=True)
    if args.check_only:
        print(json.dumps(runner.normalize(args.smoke_text), indent=2), flush=True)
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(f"Normalization service ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
