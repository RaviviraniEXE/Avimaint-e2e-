from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from core.hybrid_normalization import ExpertRuleNormalizer, validate_hybrid_candidate


class ServiceError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_weight(model_path: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    raise ServiceError(f"Locked ByT5 model has no supported weight file: {model_path}")


class Runner:
    def __init__(self, lock_path: Path, device="cpu"):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise ServiceError("avimaint-normalization lacks torch/transformers") from exc

        self.lock_path = lock_path.resolve()
        self.lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
        byt5 = self.lock.get("byt5", {})
        rules = self.lock.get("normalization_rules", {})
        representation = str(self.lock.get("semantic_representation", ""))
        if representation != "rules_then_byt5_guarded_operational":
            raise ServiceError(f"Unsupported locked representation: {representation or 'missing'}")
        if not bool(byt5.get("enabled", False)):
            raise ServiceError("ByT5 runtime branch is disabled: " + str(byt5.get("reason", "not resolved")))
        if not bool(rules.get("enabled", False)):
            raise ServiceError("Expert rule stage is disabled: " + str(rules.get("reason", "not resolved")))

        self.model_path = Path(byt5["model_path"]).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ServiceError(f"Locked ByT5 path missing: {self.model_path}")
        weight = model_weight(self.model_path)
        actual_weight_sha = digest(weight)
        if actual_weight_sha.lower() != str(byt5.get("weight_sha256", "")).lower():
            raise ServiceError("ByT5 weight SHA-256 does not match runtime_model_lock.json")

        resource_dir = Path(rules["resource_dir"]).expanduser().resolve()
        self.rule_normalizer = ExpertRuleNormalizer(resource_dir)
        for key, path in self.rule_normalizer.files.items():
            expected = str(rules.get("sha256", {}).get(key, ""))
            if not expected or digest(path).lower() != expected.lower():
                raise ServiceError(f"Normalization rule resource SHA-256 mismatch: {key}")

        self.task_prefix = str(byt5.get("task_prefix", "") or "")
        self.max_source_length = int(byt5.get("max_source_length", 128))
        self.max_target_length = int(byt5.get("max_target_length", 128))
        self.num_beams = int(byt5.get("num_beams", 1))
        if self.num_beams != 1:
            raise ServiceError("Matched System-D deployment requires deterministic greedy decoding (num_beams=1)")

        self.torch = torch
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        print(f"Loading LOCKED rules-then-ByT5 model: {self.model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.weight_sha256 = actual_weight_sha

    @property
    def metadata(self):
        return {
            "status": "ready",
            "role": "operational_rules_then_byt5_normalizer",
            "mode": "rules_then_byt5_guarded",
            "representation": "rules_then_byt5_guarded_operational",
            "model": str(self.model_path),
            "model_weight_sha256": self.weight_sha256,
            "device": str(self.device),
            "model_input_case_adapter": "expert_rules_lowercase",
            "task_prefix": self.task_prefix,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "num_beams": self.num_beams,
            "decoding_strategy": "greedy_deterministic",
            "guard": "protected_values_directions_identifiers_anchors",
            "use_for_rq4": False,
            "use_for_rq5": False,
        }

    def normalize(self, text: str):
        original = str(text or "").strip()
        if not original:
            raise ServiceError("text is empty")
        rule_result = self.rule_normalizer.normalize(original)
        model_input = self.task_prefix + rule_result.normalized
        batch = self.tokenizer(
            model_input, return_tensors="pt", truncation=True, max_length=self.max_source_length
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.torch.no_grad():
            output = self.model.generate(
                **batch,
                max_length=self.max_target_length,
                num_beams=self.num_beams,
                do_sample=False,
            )
        candidate = self.tokenizer.decode(output[0], skip_special_tokens=True)
        candidate = re.sub(r"\s+", " ", candidate).strip().lower()
        accepted, warnings = validate_hybrid_candidate(original, rule_result.normalized, candidate)
        return {
            "original": original,
            "rule_normalized": rule_result.normalized,
            "model_input": model_input,
            "candidate_normalized": candidate,
            "normalized": candidate if accepted else original,
            "accepted_for_semantic_spert": accepted,
            "method": "rules_then_byt5_guarded" if accepted else "guard_rejected_raw_fallback",
            "warnings": warnings,
            "rule_expansions": rule_result.expansions,
            "model": str(self.model_path),
            "representation": "rules_then_byt5_guarded_operational",
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
                size = int(self.headers.get("Content-Length", "0"))
                obj = json.loads(self.rfile.read(size).decode("utf-8"))
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str):
                    raise ServiceError("Missing string field 'text'")
                self.send_json(HTTPStatus.OK, runner.normalize(text))
            except Exception as exc:
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}")

    return Handler


SAFETY_SMOKE_CASES = [
    "ON RUN UP, L/H MAG DROPPED 350 RPM.",
    "#2 INTAKE LEAKING.",
    "R/H ENG #4 CYL HAS LOW COMPRESSION (20/80 PSI).",
]

# These are corpus-grounded problem forms from the recorded System-D export.
# They deliberately avoid protected numeric/directional values: this checks
# that the neural path can produce at least one usable result, while the
# separate safety suite checks that unsafe rewrites are rejected correctly.
VIABILITY_SMOKE_CASES = [
    "ROCKER BOX COVER SCREWS LOOSE (ALL CYL).",
    "INDUCTION TUBE HOSE CLAMPS LOOSE (ALL CYL).",
    "INTAKE GASKET LEAKING.",
]

RECORDED_SYSTEM_D_SMOKE_TEXTS = [
    "rocker box cover screws loose all cylinder.",
    "induction tube hose clamps loose all cylinder.",
    "number 3 intake is leaking.",
]


def safe_result(result: dict) -> bool:
    """A smoke result is safe when accepted or explicitly raw-fallbacked."""
    if bool(result.get("accepted_for_semantic_spert")):
        return not result.get("warnings") and result.get("normalized") == result.get("candidate_normalized")
    return bool(result.get("warnings")) and result.get("normalized") == result.get("original")


def smoke_report(runner, single_text: str = "") -> dict:
    if single_text:
        viability = [runner.normalize(single_text)]
        safety = []
    else:
        safety = [runner.normalize(case) for case in SAFETY_SMOKE_CASES]
        viability = [runner.normalize(case) for case in VIABILITY_SMOKE_CASES]
    accepted = [row for row in safety + viability if row["accepted_for_semantic_spert"]]
    semantic_texts = list(dict.fromkeys(
        [str(row["normalized"]) for row in accepted] + RECORDED_SYSTEM_D_SMOKE_TEXTS
    ))
    return {
        "schema": "avimaint-v721-r4-normalization-smoke-v1",
        "decoder": "greedy_deterministic",
        "safety_results": safety,
        "viability_results": viability,
        "accepted": accepted,
        "semantic_smoke_texts": semantic_texts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-text", default="")
    parser.add_argument("--smoke-output", default="")
    args = parser.parse_args()
    runner = Runner(Path(args.lock).resolve(), args.device)
    print(json.dumps(runner.metadata, indent=2), flush=True)
    if args.check_only:
        report = smoke_report(runner, args.smoke_text)
        print(json.dumps(report, indent=2), flush=True)
        all_results = report["safety_results"] + report["viability_results"]
        unsafe = [row for row in all_results if not safe_result(row)]
        if unsafe:
            raise RuntimeError(f"Hybrid safety contract failed for {len(unsafe)} smoke result(s).")
        accepted = report["accepted"]
        if not accepted:
            raise RuntimeError(
                "Hybrid viability check failed: the model loaded but no corpus-grounded candidate passed the guard."
            )
        if args.smoke_output:
            output_path = Path(args.smoke_output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            "RULES_THEN_BYT5_GUARDED_CHECK_OK "
            f"accepted={len(accepted)}/{len(all_results)} safety_contract={len(all_results) - len(unsafe)}/{len(all_results)}"
        )
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner))
    print(f"Rules-then-ByT5 service ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
