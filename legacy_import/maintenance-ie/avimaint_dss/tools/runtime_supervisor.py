"""Single-terminal supervisor for the AviMaint-DSS V7.2.1 runtime.

Each ML service stays in its own conda environment/process, while this one
controller owns startup, identity checks, logs, browser opening and shutdown.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DASHBOARD = Path(__file__).resolve().parents[1]
MAINT_IE = DASHBOARD.parent
REPOSITORY = DASHBOARD.parents[2]
LOCK_PATH = DASHBOARD / "runtime_model_lock.json"
LOG_DIR = DASHBOARD / "runtime_logs"


def get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if response.status == 200 and isinstance(data, dict) else None
    except Exception:
        return None


def raw_ok(meta: dict | None) -> bool:
    return bool(
        meta and meta.get("status") == "ready"
        and int(meta.get("entity_types", 0)) == 9
        and int(meta.get("relation_types", 0)) == 11
        and meta.get("query_case_normalization") == "none_true_raw"
    )


def normalizer_ok(meta: dict | None, lock: dict) -> bool:
    return bool(
        meta and meta.get("status") == "ready"
        and meta.get("role") == "operational_rules_then_byt5_normalizer"
        and meta.get("representation") == "rules_then_byt5_guarded_operational"
        and meta.get("decoding_strategy") == "greedy_deterministic"
        and int(meta.get("num_beams", 0)) == 1
        and meta.get("model") == lock.get("byt5", {}).get("model_path")
        and meta.get("model_weight_sha256") == lock.get("byt5", {}).get("weight_sha256")
    )


def semantic_ok(meta: dict | None, lock: dict) -> bool:
    return bool(
        meta and meta.get("status") == "ready"
        and meta.get("role") == "rules_then_byt5_semantic_spert"
        and meta.get("representation") == "rules_then_byt5_guarded_operational"
        and meta.get("weights_sha256") == lock.get("normalized_spert", {}).get("weight_sha256")
    )


def api_ok(meta: dict | None) -> bool:
    return bool(
        meta and meta.get("status") == "ready"
        and meta.get("api_version") == "1.0.2"
        and meta.get("rq4_base") == "structure"
        and meta.get("candidate_split") == "train"
        and meta.get("raw_spert", {}).get("ready") is True
        and meta.get("rq5_calibrator", {}).get("ready") is True
        and meta.get("frontend", {}).get("ready") is True
        and meta.get("frontend", {}).get("version") == "5.0.1"
    )


def conda_executable() -> str:
    candidates = [os.environ.get("CONDA_EXE", ""), shutil.which("conda.exe") or "", shutil.which("conda") or ""]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    raise RuntimeError("Conda executable not found. Start FINAL_12 from Anaconda Prompt.")


def conda_command(environment: str, arguments: list[str]) -> list[str]:
    executable = conda_executable()
    command = [executable, "run", "--no-capture-output", "-n", environment, *arguments]
    if executable.lower().endswith((".bat", ".cmd")):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", "call", *command]
    return command


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_path: Path
    stream: object


class Supervisor:
    def __init__(self, lock: dict):
        self.lock = lock
        self.processes: list[ManagedProcess] = []
        self.stop_requested = False

    def request_stop(self, *_):
        self.stop_requested = True

    def start(self, name: str, environment: str, arguments: list[str]) -> ManagedProcess:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{name}.log"
        stream = log_path.open("w", encoding="utf-8", buffering=1)
        command = [sys.executable, *arguments] if environment == "__current__" else conda_command(environment, arguments)
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            command,
            cwd=str(DASHBOARD),
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        managed = ManagedProcess(name, process, log_path, stream)
        self.processes.append(managed)
        print(f"  starting {name} (PID {process.pid}); log: {log_path.name}", flush=True)
        return managed

    @staticmethod
    def tail(path: Path, lines: int = 18) -> str:
        try:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except Exception:
            return "(log unavailable)"

    def ensure(
        self,
        name: str,
        url: str,
        validator: Callable[[dict | None], bool],
        environment: str,
        arguments: list[str],
        timeout: int,
        required: bool,
        request_timeout: float = 3.0,
    ) -> bool:
        current = get_json(url, request_timeout)
        if validator(current):
            print(f"  ready   {name} (reusing validated listener)", flush=True)
            return True
        if current is not None:
            message = f"{name} port is occupied by a service with the wrong identity; it was not terminated."
            if required:
                raise RuntimeError(message)
            print(f"  warning {message}", flush=True)
            return False
        managed = self.start(name, environment, arguments)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.stop_requested:
            if managed.process.poll() is not None:
                break
            if validator(get_json(url, request_timeout)):
                print(f"  ready   {name}", flush=True)
                return True
            time.sleep(2)
        message = f"{name} failed its health/identity check.\n{self.tail(managed.log_path)}"
        if required:
            raise RuntimeError(message)
        print(f"  warning {message}", flush=True)
        return False

    def stop(self):
        print("\nStopping services started by this launcher...", flush=True)
        for item in reversed(self.processes):
            if item.process.poll() is not None:
                item.stream.close()
                continue
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(item.process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                    )
                else:
                    item.process.terminate()
                    item.process.wait(timeout=10)
            except Exception:
                try:
                    item.process.kill()
                except Exception:
                    pass
            item.stream.close()


def load_lock() -> dict:
    if not LOCK_PATH.is_file():
        raise RuntimeError("runtime_model_lock.json is missing. Apply the V7.2.1 hotfix first.")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("schema") != "avimaint-runtime-model-lock-v5":
        raise RuntimeError("runtime_model_lock.json is not the V7.2.1 model lock (schema v5).")
    if lock.get("runtime_revision") != "v7.2.1-r4":
        raise RuntimeError("runtime_model_lock.json is not the corrected V7.2.1 R4 runtime lock.")
    return lock


def status_line(label: str, ready: bool, optional: bool = False):
    state = "READY" if ready else ("FALLBACK" if optional else "FAILED")
    print(f"  {label:<28} {state}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--legacy-streamlit", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    lock = load_lock()
    if args.status_only:
        status_line("Validated TRUE-RAW SpERT", raw_ok(get_json("http://127.0.0.1:8765/health")))
        status_line("Rules -> ByT5", normalizer_ok(get_json("http://127.0.0.1:8766/health"), lock), True)
        status_line("Hybrid semantic SpERT", semantic_ok(get_json("http://127.0.0.1:8767/health"), lock), True)
        status_line("FastAPI + frontend", api_ok(get_json("http://127.0.0.1:8780/api/v1/health")))
        return 0

    supervisor = Supervisor(lock)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, supervisor.request_stop)

    print("=" * 68)
    print(" AviMaint-DSS V7.2.1 R4 - single launcher")
    print(" Required: TRUE-RAW SpERT + RQ5 calibrator + API/frontend")
    print(" Optional: expert rules -> ByT5 -> matched semantic SpERT")
    print(" Reranker and calibrator load inside the FastAPI process")
    print("=" * 68)
    try:
        raw_ready = supervisor.ensure(
            "validated_raw_spert", "http://127.0.0.1:8765/health", raw_ok,
            "avimaint-spert",
            ["python", "-u", "services/spert_query_service.py", "--project-root", str(MAINT_IE),
             "--spert-root", str(REPOSITORY / "external" / "spert")],
            timeout=180, required=True,
        )

        byt5_enabled = bool(lock.get("byt5", {}).get("enabled") and lock.get("normalization_rules", {}).get("enabled"))
        normalization_ready = False
        if byt5_enabled:
            normalization_ready = supervisor.ensure(
                "rules_then_byt5", "http://127.0.0.1:8766/health",
                lambda meta: normalizer_ok(meta, lock), "avimaint-normalization",
                ["python", "-u", "services/normalization_query_service.py", "--lock", str(LOCK_PATH),
                 "--host", "127.0.0.1", "--port", "8766", "--device", "cpu"],
                timeout=240, required=False,
            )
        else:
            print("  fallback rules -> ByT5 disabled by model lock", flush=True)

        semantic_enabled = bool(
            normalization_ready
            and lock.get("normalized_spert", {}).get("enabled")
            and lock.get("normalized_spert", {}).get("verified_representation")
        )
        semantic_ready = False
        if semantic_enabled:
            semantic_ready = supervisor.ensure(
                "hybrid_semantic_spert", "http://127.0.0.1:8767/health",
                lambda meta: semantic_ok(meta, lock), "spert",
                ["python", "-u", "services/normalized_spert_query_service.py",
                 "--project-root", str(MAINT_IE), "--lock", str(LOCK_PATH),
                 "--host", "127.0.0.1", "--port", "8767", "--cpu"],
                timeout=240, required=False,
            )
        else:
            print("  fallback hybrid semantic SpERT disabled/unavailable; raw branch remains active", flush=True)

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        api_ready = supervisor.ensure(
            "api_frontend", "http://127.0.0.1:8780/api/v1/health", api_ok,
            "__current__",
            ["-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8780"],
            timeout=300, required=True, request_timeout=60.0,
        )

        if args.legacy_streamlit:
            supervisor.start(
                "legacy_streamlit", "__current__",
                ["-m", "streamlit", "run", "app.py", "--server.port", "8502"],
            )

        api_meta = get_json("http://127.0.0.1:8780/api/v1/health", timeout=10) or {}
        print("\nRuntime summary")
        status_line("Validated TRUE-RAW SpERT", raw_ready)
        status_line("Rules -> ByT5", normalization_ready, True)
        status_line("Hybrid semantic SpERT", semantic_ready, True)
        status_line("RQ5 calibrator", bool(api_meta.get("rq5_calibrator", {}).get("ready")))
        status_line("Presentation reranker", bool(api_meta.get("reranker", {}).get("ready")), True)
        status_line("Phase 5 API/frontend", api_ready)
        print("\nDashboard: http://127.0.0.1:8780/")
        print("API docs : http://127.0.0.1:8780/docs")
        print(f"Logs     : {LOG_DIR}")
        print("Press Ctrl+C to stop only the processes started by this launcher.\n")
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:8780/")

        while not supervisor.stop_requested:
            for item in supervisor.processes:
                if item.process.poll() is not None and item.name in {"validated_raw_spert", "api_frontend"}:
                    raise RuntimeError(
                        f"Required service {item.name} stopped unexpectedly.\n{supervisor.tail(item.log_path)}"
                    )
            time.sleep(2)
        return 0
    except Exception as exc:
        print(f"\nSTARTUP FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        supervisor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
