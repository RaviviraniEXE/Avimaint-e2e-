from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from spert_runtime import (  # noqa: E402
    SpERTRunner,
    SpERTRuntimeError,
    resolve_runtime_paths,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def make_handler(runner: SpERTRunner, api_token: str = ""):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AviMaintSpERT/1.0"

        def _authorised(self) -> bool:
            if not api_token:
                return True
            supplied = self.headers.get("Authorization", "")
            return supplied == f"Bearer {api_token}"

        def _send(self, status: int, value: Any) -> None:
            payload = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(HTTPStatus.OK, runner.metadata)
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/predict":
                self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            if not self._authorised():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > 100_000:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Invalid request size"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str):
                    raise SpERTRuntimeError(
                        "Request JSON must contain a string field named 'text'."
                    )
                result = runner.predict(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"Invalid JSON request: {exc}"},
                )
                return
            except SpERTRuntimeError as exc:
                self._send(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": str(exc)},
                )
                return
            # This is the HTTP process boundary. Do not expose model/runtime
            # internals in an unexpected 500 response.
            except Exception:  # noqa: BLE001
                self._send(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Unexpected inference failure"},
                )
                return
            self._send(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stdout.write(
                f"{self.log_date_time_string()} {self.client_address[0]} {format % args}\n"
            )

    return Handler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load AviMaint SpERT once and expose query inference."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--spert-root")
    parser.add_argument("--model-path")
    parser.add_argument("--types-path")
    parser.add_argument("--host", default=os.getenv("AVIMAINT_SPERT_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("AVIMAINT_SPERT_PORT", "8765")),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-text",
        default="Number 4 cylinder recorded low compression of 20/80 psi.",
    )
    parser.add_argument("--relation-threshold", type=float, default=0.4)
    parser.add_argument("--max-span-size", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=1000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        paths = resolve_runtime_paths(
            args.project_root,
            spert_root=args.spert_root,
            model_path=args.model_path,
            types_path=args.types_path,
        )
        print("Loading final AviMaint SpERT checkpoint...")
        runner = SpERTRunner(
            paths,
            rel_filter_threshold=args.relation_threshold,
            max_span_size=args.max_span_size,
            max_pairs=args.max_pairs,
            force_cpu=args.cpu,
        )
        print(json.dumps(runner.metadata, indent=2))
        if args.check_only:
            print(json.dumps(runner.predict(args.smoke_text), indent=2))
            print("SPERT CHECK COMPLETED SUCCESSFULLY.")
            return 0
        server = ThreadingHTTPServer(
            (args.host, args.port),
            make_handler(
                runner,
                api_token=os.getenv("AVIMAINT_SPERT_TOKEN", ""),
            ),
        )
        print(f"SpERT query service ready at http://{args.host}:{args.port}")
        print("Keep this window open. Press Ctrl+C to stop the service.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except SpERTRuntimeError as exc:
        print()
        print("SPERT SERVICE COULD NOT START")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

