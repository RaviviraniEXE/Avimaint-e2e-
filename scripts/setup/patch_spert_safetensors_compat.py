from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UTIL = ROOT / "external" / "spert" / "spert" / "util.py"
PATCH_DIR = ROOT / "external" / "spert" / "AVIMAINT_PATCHES"
BACKUP = PATCH_DIR / "util.py.pre_safetensors_backup"
MANIFEST = PATCH_DIR / "safetensors_checkpoint_compat.json"
MARKER = "AVIMAINT_SAFETENSORS_COMPAT"

NEW_FUNCTION = r'''def check_version(config, model_class, model_path):
    # AVIMAINT_SAFETENSORS_COMPAT: Transformers may save model.safetensors
    # instead of the legacy pytorch_model.bin expected by upstream SpERT.
    if os.path.exists(model_path):
        if os.path.isdir(model_path):
            bin_path = os.path.join(model_path, 'pytorch_model.bin')
            safe_path = os.path.join(model_path, 'model.safetensors')
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location=torch.device('cpu'))
            elif os.path.exists(safe_path):
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:
                    raise ImportError(
                        "SpERT checkpoint is model.safetensors but the safetensors package "
                        "is unavailable in the avimaint-spert environment."
                    ) from exc
                state_dict = load_file(safe_path, device='cpu')
            else:
                raise FileNotFoundError(
                    "No supported SpERT checkpoint weights found in %s; expected "
                    "pytorch_model.bin or model.safetensors." % model_path
                )
        else:
            if model_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:
                    raise ImportError("safetensors is required to load %s" % model_path) from exc
                state_dict = load_file(model_path, device='cpu')
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        config_dict = config.to_dict()

        # version check
        loaded_version = config_dict.get('spert_version', '1.0')
        if 'rel_classifier.weight' in state_dict and loaded_version != model_class.VERSION:
            msg = ("Current SpERT version (%s) does not match the version of the loaded model (%s).\n"
                   % (model_class.VERSION, loaded_version))
            msg += "Use the code matching your version or train a new model."
            raise Exception(msg)
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not UTIL.exists():
        raise SystemExit(f"Official SpERT util.py not found: {UTIL}")
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    original = UTIL.read_text(encoding="utf-8")
    before_hash = sha256(UTIL)

    if MARKER in original:
        print("[SPERT CHECKPOINT PATCH] already applied")
        print(f"  util={UTIL}")
        return

    tree = ast.parse(original)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_version"), None)
    if node is None or not getattr(node, "end_lineno", None):
        raise SystemExit("Could not locate upstream check_version() in SpERT util.py")

    if not BACKUP.exists():
        BACKUP.write_text(original, encoding="utf-8")

    lines = original.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    replacement = NEW_FUNCTION.rstrip() + "\n"
    patched = "".join(lines[:start]) + replacement + "".join(lines[end:])
    UTIL.write_text(patched, encoding="utf-8")

    # Parse again to fail fast on accidental syntax errors.
    ast.parse(UTIL.read_text(encoding="utf-8"))
    after_hash = sha256(UTIL)
    manifest = {
        "patch": "SpERT safetensors checkpoint compatibility",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file": str(UTIL),
        "backup": str(BACKUP),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "behavioral_scope": "checkpoint file-format compatibility only; no model architecture, weights, data, hyperparameters, or evaluation policy changed",
        "supported_weights": ["pytorch_model.bin", "model.safetensors"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[SPERT CHECKPOINT PATCH] APPLIED")
    print(f"  util={UTIL}")
    print(f"  backup={BACKUP}")
    print(f"  manifest={MANIFEST}")
    print("  supported=pytorch_model.bin OR model.safetensors")


if __name__ == "__main__":
    main()
