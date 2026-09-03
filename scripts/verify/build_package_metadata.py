"""Regenerate the portable project tree and SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "v4_5",
}
EXCLUDED_PREFIXES = {
    Path("outputs/runs/rq5_leakage_safe_case_retrieval"),
    Path("outputs/runs/crosscut_robustness"),
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if any(relative == prefix or prefix in relative.parents for prefix in EXCLUDED_PREFIXES):
        return False
    return path.name not in {"MANIFEST.sha256", "PROJECT_TREE.txt"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    tree = "\n".join(path.relative_to(ROOT).as_posix() for path in files) + "\n"
    (ROOT / "PROJECT_TREE.txt").write_text(tree, encoding="utf-8")
    manifest_rows = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    manifest_rows.append(f"{digest(ROOT / 'PROJECT_TREE.txt')}  PROJECT_TREE.txt")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"Metadata generated for {len(files) + 1} files")


if __name__ == "__main__":
    main()
