#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$ROOT/external/spert"
REMOTE="https://github.com/lavis-nlp/spert.git"
REF="${1:-master}"
if [[ -d "$TARGET/.git" ]]; then
  [[ "$(git -C "$TARGET" remote get-url origin)" == "$REMOTE" ]] || {
    echo "external/spert does not point to the official upstream" >&2; exit 2;
  }
  [[ -z "$(git -C "$TARGET" status --porcelain)" ]] || {
    echo "external/spert has local modifications; refusing to overwrite them" >&2; exit 3;
  }
else
  mkdir -p "$ROOT/external"
  if [[ -d "$TARGET" ]]; then
    find "$TARGET" -mindepth 1 -maxdepth 1 ! -name README.md -print -quit | grep -q . && {
      echo "external/spert is not empty" >&2; exit 4;
    }
    rm -f "$TARGET/README.md"
  fi
  git clone --origin origin --branch master --single-branch "$REMOTE" "$TARGET"
fi
if [[ "$REF" == master ]]; then git -C "$TARGET" checkout master; else git -C "$TARGET" checkout --detach "$REF"; fi
COMMIT="$(git -C "$TARGET" rev-parse HEAD)"
python - "$TARGET/UPSTREAM_PROVENANCE.json" "$REMOTE" "$REF" "$COMMIT" <<'PY'
import datetime, json, pathlib, sys
path, remote, ref, commit = sys.argv[1:]
payload = {"upstream": remote, "requested_ref": ref, "resolved_commit": commit,
           "repository_archived_utc": "2025-04-02",
           "cloned_or_verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "source_modified_by_avimaint": False}
pathlib.Path(path).write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
PY
echo "Official SpERT ready at commit $COMMIT"
