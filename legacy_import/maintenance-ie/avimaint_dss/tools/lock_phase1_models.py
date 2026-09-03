"""Safe runtime model discovery for AviMaint-DSS.

Important correction:
A locally available SpERT checkpoint is NOT automatically called
"normalized-matched" merely because it is the only final_model on disk.

The semantic SpERT branch is enabled only when one of the following is true:
1. AVIMAINT_NORMALIZED_SPERT_MODEL explicitly points to the intended model; or
2. frozen operational provenance clearly references the checkpoint/run; or
3. the checkpoint path itself explicitly identifies the normalized/selective
   representation.

Otherwise the checkpoint is recorded as an unverified candidate and the
dashboard safely uses the validated TRUE-RAW SpERT branch.

The same fail-safe principle is used for ByT5: ambiguity disables the optional
normalization branch instead of blocking installation or guessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

FROZEN_OPERATIONAL_MODEL_ID = "1f9c094789f591991ad26ca65ca1c36689b57ff0dd248cd30b4fc81b853b4ad9"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )


def weight_file(path: Path) -> Path:
    sf = path / "model.safetensors"
    return sf if sf.is_file() else path / "pytorch_model.bin"


def frozen_roots(project: Path) -> list[Path]:
    candidates = [
        project / "outputs" / "frozen" / "final_operational_ie_kg",
        project / "legacy_import" / "maintenance-ie" / "outputs" / "frozen" / "final_operational_ie_kg",
    ]
    return [p.resolve() for p in candidates if p.is_dir()]


def provenance_text(roots: Iterable[Path]) -> str:
    chunks = []
    allowed = {".json", ".yaml", ".yml", ".txt", ".csv", ".md", ".log"}
    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            try:
                if p.stat().st_size <= 5_000_000:
                    chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return "\n".join(chunks).lower()


def discover_spert(project: Path) -> list[Path]:
    roots = [
        project / "legacy_import" / "maintenance-ie" / "outputs" / "spert",
        project / "outputs" / "spert",
        project / "outputs" / "frozen" / "final_operational_ie_kg",
        project / "legacy_import" / "maintenance-ie" / "outputs" / "frozen" / "final_operational_ie_kg",
    ]
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for cfg in root.rglob("config.json"):
            d = cfg.parent.resolve()
            if not valid_model_dir(d):
                continue
            low = str(d).lower()
            if d.name.lower() == "final_model" or "spert" in low:
                found[str(d).lower()] = d
    return list(found.values())


def lock_semantic_spert(project: Path, types_path: Path) -> dict:
    override = os.getenv("AVIMAINT_NORMALIZED_SPERT_MODEL", "").strip()
    if override:
        d = Path(override).expanduser().resolve()
        if not valid_model_dir(d):
            return {
                "enabled": False,
                "verified_representation": False,
                "reason": f"Explicit AVIMAINT_NORMALIZED_SPERT_MODEL is invalid: {d}",
                "model_path": "",
                "weights_path": "",
                "weight_sha256": "",
                "types_path": str(types_path) if types_path.is_file() else "",
                "selection": "invalid_explicit_override",
                "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
            }
        wf = weight_file(d)
        return {
            "enabled": bool(types_path.is_file()),
            "verified_representation": True,
            "reason": "Explicitly selected through AVIMAINT_NORMALIZED_SPERT_MODEL.",
            "model_path": str(d),
            "weights_path": str(wf),
            "weight_sha256": sha256_file(wf),
            "types_path": str(types_path.resolve()) if types_path.is_file() else "",
            "selection": "explicit_override",
            "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
        }

    candidates = discover_spert(project)
    roots = frozen_roots(project)
    prov = provenance_text(roots)
    root_strings = [str(r).lower() for r in roots]

    rows = []
    print("Scanning semantic-SpERT candidates...")
    for i, d in enumerate(candidates, 1):
        wf = weight_file(d)
        digest = sha256_file(wf)
        low = str(d).lower()
        run_name = d.parent.name.lower()

        evidence = []
        verified = False
        score = 0

        if any(low.startswith(r) for r in root_strings):
            verified = True
            score += 1000
            evidence.append("checkpoint physically stored in frozen operational tree")

        if low in prov:
            verified = True
            score += 900
            evidence.append("exact checkpoint path referenced by frozen operational provenance")
        elif len(run_name) >= 8 and run_name in prov:
            verified = True
            score += 700
            evidence.append(f"run id '{d.parent.name}' referenced by frozen operational provenance")

        if "selective_byt5" in low or "selective-byt5" in low or "normalized" in low or "normaliz" in low:
            score += 100
            evidence.append("path name suggests normalization, but path naming alone is not accepted as proof")

        rows.append({
            "path": d,
            "weights": wf,
            "sha": digest,
            "verified": verified,
            "score": score,
            "evidence": evidence,
        })
        print(
            f"  [{i}/{len(candidates)}] verified={verified} score={score:4d} "
            f"sha={digest}  {d}",
            flush=True,
        )

    verified_rows = [r for r in rows if r["verified"]]
    if verified_rows:
        verified_rows.sort(key=lambda r: (-r["score"], str(r["path"])))
        best_score = verified_rows[0]["score"]
        best = [r for r in verified_rows if r["score"] == best_score]
        if len(best) == 1:
            r = best[0]
            return {
                "enabled": bool(types_path.is_file()),
                "verified_representation": True,
                "reason": "; ".join(r["evidence"]),
                "model_path": str(r["path"]),
                "weights_path": str(r["weights"]),
                "weight_sha256": r["sha"],
                "types_path": str(types_path.resolve()) if types_path.is_file() else "",
                "selection": "verified_provenance",
                "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
            }

        return {
            "enabled": False,
            "verified_representation": False,
            "reason": "More than one equally strong verified semantic SpERT candidate exists; no model was guessed.",
            "model_path": "",
            "weights_path": "",
            "weight_sha256": "",
            "types_path": str(types_path.resolve()) if types_path.is_file() else "",
            "selection": "ambiguous_verified_candidates",
            "candidates": [
                {
                    "model_path": str(r["path"]),
                    "weight_sha256": r["sha"],
                    "evidence": r["evidence"],
                }
                for r in best
            ],
            "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
        }

    # Critical safety correction: a unique candidate is recorded, but not
    # automatically promoted to "normalized-matched".
    candidate_note = ""
    candidate_path = ""
    candidate_sha = ""
    if len(rows) == 1:
        candidate_note = (
            "One local SpERT final_model exists, but no frozen/representation "
            "provenance proves that it is the normalized-matched checkpoint. "
            "The operational semantic branch is therefore disabled."
        )
        candidate_path = str(rows[0]["path"])
        candidate_sha = rows[0]["sha"]
    elif len(rows) > 1:
        candidate_note = (
            "Local SpERT candidates exist, but none is proven to be the normalized "
            "representation checkpoint. No model was guessed."
        )
    else:
        candidate_note = "No local normalized semantic SpERT checkpoint was found."

    return {
        "enabled": False,
        "verified_representation": False,
        "reason": candidate_note,
        "model_path": "",
        "weights_path": "",
        "weight_sha256": "",
        "types_path": str(types_path.resolve()) if types_path.is_file() else "",
        "selection": "safe_raw_fallback",
        "unverified_candidate_path": candidate_path,
        "unverified_candidate_weight_sha256": candidate_sha,
        "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
    }


def trainer_best(state: Path) -> Path | None:
    try:
        obj = json.loads(state.read_text(encoding="utf-8"))
        best = obj.get("best_model_checkpoint")
        if not best:
            return None
        p = Path(best)
        if not p.is_absolute():
            p = (state.parent / p).resolve()
        return p.resolve() if valid_model_dir(p) else None
    except Exception:
        return None



def infer_byt5_task_prefix(project: Path) -> str:
    try:
        import yaml
    except Exception:
        return ""
    candidates=[project/"configs"/"normalization"/"byt5_gold.yaml", project/"configs"/"normalization"/"byt5_gold.yml"]
    def walk(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if str(k).lower() in {"input_prefix","task_prefix","source_prefix"} and isinstance(v,str):
                    return v
                x=walk(v)
                if x is not None: return x
        elif isinstance(obj,list):
            for v in obj:
                x=walk(v)
                if x is not None: return x
        return None
    for path in candidates:
        if not path.is_file(): continue
        try: obj=yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception: continue
        x=walk(obj)
        if isinstance(x,str): return x
    return ""

def lock_byt5(project: Path) -> dict:
    override = os.getenv("AVIMAINT_BYT5_MODEL", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not valid_model_dir(p):
            return {
                "enabled": False,
                "reason": f"Explicit AVIMAINT_BYT5_MODEL is invalid: {p}",
                "model_path": "",
                "source": "invalid_explicit_override",
            }
        return {
            "enabled": True,
            "reason": "Explicitly selected through AVIMAINT_BYT5_MODEL.",
            "model_path": str(p),
            "source": "AVIMAINT_BYT5_MODEL",
            "task_prefix": infer_byt5_task_prefix(project),
        }

    rows: dict[str, dict] = {}
    for state in project.rglob("trainer_state.json"):
        low = str(state).lower()
        if ".git" in low or "__pycache__" in low:
            continue
        best = trainer_best(state)
        if best is None:
            continue
        bl = str(best).lower()
        if "byt5" not in bl and "byt5" not in low:
            continue

        score = 0
        if "byt5" in bl:
            score += 30
        if "gold" in bl or "gold" in low:
            score += 30
        if "normalization" in bl or "normalization" in low:
            score += 10

        key = str(best).lower()
        row = {
            "model_path": str(best),
            "source": str(state.resolve()),
            "score": score,
        }
        if key not in rows or score > rows[key]["score"]:
            rows[key] = row

    if not rows:
        return {
            "enabled": False,
            "reason": (
                "No unambiguous trained ByT5 checkpoint was resolved from trainer_state.json. "
                "Set AVIMAINT_BYT5_MODEL to enable the optional live normalization branch."
            ),
            "model_path": "",
            "source": "not_resolved",
        }

    ranked = sorted(rows.values(), key=lambda x: (-x["score"], x["model_path"]))
    top_score = ranked[0]["score"]
    top = [x for x in ranked if x["score"] == top_score]
    if len(top) != 1:
        return {
            "enabled": False,
            "reason": (
                "Multiple equally authoritative ByT5 checkpoints were found. "
                "Set AVIMAINT_BYT5_MODEL to the exact checkpoint; no model was guessed."
            ),
            "model_path": "",
            "source": "ambiguous",
            "candidates": top,
        }

    return {
        "enabled": True,
        "reason": "Unique highest-authority ByT5 gold checkpoint from trainer_state.json.",
        "model_path": top[0]["model_path"],
        "source": top[0]["source"],
        "task_prefix": infer_byt5_task_prefix(project),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    project = Path(args.project_root).expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root not found: {project}")

    maint_ie = project / "legacy_import" / "maintenance-ie"
    types = maint_ie / "outputs" / "spert" / "avimaint_types.json"

    semantic = lock_semantic_spert(project, types)
    byt5 = lock_byt5(project)

    lock = {
        "schema": "avimaint-runtime-model-lock-v4",
        "project_root": str(project),
        "semantic_representation": "normalized_operational",
        "normalized_spert": semantic,
        "byt5": byt5,
        "research_branch": {
            "rq4": "true_raw_structure",
            "rq5_recalibrated": False,
        },
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print("\nPHASE1_MODEL_LOCK_V4_OK")
    print("ByT5 enabled:", byt5["enabled"])
    print("ByT5 reason:", byt5["reason"])
    if byt5.get("model_path"):
        print("ByT5 model:", byt5["model_path"])
    print("Normalized semantic SpERT enabled:", semantic["enabled"])
    print("Normalized semantic SpERT verified:", semantic["verified_representation"])
    print("Semantic reason:", semantic["reason"])
    if semantic.get("model_path"):
        print("Semantic model:", semantic["model_path"])
        print("Semantic weight SHA:", semantic["weight_sha256"])
    elif semantic.get("unverified_candidate_path"):
        print("Unverified candidate kept DISABLED:", semantic["unverified_candidate_path"])
        print("Unverified candidate weight SHA:", semantic["unverified_candidate_weight_sha256"])
    print("Lock:", out)


if __name__ == "__main__":
    main()
