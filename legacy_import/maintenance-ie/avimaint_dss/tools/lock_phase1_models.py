"""Create the fail-closed V7.2.1 runtime model lock.

The operational semantic chain is enabled only as one matched unit:

    bundled expert rules -> gold ByT5 -> rules_then_byt5 SpERT

The independently validated TRUE-RAW SpERT branch used by frozen RQ4/RQ5 is
outside this lock and is never replaced by the optional chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


REPRESENTATION = "rules_then_byt5_guarded_operational"
REGISTRY_KEY = "rules_then_byt5"
FROZEN_OPERATIONAL_MODEL_ID = "1f9c094789f591991ad26ca65ca1c36689b57ff0dd248cd30b4fc81b853b4ad9"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_model_dir(path: Path) -> bool:
    return bool(
        path.is_dir()
        and (path / "config.json").is_file()
        and ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file())
    )


def weight_file(path: Path) -> Path:
    safetensors = path / "model.safetensors"
    return safetensors if safetensors.is_file() else path / "pytorch_model.bin"


def resolve_relative(project: Path, value: str, anchors: tuple[Path, ...] = ()) -> Path:
    raw = Path(str(value).strip().replace("\\", os.sep))
    if raw.is_absolute():
        return raw.resolve()
    candidates = [project / raw, *[anchor / raw for anchor in anchors]]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def lock_rules(dashboard_root: Path) -> dict:
    resource_dir = (dashboard_root / "data" / "normalization_rules").resolve()
    names = {
        "abbreviations": resource_dir / "abbreviations.csv",
        "misspellings": resource_dir / "misspellings.csv",
        "unexpanded": resource_dir / "unexpanded.csv",
    }
    missing = [str(path) for path in names.values() if not path.is_file()]
    if missing:
        return {
            "enabled": False,
            "reason": "Required expert rule resource(s) missing: " + ", ".join(missing),
            "resource_dir": str(resource_dir),
            "sha256": {},
        }
    return {
        "enabled": True,
        "reason": "Bundled authoritative expert rule resources verified by SHA-256.",
        "source": "System-B expert resources",
        "resource_dir": str(resource_dir),
        "sha256": {key: sha256_file(path) for key, path in names.items()},
        "numbers": "digits",
        "lowercase": True,
    }


def trainer_best(project: Path, state: Path) -> Path | None:
    try:
        obj = json.loads(state.read_text(encoding="utf-8"))
        best = str(obj.get("best_model_checkpoint", "") or "").strip()
        if not best:
            return None
        # Hugging Face may store a repository-relative path. Resolving it under
        # state.parent caused the old duplicated outputs/.../outputs/... bug.
        candidate = resolve_relative(project, best, (state.parent, state.parent.parent))
        return candidate if valid_model_dir(candidate) else None
    except Exception:
        return None


def infer_byt5_task_prefix(project: Path) -> tuple[str, str, str]:
    """Resolve a recorded training prefix, if one exists, and hash its source."""
    try:
        import yaml
    except Exception:
        return "", "", ""

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in {"input_prefix", "task_prefix", "source_prefix"} and isinstance(value, str):
                    return value
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found is not None:
                    return found
        return None

    for path in (
        project / "configs" / "normalization" / "byt5_gold.yaml",
        project / "configs" / "normalization" / "byt5_gold.yml",
    ):
        if not path.is_file():
            continue
        try:
            value = walk(yaml.safe_load(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if isinstance(value, str):
            return value, str(path.resolve()), sha256_file(path)
    return "", "", ""


def lock_byt5(project: Path) -> dict:
    references: dict[str, dict] = {}
    for state in project.rglob("trainer_state.json"):
        low = str(state).lower()
        if any(marker in low for marker in (".git", "__pycache__", "node_modules")):
            continue
        best = trainer_best(project, state)
        if best is None or "byt5" not in (str(best) + str(state)).lower():
            continue
        key = os.path.normcase(str(best))
        row = references.setdefault(key, {"path": best, "states": []})
        row["states"].append(str(state.resolve()))

    override = os.getenv("AVIMAINT_BYT5_MODEL", "").strip()
    if override:
        selected = Path(override).expanduser().resolve()
        if not valid_model_dir(selected):
            return {
                "enabled": False, "reason": f"AVIMAINT_BYT5_MODEL is invalid: {selected}",
                "model_path": "", "source": "invalid_explicit_override",
            }
        source = "AVIMAINT_BYT5_MODEL"
        states = references.get(os.path.normcase(str(selected)), {}).get("states", [])
    elif len(references) == 1:
        row = next(iter(references.values()))
        selected = row["path"]
        states = row["states"]
        source = "unanimous_trainer_state_best"
    else:
        return {
            "enabled": False,
            "reason": (
                "No unique ByT5 best checkpoint was proven by trainer_state.json. "
                "Set AVIMAINT_BYT5_MODEL to the exact trained checkpoint."
                if not references else
                "Trainer states resolve to multiple ByT5 best checkpoints; no checkpoint was guessed."
            ),
            "model_path": "",
            "source": "not_resolved" if not references else "ambiguous",
            "candidates": [str(row["path"]) for row in references.values()],
        }

    weight = weight_file(selected)
    task_prefix, prefix_source, prefix_source_sha = infer_byt5_task_prefix(project)
    return {
        "enabled": True,
        "reason": "Exact gold ByT5 checkpoint selected and hashed.",
        "model_path": str(selected),
        "weights_path": str(weight),
        "weight_sha256": sha256_file(weight),
        "config_sha256": sha256_file(selected / "config.json"),
        "source": source,
        "trainer_states": states,
        "task_prefix": task_prefix,
        "task_prefix_source": prefix_source,
        "task_prefix_source_sha256": prefix_source_sha,
        "max_source_length": 128,
        "max_target_length": 128,
        "num_beams": 1,
        "decoding_strategy": "greedy_deterministic",
        "input_stage": "expert_rules_lowercase",
    }


def parse_types(path: Path) -> tuple[int, int]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        entities = obj.get("entities", {})
        relations = obj.get("relations", {})
        return len(entities), len(relations)
    except Exception:
        return 0, 0


def resolve_types(project: Path, maint_ie: Path, entry: dict, export_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for key in ("types_path", "type_path", "types"):
        if entry.get(key):
            candidates.append(resolve_relative(project, entry[key], (maint_ie, export_dir)))
    config_value = entry.get("config_path")
    if config_value:
        config_path = resolve_relative(project, config_value, (maint_ie,))
        if config_path.is_file():
            text = config_path.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"(?im)^\s*(?:types_path|types)\s*=\s*([^\r\n#]+)", text)
            if match:
                candidates.append(resolve_relative(project, match.group(1).strip(), (maint_ie, export_dir)))
    candidates.extend([
        export_dir / "avimaint_types.json",
        export_dir / "types.json",
        maint_ie / "outputs" / "spert" / "avimaint_types.json",
    ])
    for candidate in candidates:
        if candidate.is_file() and parse_types(candidate) == (9, 11):
            return candidate.resolve()
    return None


def disabled_semantic(reason: str, selection: str, **extra) -> dict:
    return {
        "enabled": False,
        "verified_representation": False,
        "reason": reason,
        "model_path": "",
        "weights_path": "",
        "weight_sha256": "",
        "types_path": "",
        "types_sha256": "",
        "selection": selection,
        "registry_key": REGISTRY_KEY,
        "representation": REPRESENTATION,
        "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
        **extra,
    }


def lock_semantic_spert(project: Path, maint_ie: Path, byt5: dict, rules: dict) -> dict:
    registry = maint_ie / "outputs" / "reports" / "normalization_spert_matched_v2" / "MODEL_REGISTRY_V2.json"
    if not registry.is_file():
        return disabled_semantic("Matched normalization/SpERT model registry is missing.", "registry_missing")
    try:
        registry_obj = json.loads(registry.read_text(encoding="utf-8"))
        entry = registry_obj[REGISTRY_KEY]
    except Exception as exc:
        return disabled_semantic(
            f"Registry has no valid '{REGISTRY_KEY}' entry: {type(exc).__name__}", "registry_invalid"
        )

    model = resolve_relative(project, str(entry.get("final_model_path", "")), (maint_ie,))
    export_dir = resolve_relative(project, str(entry.get("export_dir", "")), (maint_ie,))
    if not valid_model_dir(model):
        return disabled_semantic(f"Registered rules_then_byt5 SpERT model is invalid: {model}", "registered_model_invalid")
    types = resolve_types(project, maint_ie, entry, export_dir)
    if types is None:
        return disabled_semantic("No 9-entity/11-relation type definition was resolved for the registered model.", "types_invalid")

    override = os.getenv("AVIMAINT_NORMALIZED_SPERT_MODEL", "").strip()
    if override:
        supplied = Path(override).expanduser().resolve()
        if os.path.normcase(str(supplied)) != os.path.normcase(str(model)):
            return disabled_semantic(
                "AVIMAINT_NORMALIZED_SPERT_MODEL does not match the registry's rules_then_byt5 checkpoint; refused.",
                "mismatched_explicit_override",
                registered_model_path=str(model),
                rejected_override_path=str(supplied),
            )
    if not byt5.get("enabled") or not rules.get("enabled"):
        return disabled_semantic(
            "The semantic SpERT model is registered, but its required rules-then-ByT5 input chain is not fully locked.",
            "upstream_chain_disabled",
            registered_model_path=str(model),
        )

    weight = weight_file(model)
    entity_count, relation_count = parse_types(types)
    return {
        "enabled": True,
        "verified_representation": True,
        "reason": "Exact rules_then_byt5 model selected from MODEL_REGISTRY_V2.json with matched upstream chain.",
        "model_path": str(model),
        "weights_path": str(weight),
        "weight_sha256": sha256_file(weight),
        "types_path": str(types),
        "types_sha256": sha256_file(types),
        "entity_types": entity_count,
        "relation_types": relation_count,
        "selection": "model_registry_v2_exact_key",
        "registry_path": str(registry.resolve()),
        "registry_sha256": sha256_file(registry),
        "registry_key": REGISTRY_KEY,
        "representation": REPRESENTATION,
        "metrics": entry.get("metrics", {}),
        "frozen_operational_model_id": FROZEN_OPERATIONAL_MODEL_ID,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(args.project_root).expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root not found: {project}")
    dashboard_root = Path(__file__).resolve().parents[1]
    maint_ie = project / "legacy_import" / "maintenance-ie"
    rules = lock_rules(dashboard_root)
    byt5 = lock_byt5(project)
    semantic = lock_semantic_spert(project, maint_ie, byt5, rules)

    lock = {
        "schema": "avimaint-runtime-model-lock-v5",
        "runtime_revision": "v7.2.1-r4",
        "project_root": str(project),
        "semantic_representation": REPRESENTATION,
        "normalization_rules": rules,
        "byt5": byt5,
        "normalized_spert": semantic,
        "research_branch": {
            "rq4": "true_raw_structure",
            "rq5_recalibrated": False,
            "modified_by_this_lock": False,
        },
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print("PHASE1_MODEL_LOCK_V5_OK")
    print("Representation:", REPRESENTATION)
    print("Expert rules enabled:", rules["enabled"])
    print("ByT5 enabled:", byt5["enabled"])
    print("ByT5 reason:", byt5["reason"])
    if byt5.get("model_path"):
        print("ByT5 model:", byt5["model_path"])
        print("ByT5 weight SHA:", byt5["weight_sha256"])
    print("Hybrid semantic SpERT enabled:", semantic["enabled"])
    print("Hybrid semantic SpERT verified:", semantic["verified_representation"])
    print("Semantic reason:", semantic["reason"])
    if semantic.get("model_path"):
        print("Semantic model:", semantic["model_path"])
        print("Semantic weight SHA:", semantic["weight_sha256"])
        print("Types SHA:", semantic["types_sha256"])
    print("Lock:", output)


if __name__ == "__main__":
    main()
