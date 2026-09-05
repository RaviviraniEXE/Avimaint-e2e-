import json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent
cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
lock = json.loads((root / "runtime_model_lock.json").read_text(encoding="utf-8"))

assert cfg["retrieval"]["default_mode"] == "structure"
assert cfg["normalization"]["use_for_rq4"] is False
assert cfg["normalization"]["use_for_rq5"] is False
assert cfg["normalization"]["mode"] == "rules_then_byt5_guarded"
assert cfg["semantic_extraction"]["use_for_rq4"] is False
assert cfg["semantic_extraction"]["use_for_rq5"] is False
assert cfg["semantic_extraction"]["activation"] == "runtime_model_lock_verified_only"
assert cfg["phase2"]["enabled"] is True
assert cfg["phase2"]["invent_combined_action"] is False
assert lock["schema"] == "avimaint-runtime-model-lock-v5"
assert lock["runtime_revision"] == "v7.2.1-r4"
assert lock["semantic_representation"] == "rules_then_byt5_guarded_operational"
rules = lock["normalization_rules"]
assert rules["enabled"] is True
assert set(rules["sha256"]) == {"abbreviations", "misspellings", "unexpanded"}

sem = lock["normalized_spert"]
if lock["byt5"]["enabled"]:
    assert lock["byt5"]["num_beams"] == 1
    assert lock["byt5"]["decoding_strategy"] == "greedy_deterministic"
if sem["enabled"]:
    assert sem["verified_representation"] is True
    assert sem["registry_key"] == "rules_then_byt5"
    assert sem["selection"] == "model_registry_v2_exact_key"
    assert sem["model_path"]
    assert sem["weight_sha256"]
    assert sem["types_sha256"]
    assert lock["byt5"]["enabled"] is True
else:
    assert sem["verified_representation"] is False

print("PHASE1_PHASE2_SEPARATION_V5_OK")
print("RQ4_BASE", cfg["retrieval"]["default_mode"])
print("BYT5_ENABLED", lock["byt5"]["enabled"])
print("SEMANTIC_SPERT_ENABLED", sem["enabled"])
print("SEMANTIC_SPERT_VERIFIED", sem["verified_representation"])
print("SEMANTIC_SELECTION", sem["selection"])
print("SEMANTIC_REASON", sem["reason"])
print("PHASE2_DECOMPOSITION", cfg["phase2"]["decomposition"])
