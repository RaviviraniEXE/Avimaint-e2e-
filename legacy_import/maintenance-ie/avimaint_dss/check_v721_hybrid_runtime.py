"""Static regression suite for the V7.2.1 matched hybrid runtime."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from core.hybrid_normalization import ExpertRuleNormalizer, validate_hybrid_candidate
from services.normalization_query_service import safe_result, smoke_report
from services.normalized_spert_query_service import prediction_contract_ok
from tools.lock_phase1_models import lock_byt5, lock_rules, lock_semantic_spert
from tools.runtime_supervisor import api_ok, normalizer_ok, raw_ok, semantic_ok


ROOT = Path(__file__).resolve().parent
RULE_DIR = ROOT / "data" / "normalization_rules"


rules = ExpertRuleNormalizer(RULE_DIR)
expected = {
    "ON RUN UP, L/H MAG DROPPED 350 RPM.": "on run up, left-hand magneto dropped 350 rpm.",
    "#2 INTAKE LEAKING.": "number 2 intake leaking.",
    "R/H ENG #4 CYL HAS LOW COMPRESSION (20/80 PSI).": (
        "right-hand engine number 4 cylinder has low compression 20/80 psi."
    ),
}
for raw, normalized in expected.items():
    assert rules.normalize(raw).normalized == normalized

bad_ok, bad_reasons = validate_hybrid_candidate(
    "L/H MAG EXC RPM DROP DURING RUN UP",
    "left-hand magneto exc rpm drop during run up",
    "NUMBER ONE, HUNDRED MAGNETO EXHAUST RPM DROP DURING RUN UP",
)
assert bad_ok is False
assert any("number words" in reason for reason in bad_reasons)
assert any("directional" in reason for reason in bad_reasons)

good_ok, good_reasons = validate_hybrid_candidate(
    "R/H ENG #4 CYL HAS LOW COMPRESSION (20/80 PSI).",
    expected["R/H ENG #4 CYL HAS LOW COMPRESSION (20/80 PSI)."],
    expected["R/H ENG #4 CYL HAS LOW COMPRESSION (20/80 PSI)."],
)
assert good_ok is True, good_reasons


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


with tempfile.TemporaryDirectory() as temp:
    project = Path(temp)
    maint_ie = project / "legacy_import" / "maintenance-ie"
    byt5 = project / "outputs" / "normalization" / "models" / "byt5_gold_v1" / "checkpoint-1953"
    write(byt5 / "config.json", "{}")
    write(byt5 / "model.safetensors", "byt5-test-weight")
    write(project / "configs" / "normalization" / "byt5_gold.yaml", "task_prefix: 'normalize: '")
    trainer = project / "outputs" / "normalization" / "models" / "byt5_gold_v1" / "checkpoint-2604" / "trainer_state.json"
    write(trainer, json.dumps({"best_model_checkpoint": "outputs/normalization/models/byt5_gold_v1/checkpoint-1953"}))

    semantic = maint_ie / "outputs" / "spert_normalized" / "rules_then_byt5" / "save" / "run-1" / "final_model"
    write(semantic / "config.json", "{}")
    write(semantic / "model.safetensors", "semantic-test-weight")
    types = maint_ie / "outputs" / "spert" / "avimaint_types.json"
    write(types, json.dumps({
        "entities": {f"E{i}": {} for i in range(9)},
        "relations": {f"R{i}": {} for i in range(11)},
    }))
    registry = maint_ie / "outputs" / "reports" / "normalization_spert_matched_v2" / "MODEL_REGISTRY_V2.json"
    write(registry, json.dumps({"rules_then_byt5": {
        "normalization_system": "rules_then_byt5",
        "export_dir": str((maint_ie / "outputs" / "spert_normalized" / "rules_then_byt5").relative_to(project)),
        "final_model_path": str(semantic.relative_to(project)),
        "metrics": {"strict_relation_micro_f1": 0.8135},
    }}))

    saved_byt5 = os.environ.pop("AVIMAINT_BYT5_MODEL", None)
    saved_semantic = os.environ.pop("AVIMAINT_NORMALIZED_SPERT_MODEL", None)
    try:
        byt5_lock = lock_byt5(project)
        rule_lock = lock_rules(ROOT)
        semantic_lock = lock_semantic_spert(project, maint_ie, byt5_lock, rule_lock)
        assert byt5_lock["enabled"] is True
        assert byt5_lock["source"] == "unanimous_trainer_state_best"
        assert byt5_lock["num_beams"] == 1
        assert byt5_lock["decoding_strategy"] == "greedy_deterministic"
        assert byt5_lock["task_prefix"] == "normalize: "
        assert byt5_lock["task_prefix_source_sha256"]
        assert semantic_lock["enabled"] is True
        assert semantic_lock["selection"] == "model_registry_v2_exact_key"
        assert semantic_lock["registry_key"] == "rules_then_byt5"

        wrong = maint_ie / "outputs" / "spert_normalized" / "byt5" / "final_model"
        write(wrong / "config.json", "{}")
        write(wrong / "model.safetensors", "wrong-weight")
        os.environ["AVIMAINT_NORMALIZED_SPERT_MODEL"] = str(wrong)
        refused = lock_semantic_spert(project, maint_ie, byt5_lock, rule_lock)
        assert refused["enabled"] is False
        assert refused["selection"] == "mismatched_explicit_override"
    finally:
        if saved_byt5 is not None:
            os.environ["AVIMAINT_BYT5_MODEL"] = saved_byt5
        else:
            os.environ.pop("AVIMAINT_BYT5_MODEL", None)
        if saved_semantic is not None:
            os.environ["AVIMAINT_NORMALIZED_SPERT_MODEL"] = saved_semantic
        else:
            os.environ.pop("AVIMAINT_NORMALIZED_SPERT_MODEL", None)


lock = {
    "byt5": {"model_path": "B", "weight_sha256": "B-SHA"},
    "normalized_spert": {"weight_sha256": "S-SHA"},
}
assert raw_ok({"status": "ready", "entity_types": 9, "relation_types": 11,
               "query_case_normalization": "none_true_raw"})
assert not raw_ok({"status": "ready", "entity_types": 9, "relation_types": 11,
                   "query_case_normalization": "ascii_uppercase"})
assert normalizer_ok({
    "status": "ready", "role": "operational_rules_then_byt5_normalizer",
    "representation": "rules_then_byt5_guarded_operational", "model": "B",
    "model_weight_sha256": "B-SHA", "num_beams": 1,
    "decoding_strategy": "greedy_deterministic",
}, lock)
assert semantic_ok({
    "status": "ready", "role": "rules_then_byt5_semantic_spert",
    "representation": "rules_then_byt5_guarded_operational", "weights_sha256": "S-SHA",
}, lock)
assert api_ok({
    "status": "ready", "api_version": "1.0.2", "rq4_base": "structure",
    "candidate_split": "train", "raw_spert": {"ready": True},
    "rq5_calibrator": {"ready": True}, "frontend": {"ready": True, "version": "5.0.1"},
})


class FakeRunner:
    def normalize(self, text):
        if "350" in text:
            return {
                "original": text, "candidate_normalized": "magneto dropped thirty rpm",
                "normalized": text, "accepted_for_semantic_spert": False,
                "warnings": ["numeric values changed"],
            }
        candidate = text.lower().replace("(all cyl)", "all cylinder")
        return {
            "original": text, "candidate_normalized": candidate,
            "normalized": candidate, "accepted_for_semantic_spert": True,
            "warnings": [],
        }


report = smoke_report(FakeRunner())
assert report["decoder"] == "greedy_deterministic"
assert report["accepted"]
assert all(safe_result(row) for row in report["safety_results"] + report["viability_results"])
assert "number 3 intake is leaking." in report["semantic_smoke_texts"]

assert prediction_contract_ok({"entities": [], "relations": []}) == (True, "")
ok, reason = prediction_contract_ok({
    "entities": [{"type": "FAULT"}],
    "relations": [{"type": "ISSUE_ON_ITEM", "head": 0, "tail": 2}],
})
assert ok is False and "outside" in reason

print("V7_2_1_HYBRID_RUNTIME_REGRESSION_OK")
