import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_populations_and_leakage_contract() -> None:
    split = json.loads((ROOT / "legacy_import/maintenance-ie/outputs/splits.json").read_text())
    assert (len(split["train"]), len(split["dev"]), len(split["test"])) == (1275, 100, 225)
    audit = json.loads(
        (ROOT / "outputs/reports/annotation_audit/frozen_split_audit.json").read_text()
    )
    assert audit["valid"] is True
    assert audit["rare_records_in_dev_or_test"] == 0
    assert not any(audit["exact_group_overlap"].values())


def test_full_and_core_schema_ablation() -> None:
    full = yaml.safe_load((ROOT / "legacy_import/maintenance-ie/config/schema.yaml").read_text())
    core = yaml.safe_load(
        (ROOT / "legacy_import/maintenance-ie/config/schema_core.yaml").read_text()
    )
    assert (len(full["entities"]), len(full["relations"])) == (9, 11)
    assert (len(core["entities"]), len(core["relations"])) == (8, 10)
    assert "REFERENCE" in full["entities"] and "REFERENCE" not in core["entities"]
    assert "ACTION_FOLLOWS_REFERENCE" in full["relations"]
    assert "ACTION_FOLLOWS_REFERENCE" not in core["relations"]


def test_rq_configs_cover_six_questions() -> None:
    configs = [
        yaml.safe_load(path.read_text())["experiment"]
        for path in sorted((ROOT / "configs/experiments").glob("rq*.yaml"))
    ]
    assert {config["research_question"] for config in configs} == {
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
        "RQ5",
        "RQ6",
    }
    rq5 = next(config for config in configs if config["research_question"] == "RQ5")
    assert rq5["prohibit_action_text_in_query"] is True
    rq4 = next(config for config in configs if config["research_question"] == "RQ4")
    assert rq4["never_merge_maintie_into_aviation_cases"] is True
