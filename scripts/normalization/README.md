# Numbered normalization launchers

These `.bat` files are the familiar Windows/Anaconda Prompt entry points. The
real implementation is under `src/avimaint/normalization/`; the launchers call
that code with the versioned YAML configuration.

| Step | Launcher | Main implementation |
|---:|---|---|
| 1 | `01_audit_reference.bat` | `audit.py` |
| 2 | `02_prepare_approved_pairs.bat` | `audit.py` |
| 3 | `03_create_cluster_safe_split.bat` | `splitting.py` |
| 4 | `04_train_byt5_gold.bat` | `training.py` |
| 5 | `05_run_validation_comparison.bat` | `prediction.py`, `evaluation.py` |
| 5a | `05a_rerun_corrected_rule_systems.bat` | Re-run only rules and rules-then-ByT5 after a validated rule-contract correction |
| 5b | `05b_run_validation_ablations.bat` | most-frequent-replacement and selective-ByT5 validation ablations |
| 5c | `05c_run_expert_sensitivity.bat` | separate expert-completion sensitivity population |
| 5d | `05d_recheck_selected_safety.bat` | fast selected-system safety recheck using cached ByT5 candidates |
| 5e | `05e_freeze_validation_selection.bat` | hash and freeze the validation-only choice before test access |
| 6 | `06_run_final_test_once.bat` | locked final evaluation |
| 7 | `07_make_silver_data.bat` | `silver.py`, `rules.py` |
| 8 | `08_train_silver_then_gold.bat` | `training.py` |
| 9 | `09_predict_full_corpus.bat` | field-preserving full-corpus inference |
| 10 | `10_compare_downstream_ie.bat` | audited span projection + frozen IE ablation |

The old normalization scripts are not copied into the new implementation. Put
old source under `legacy_import/normalization/` only when it is needed for an
auditable baseline comparison.

`05c` never participates in model selection. It measures agreement on 1,045
expert completions whose missing content may not be recoverable from the raw
fixed-width record, so it must be reported separately from primary lexical
normalization.

`05d` does not retrain or regenerate ByT5 predictions when
`validation_byt5.csv` is present. It applies the frozen selective fallback to
those candidates and recomputes the 1,000-sample bootstrap intervals.

Step 9 normally exports only raw evidence and the selected safe normalizer. Use
`09_predict_full_corpus.bat ALL` only when full-corpus outputs for every
comparison system are explicitly required.
