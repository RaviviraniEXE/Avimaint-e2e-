# Final safe RQ4/RQ5 pipeline

The frozen Selective-ByT5 full-corpus IE/KG is not retrained or overwritten.

1. `FINAL_00_VERIFY_PREREQUISITES.bat`
2. `FINAL_01_PREPARE_RQ4_PROTOCOL.bat`
3. `FINAL_02_CHECK_MATCHED_SPERT.bat` — must resolve the corrected **raw** model from `MODEL_REGISTRY_V2.json` and pass 1600/1600 tokenizer parity.
4. Open a second Anaconda Prompt and run `FINAL_03_START_MATCHED_SPERT.bat`; keep it open.
5. In the first prompt run `FINAL_04_PRECOMPUTE_PROBLEM_SPERT.bat` — requires exactly 6169 unique PROBLEM-only predictions.
6. Run `FINAL_05_RQ4_DEV.bat`. Review `outputs/runs/rq4_case_retrieval/dev/RQ4_DEV_SELECTION.json`.
7. Only after DEV selection is accepted, run `FINAL_06_RQ4_TEST_ONCE.bat`. Type `FINAL`. A lock prevents rerunning the final test.
8. Run `FINAL_07_RQ5_PLANNING_SUPPORT.bat`. Confidence is calibrated to historical action-family agreement only, not technical correctness.
9. Run `FINAL_10_FREEZE_RQ4_RQ5.bat`.
10. Run `FINAL_11_VERIFY_ALL.bat`.
11. Run `FINAL_12_START_DASHBOARD.bat`.

## Scientific safeguards
- Retrieval query/features: PROBLEM only.
- ACTION never enters clustering, query SpERT, or retrieval features.
- Conservative near-duplicate leakage groups are kept entirely within one split.
- Evidence votes are deduplicated by exact problem cluster.
- Similarity score does not equal confidence.
- Weak evidence abstains from an action recommendation while nearest traceable cases remain visible.
- RQ4 TEST is one-time locked.
- RQ5 calibration uses DEV only.
- Legacy filenames `run_rq5_retrieval.py` and `run_rq6_uncertainty.py` are not thesis numbering. Thesis numbering is RQ4 retrieval and RQ5 planning support.
