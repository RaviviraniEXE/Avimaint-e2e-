# Manual review add-on for the final RQ4/RQ5 pipeline

This add-on supplies the two optional scripts that were referenced in guidance but
were accidentally omitted from the original final-safe hotfix ZIP.

It does **not** alter RQ4 DEV selection, the locked RQ4 TEST, RQ5 calibration,
normalization, IE models, the 6,169-record extraction, or the KG.

## Files added

- `FINAL_08_BUILD_MANUAL_REVIEW_POOL_OPTIONAL.bat`
- `FINAL_09_SCORE_MANUAL_REVIEW_AFTER_FILLING.bat`
- `final_manual_review.py`

## Workflow

1. Run `FINAL_08_BUILD_MANUAL_REVIEW_POOL_OPTIONAL.bat`.
2. Open:
   `D:\avimaint-e2e-research\outputs\runs\rq4_case_retrieval\manual_review\PHASE_A_problem_relevance_BLINDED.csv`
3. Fill only `problem_relevance_0_1_2`:
   - `0` = not relevant
   - `1` = partially relevant / plausibly comparable
   - `2` = clearly relevant / strongly comparable
4. Do **not** open `PRIVATE_DO_NOT_OPEN_UNTIL_PHASE_A_COMPLETE.jsonl` before
   finishing Phase A.
5. Run `FINAL_09_SCORE_MANUAL_REVIEW_AFTER_FILLING.bat`.
   This verifies Phase A is complete and generates the blinded Phase B file.
6. Open `PHASE_B_action_applicability_BLINDED.csv` and fill
   `action_applicability_0_1_2`:
   - `0` = not useful historical action evidence
   - `1` = partially useful / conditionally applicable historical evidence
   - `2` = clearly useful historical action evidence
7. Run `FINAL_09_SCORE_MANUAL_REVIEW_AFTER_FILLING.bat` again.
   It produces `MANUAL_REVIEW_RESULTS.json`.

The judgments concern relevance/usefulness of **historical evidence**. They are
not technical approval, safety validation, aircraft-specific authorization, or
regulatory compliance judgments.

`FINAL_10_FREEZE_RQ4_RQ5.bat` already copies the complete
`outputs/runs/rq4_case_retrieval` tree, so the manual-review results will be
included automatically when the final RQ4/RQ5 freeze is created.
