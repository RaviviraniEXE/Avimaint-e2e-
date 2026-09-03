@echo off
setlocal EnableExtensions
pushd "%~dp0\..\.."

echo ======================================================================
echo   REPRESENTATION-MATCHED NORMALIZATION -^> FULL SpERT ABLATION
echo   Raw baseline: reuse frozen existing model
echo   Train NEW: rules, byt5, selective_byt5, rules_then_byt5
echo   Split: SAME frozen 1275 train / 100 dev / 225 TEST
echo   Schema: FULL 9 entities / 11 relations
echo   Policy: fixed hyperparameters; NO post-TEST retuning
echo ======================================================================

echo.
echo [1/4] Re-project all five normalization representations strictly...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\project_normalization_to_gold.py --systems raw rules byt5 selective_byt5 rules_then_byt5 --min-coverage 0.97
if errorlevel 1 goto :failed

echo.
echo [2/4] Prepare representation-matched train/dev/test exports...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\prepare_matched_normalization_spert.py
if errorlevel 1 goto :failed

echo.
echo [3/4] Train/predict four normalized SpERT models on CUDA...
echo       This stage is RESUMABLE. Existing final_model folders are never retrained.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\ie\train_matched_normalization_spert.ps1
if errorlevel 1 goto :failed

echo.
echo [4/4] Evaluate all five conditions ONCE after all fixed models exist...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\evaluate_matched_normalization_spert.py
if errorlevel 1 goto :failed

echo.
echo ======================================================================
echo   MATCHED NORMALIZATION -^> SpERT EXPERIMENT COMPLETE
echo   Main table:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_matched\matched_normalization_spert_ablation.csv
echo   Model registry:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_matched\MODEL_REGISTRY.json
echo ======================================================================
popd
exit /b 0

:failed
set RESULT=%ERRORLEVEL%
echo.
echo EXPERIMENT STOPPED with errorlevel %RESULT%.
echo Do not tune against partial TEST results. Fix the execution problem and resume.
popd
exit /b %RESULT%
