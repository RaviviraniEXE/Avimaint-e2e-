@echo off
setlocal EnableExtensions
pushd "%~dp0\..\.."

echo ======================================================================
echo   CORRECT RQ1 MATCHED NORMALIZATION -> SpERT EXPERIMENT
echo   Finding: historical outputs\spert used normalized annotation text
echo   Action : train ONLY missing true System-A RAW SpERT model
echo   Reuse  : rules, byt5, selective_byt5, rules_then_byt5
echo   Split  : SAME frozen 1275 / 100 / 225
echo   Schema : SAME full 9 entities / 11 relations
echo   Policy : NO hyperparameter retuning
echo ======================================================================

echo.
echo [1/5] Audit historical annotation representation...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\audit_spert_annotation_representation.py
if errorlevel 1 goto :failed

echo.
echo [2/5] Re-project all five representations strictly...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\project_normalization_to_gold.py --systems raw rules byt5 selective_byt5 rules_then_byt5 --min-coverage 0.97
if errorlevel 1 goto :failed

echo.
echo [3/5] Prepare ONLY the corrected true-raw SpERT export and verify the four existing models...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\prepare_true_raw_matched_spert.py
if errorlevel 1 goto :failed

echo.
echo [4/5] Train/predict ONLY the missing true-raw SpERT model...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\ie\train_true_raw_matched_spert.ps1
if errorlevel 1 goto :failed

echo.
echo [5/5] Evaluate the corrected five-way representation-matched experiment...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\evaluate_matched_normalization_spert_v2.py
if errorlevel 1 goto :failed

echo.
echo ======================================================================
echo   CORRECTED MATCHED NORMALIZATION -> SpERT EXPERIMENT COMPLETE
echo.
echo   Main table:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_matched_v2\matched_normalization_spert_ablation_v2.csv
echo.
echo   Model registry:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_matched_v2\MODEL_REGISTRY_V2.json
echo ======================================================================
popd
exit /b 0

:failed
set RESULT=%ERRORLEVEL%
echo.
echo CORRECTION STOPPED with errorlevel %RESULT%.
echo Do not interpret partial results and do not retune against TEST.
popd
exit /b %RESULT%
