@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo.
echo ======================================================================
echo   RQ2 CORE-SCHEMA CLASSICAL IE
 echo  Model 1: CRF NER ^(L-BFGS, DEV-tuned c1/c2^)
echo   Model 2: Logistic Regression RE ^(balanced classes, DEV-tuned C^)
echo   Schema : CORE 8 entities / 10 relations
echo   Split  : frozen TRAIN / DEV / TEST - TEST never used for tuning
echo   Compute: CPU
echo   Live   : progress, current config, DEV F1, best-so-far, elapsed, ETA
echo ======================================================================
echo.

pushd "%ROOT%"
echo [PRECHECK] Auditing the frozen split before training...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\audit_frozen_split.py
if errorlevel 1 goto :failed_root
popd

pushd "%ROOT%\legacy_import\maintenance-ie"
echo.
echo [TRAIN] Starting CORE classical experiment...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\05_train_eval.py --tiers 1 --tune --require-frozen-split --run-id core_tier1 --gold-glob "outputs/gold_core/*.jsonl" --schema-path config/schema_core.yaml --report-name ie_results_core
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" goto :failed

echo.
echo ======================================================================
echo   CORE CLASSICAL RUN COMPLETE
echo   Results : legacy_import\maintenance-ie\outputs\reports\ie_results_core.json
echo   Log     : legacy_import\maintenance-ie\outputs\reports\ie_results_core_log.csv
echo   Manifest: legacy_import\maintenance-ie\outputs\reports\ie_results_core_latest_training_manifest.json
echo   Trace   : legacy_import\maintenance-ie\outputs\reports\training_logs\
echo ======================================================================
exit /b 0

:failed_root
set RESULT=%ERRORLEVEL%
popd
goto :failed

:failed
echo.
echo ERROR: core classical IE run failed with exit code %RESULT%.
echo The JSONL training trace preserves completed tuning steps for diagnosis.
pause
exit /b %RESULT%
