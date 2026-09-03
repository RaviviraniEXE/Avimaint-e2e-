@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo.
echo ======================================================================
echo   FINAL FULL-SCHEMA IE REPORT FROM EXISTING RESULTS ONLY
echo   NO MODEL TRAINING / NO RETRAINING
 echo  Reuses: Tier1 + Tier2 + Tier3 results already computed
 echo  Scores: existing frozen-test SpERT predictions
 echo  Safety: recovers overwritten generic result aliases when possible
 echo  Significance repeats are NOT started by this command
 echo ======================================================================
echo.

pushd "%ROOT%"
echo [PRECHECK] Auditing frozen aviation split...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\ie\audit_frozen_split.py
if errorlevel 1 goto :failed_root
popd

pushd "%ROOT%\legacy_import\maintenance-ie"

echo.
echo [1/2] Importing existing SpERT frozen-test predictions - no training...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\06b_import_spert_preds.py outputs\spert\predictions_test.json
if errorlevel 1 goto :failed_legacy

echo.
echo [2/2] Recovering existing Tier1/Tier2/Tier3 metrics and building combined report...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\12_report_existing.py --spert outputs\reports\spert_test.json --bootstrap 1000 --run-id aviation_existing_all_tiers
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" goto :failed

echo.
echo ======================================================================
echo   EXISTING-RESULT REPORT COMPLETE - NO TRAINING WAS PERFORMED
 echo  Combined: legacy_import\maintenance-ie\outputs\reports\ie_results_combined_existing.json
 echo  Tables  : legacy_import\maintenance-ie\outputs\reports\tables_existing\
 echo  History : legacy_import\maintenance-ie\outputs\reports\result_history\
 echo ======================================================================
exit /b 0

:failed_root
set RESULT=%ERRORLEVEL%
popd
goto :failed

:failed_legacy
set RESULT=%ERRORLEVEL%
popd

:failed
echo.
echo ERROR: existing-result IE report failed with exit code %RESULT%.
echo IMPORTANT: no model training was started by this script.
pause
exit /b %RESULT%
