@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo.
echo ======================================================================
echo   RQ3 FULL-SCHEMA CLASSICAL IE
 echo  Model 1: CRF NER ^(L-BFGS, DEV-tuned c1/c2^)
echo   Model 2: Logistic Regression RE ^(balanced classes, DEV-tuned C^)
echo   Schema : FULL 9 entities / 11 relations
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
if not exist outputs\splits.json (
  echo ERROR: final frozen split missing. Run scripts\ie\02_freeze_split.bat first.
  popd
  exit /b 3
)

echo.
echo [TRAIN 1/2] Live Tier-1 fit and compact evaluation artifact...
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\05_train_eval.py --tiers 1 --tune --require-frozen-split --run-id aviation_tier1
if errorlevel 1 goto :failed_legacy

echo.
echo [REPORT 2/2] Re-fitting the deterministic Tier-1 baseline for the thesis report,
echo              saving models and bootstrap confidence intervals.
echo              This second fit is retained for compatibility with 09_report.py.
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\09_report.py --tiers 1 --tune --bootstrap 1000 --save-models --run-id aviation_tier1_final
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" goto :failed

echo.
echo ======================================================================
echo   FULL CLASSICAL RUN COMPLETE
echo   Quick results : legacy_import\maintenance-ie\outputs\reports\ie_results.json
echo   Thesis tables : legacy_import\maintenance-ie\outputs\reports\tables\
echo   Saved model   : legacy_import\maintenance-ie\outputs\models\
echo   Live traces   : legacy_import\maintenance-ie\outputs\reports\training_logs\
echo ======================================================================
exit /b 0

:failed_root
set RESULT=%ERRORLEVEL%
popd
goto :failed

:failed_legacy
set RESULT=%ERRORLEVEL%
popd
goto :failed

:failed
echo.
echo ERROR: full classical IE run failed with exit code %RESULT%.
pause
exit /b %RESULT%
