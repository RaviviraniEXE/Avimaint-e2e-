@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   NORMALIZATION -^> FROZEN FULL SpERT ABLATION
echo   Systems: raw, rules, byt5, selective_byt5, rules_then_byt5
echo   Model  : SAME existing full-schema SpERT for every condition
echo   Policy : NO TRAINING / NO RETUNING
echo ======================================================================

if not exist "outputs\normalization\full_corpus\raw.csv" goto :missing_norm
if not exist "outputs\normalization\full_corpus\rules.csv" goto :missing_norm
if not exist "outputs\normalization\full_corpus\byt5.csv" goto :missing_norm
if not exist "outputs\normalization\full_corpus\selective_byt5.csv" goto :missing_norm
if not exist "outputs\normalization\full_corpus\rules_then_byt5.csv" goto :missing_norm

if not exist "legacy_import\maintenance-ie\outputs\spert\test.json" (
  echo ERROR: missing authoritative full-schema SpERT test.json
  goto :failed
)

if not exist "scripts\ie\06_predict_existing_spert.ps1" (
  echo ERROR: missing scripts\ie\06_predict_existing_spert.ps1
  goto :failed
)

echo.
echo [1/4] Strictly re-projecting all five normalization variants...
call conda run --no-capture-output -n avimaint-ie-classical python scripts\ie\project_normalization_to_gold.py --systems raw rules byt5 selective_byt5 rules_then_byt5 --min-coverage 0.97
if errorlevel 1 goto :failed

echo.
echo [2/4] Preparing the identical frozen 225-record SpERT test variants...
call conda run --no-capture-output -n avimaint-ie-classical python scripts\ie\prepare_normalization_spert_ablation.py
if errorlevel 1 goto :failed

echo.
echo [3/4] Running the SAME existing frozen SpERT checkpoint five times...
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ie\run_normalization_spert_ablation.ps1
if errorlevel 1 goto :failed

echo.
echo [4/4] Evaluating entities + strict relations and checking RAW parity...
call conda run --no-capture-output -n avimaint-ie-classical python scripts\ie\evaluate_normalization_spert_ablation.py
if errorlevel 1 goto :failed

echo.
echo ======================================================================
echo   EXPERIMENT COMPLETE
echo   No model was trained or tuned.
echo.
echo   Main table:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_ablation\normalization_spert_ablation.csv
echo.
echo   Manifest:
echo   legacy_import\maintenance-ie\outputs\reports\normalization_spert_ablation\FINAL_NORMALIZATION_SPERT_MANIFEST.json
echo ======================================================================
popd
exit /b 0

:missing_norm
echo ERROR: all five full-corpus normalization CSV files are required.
echo Run:
echo   scripts\normalization\09_predict_full_corpus.bat ALL
goto :failed

:failed
set RESULT=%ERRORLEVEL%
if "%RESULT%"=="0" set RESULT=1
echo.
echo EXPERIMENT FAILED. Do not interpret partial outputs.
popd
exit /b %RESULT%
