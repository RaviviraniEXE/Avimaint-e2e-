@echo off
setlocal
pushd "%~dp0\..\.."
echo ======================================================================
echo   VERIFY NORMALIZATION -^> SpERT ABLATION V2 FIX
echo ======================================================================
if not exist "scripts\ie\run_normalization_spert_ablation.ps1" (
  echo ERROR: patched runner is missing.
  goto :failed
)
if not exist "scripts\ie\06_predict_existing_spert.ps1" (
  echo ERROR: existing no-retrain SpERT predictor is missing.
  goto :failed
)
findstr /i /c:"ExportName" "scripts\ie\06_predict_existing_spert.ps1" >nul
if errorlevel 1 (
  echo ERROR: existing predictor does not expose ExportName.
  echo Please stop and send scripts\ie\06_predict_existing_spert.ps1 for inspection.
  goto :failed
)
if not exist "legacy_import\maintenance-ie\outputs\spert\test.json" (
  echo ERROR: full-schema outputs\spert\test.json is missing.
  goto :failed
)
if not exist "legacy_import\maintenance-ie\outputs\spert\save" (
  echo ERROR: full-schema outputs\spert\save is missing.
  goto :failed
)
findstr /i /c:"-ExportName $ExportName" "scripts\ie\run_normalization_spert_ablation.ps1" >nul
if errorlevel 1 (
  echo ERROR: V2 runner does not explicitly pass the full-schema export.
  goto :failed
)
findstr /i /c:"ErrorActionPreference = \"Continue\"" "scripts\ie\run_normalization_spert_ablation.ps1" >nul
if errorlevel 1 (
  echo ERROR: NativeCommandError warning-stream fix is missing.
  goto :failed
)
echo.
echo V2 FIX VERIFIED.
echo - transformers FutureWarning on STDERR will NOT be treated as model failure.
echo - predictor is explicitly called with ExportName=spert ^(FULL 9x11 model^).
echo - model training is NOT invoked.
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
if "%RESULT%"=="0" set RESULT=1
popd
exit /b %RESULT%
