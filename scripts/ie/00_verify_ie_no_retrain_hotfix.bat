@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo ======================================================================
echo   VERIFY IE NO-RETRAIN + RESULT-SAFETY HOTFIX
 echo ======================================================================

if not exist "%ROOT%\legacy_import\maintenance-ie\scripts\12_report_existing.py" (
  echo ERROR: scripts\12_report_existing.py missing.
  exit /b 2
)

findstr /C:"NO MODEL TRAINING / NO RETRAINING" "%ROOT%\scripts\ie\07_import_spert_and_report.bat" >nul
if errorlevel 1 (
  echo ERROR: 07_import_spert_and_report.bat is not the patched no-retrain launcher.
  exit /b 3
)

findstr /C:"result_history" "%ROOT%\legacy_import\maintenance-ie\scripts\05_train_eval.py" >nul
if errorlevel 1 (
  echo ERROR: 05_train_eval.py overwrite protection missing.
  exit /b 4
)

findstr /C:"metrics stable alias" "%ROOT%\legacy_import\maintenance-ie\scripts\09_report.py" >nul
if errorlevel 1 (
  echo ERROR: 09_report.py metrics overwrite protection missing.
  exit /b 5
)

echo [1/2] Static patch markers: PASS
pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run --no-capture-output -n avimaint-ie-neural python -m py_compile scripts\05_train_eval.py scripts\09_report.py scripts\12_report_existing.py
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" exit /b %RESULT%

echo [2/2] Python syntax: PASS
echo.
echo HOTFIX VERIFIED.
echo 07_import_spert_and_report.bat is report-only and will not train models.
exit /b 0
