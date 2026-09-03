@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."
set "PY=%ROOT%\legacy_import\maintenance-ie\scripts\13_generate_final_figures_existing.py"
set "BATFILE=%ROOT%\scripts\ie\08_generate_final_ie_figures.bat"

echo ======================================================================
echo   VERIFY FINAL IE FIGURE HOTFIX
 echo  Safety: report/figure generation only; no training
 echo ======================================================================

if not exist "%PY%" (
  echo ERROR: missing %PY%
  exit /b 2
)
if not exist "%BATFILE%" (
  echo ERROR: missing %BATFILE%
  exit /b 2
)

findstr /C:"NO MODEL TRAINING" "%PY%" >nul || (
  echo ERROR: Python safety marker missing.
  exit /b 3
)
findstr /C:"Old outputs/reports/figures are preserved" "%PY%" >nul || (
  echo ERROR: legacy-figure preservation marker missing.
  exit /b 3
)
findstr /C:"NO MODEL TRAINING / NO RETRAINING" "%BATFILE%" >nul || (
  echo ERROR: batch safety marker missing.
  exit /b 3
)

pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run -n avimaint-ie-classical python -m py_compile scripts\13_generate_final_figures_existing.py
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" (
  echo ERROR: Python syntax verification failed.
  exit /b %RESULT%
)

echo [1/2] Static safety markers: PASS
echo [2/2] Python syntax: PASS
echo.
echo HOTFIX VERIFIED.
echo Run scripts\ie\08_generate_final_ie_figures.bat
exit /b 0
