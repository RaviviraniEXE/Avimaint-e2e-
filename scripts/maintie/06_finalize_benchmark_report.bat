@echo off
setlocal
cd /d "%~dp0..\..\legacy_import\maintie-bench"

echo ======================================================================
echo   MAINTIE FINAL BENCHMARK REPORT - EXISTING ARTIFACTS ONLY
echo   NO TRAINING / NO RETRAINING
echo ======================================================================

set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo ERROR: Conda executable not found at "%CONDA_EXE%"
  exit /b 1
)

"%CONDA_EXE%" run --no-capture-output -n avimaint-ie-neural python -u scripts\06c_finalize_benchmark_report.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo FINAL MAINTIE REPORT COMPLETE - NO TRAINING WAS PERFORMED
endlocal
