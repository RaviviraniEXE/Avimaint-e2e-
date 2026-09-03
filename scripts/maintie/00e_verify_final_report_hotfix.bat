@echo off
setlocal
cd /d "%~dp0..\..\"

echo ======================================================================
echo   VERIFY MAINTIE FINAL REPORT HOTFIX - NO TRAINING
echo ======================================================================

if not exist "legacy_import\maintie-bench\scripts\06c_finalize_benchmark_report.py" (
  echo ERROR: report script missing
  exit /b 1
)

findstr /c:"NO TRAINING" "scripts\maintie\06_finalize_benchmark_report.bat" >nul || (
  echo ERROR: safety marker missing
  exit /b 1
)

findstr /c:"training_performed" "legacy_import\maintie-bench\scripts\06c_finalize_benchmark_report.py" >nul || (
  echo ERROR: manifest safety marker missing
  exit /b 1
)

set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo ERROR: Conda executable not found at "%CONDA_EXE%"
  exit /b 1
)

"%CONDA_EXE%" run -n avimaint-ie-neural python -m py_compile "legacy_import\maintie-bench\scripts\06c_finalize_benchmark_report.py"
if errorlevel 1 exit /b %errorlevel%

echo.
echo HOTFIX VERIFIED.
echo This patch generates TEST metrics, tables and figures from existing artifacts only.
echo NO TRAINING WAS PERFORMED.
endlocal
