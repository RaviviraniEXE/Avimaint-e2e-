@echo off
title Setup AviMaint dashboard env (conda)
cd /d "%~dp0"
echo ============================================================
echo   Creating a FRESH conda env "avimaint-dash" for the
echo   dashboard (isolated from your SpERT env - no conflicts).
echo ============================================================
echo.
echo [1/3] Creating conda env (python 3.11)...
call conda create -n avimaint-dash python=3.11 -y
if errorlevel 1 goto :err
echo.
echo [2/3] Activating and upgrading pip...
call conda activate avimaint-dash
if errorlevel 1 goto :err
python -m pip install --upgrade pip
echo.
echo [3/3] Installing all dashboard dependencies (clean)...
python -m pip install -r requirements.txt
if errorlevel 1 goto :err
echo.
echo ============================================================
echo   DONE. Start the dashboard with:  run_dashboard_conda.bat
echo ============================================================
pause
exit /b 0
:err
echo.
echo Setup failed. Make sure you ran this from an "Anaconda Prompt (miniconda3)"
echo so that the 'conda' command is available.
pause
exit /b 1

