@echo off
setlocal
cd /d "%~dp0"
echo ================================================================================
echo   OPTIONAL BLINDED MANUAL REVIEW - PHASE A
echo   Uses the already locked DEV-selected RQ4 system; no tuning is performed.
echo ================================================================================
conda run --no-capture-output -n avimaint-dash python -u final_manual_review.py build --queries 100 --top-k 5
exit /b %ERRORLEVEL%
