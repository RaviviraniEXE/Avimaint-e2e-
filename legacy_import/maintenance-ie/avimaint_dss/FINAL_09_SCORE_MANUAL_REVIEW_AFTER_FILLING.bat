@echo off
setlocal
cd /d "%~dp0"
echo ================================================================================
echo   OPTIONAL BLINDED MANUAL REVIEW - ADVANCE / SCORE
echo ================================================================================
echo First run after Phase A completion: reveals Phase B actions.
echo Second run after Phase B completion: scores the manual review.
echo ================================================================================
conda run --no-capture-output -n avimaint-dash python -u final_manual_review.py advance
exit /b %ERRORLEVEL%
