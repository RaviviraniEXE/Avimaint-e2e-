@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_precompute_problem_spert.py
exit /b %ERRORLEVEL%
