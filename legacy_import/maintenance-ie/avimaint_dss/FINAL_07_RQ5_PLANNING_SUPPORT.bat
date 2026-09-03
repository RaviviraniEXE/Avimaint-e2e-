@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_rq5_planning_support.py
exit /b %ERRORLEVEL%
