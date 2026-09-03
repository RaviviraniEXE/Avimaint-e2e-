@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_freeze_rq4_rq5.py
exit /b %ERRORLEVEL%
