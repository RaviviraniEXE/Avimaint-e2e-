@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_prepare_rq4.py
exit /b %ERRORLEVEL%
