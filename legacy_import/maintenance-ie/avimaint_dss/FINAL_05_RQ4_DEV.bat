@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_rq4_evaluate.py --partition dev
exit /b %ERRORLEVEL%
