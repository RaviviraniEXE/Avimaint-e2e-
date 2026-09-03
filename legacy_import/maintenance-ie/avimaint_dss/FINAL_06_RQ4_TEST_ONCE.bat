@echo off
cd /d "%~dp0"
echo WARNING: this permanently locks the final RQ4 TEST. Continue only after reviewing DEV selection.
set /p OK=Type FINAL to continue: 
if /I not "%OK%"=="FINAL" exit /b 2
call conda run --no-capture-output -n avimaint-dash python -u final_rq4_evaluate.py --partition test --confirm-final-test
exit /b %ERRORLEVEL%
