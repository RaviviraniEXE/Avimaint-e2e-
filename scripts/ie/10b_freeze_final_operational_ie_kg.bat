@echo off
setlocal
pushd "%~dp0\..\.."
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\10b_freeze_final_operational_ie_kg.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
