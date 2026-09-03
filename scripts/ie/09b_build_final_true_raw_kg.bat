@echo off
setlocal
pushd "%~dp0\..\.."
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\09b_build_final_true_raw_kg.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
