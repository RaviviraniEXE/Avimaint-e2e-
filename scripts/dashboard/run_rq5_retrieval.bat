@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie\avimaint_dss"
call conda run -n avimaint-retrieval python run_rq5_retrieval.py %*
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" pause
exit /b %RESULT%
