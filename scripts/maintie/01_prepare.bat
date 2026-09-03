@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"
if not exist data\raw\gold_release.json copy /Y "..\..\data\maintie\raw\gold_release.json" data\raw\gold_release.json >nul
call conda run -n avimaint-ie-classical python scripts\00_convert_maintie.py
if errorlevel 1 goto :failed
call conda run -n avimaint-ie-classical python scripts\00_gold_status.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
