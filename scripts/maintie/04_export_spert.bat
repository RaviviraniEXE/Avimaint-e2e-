@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"
call conda run -n avimaint-ie-classical python scripts\06_export_spert.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
