@echo off
setlocal
pushd "%~dp0\..\.."
call conda run -n avimaint-ie-classical python scripts\ie\install_and_audit_annotations.py
if errorlevel 1 goto :failed
call conda run -n avimaint-ie-classical python scripts\ie\build_schema_views.py
if errorlevel 1 goto :failed
pushd legacy_import\maintenance-ie
call conda run -n avimaint-ie-classical python scripts\00_gold_status.py
set RESULT=%ERRORLEVEL%
popd
popd
exit /b %RESULT%
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
