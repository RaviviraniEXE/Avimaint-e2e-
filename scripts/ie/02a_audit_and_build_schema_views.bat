@echo off
setlocal
pushd "%~dp0\..\.."
call conda run -n avimaint-ie-classical python scripts\ie\install_and_audit_annotations.py
if errorlevel 1 goto :failed
call conda run -n avimaint-ie-classical python scripts\ie\build_schema_views.py
if errorlevel 1 goto :failed
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
