@echo off
setlocal
pushd "%~dp0\..\.."
set SYSTEM=%~1
if "%SYSTEM%"=="" set SYSTEM=rules_then_byt5
call conda run -n avimaint-dashboard python scripts\dashboard\export_dataset.py --system %SYSTEM%
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
