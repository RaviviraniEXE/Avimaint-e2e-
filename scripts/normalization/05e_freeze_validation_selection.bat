@echo off
setlocal
pushd "%~dp0\..\.."
echo Freezing the selected normalization policy from validation artifacts only
conda run --no-capture-output -n avimaint-normalization python -u scripts\normalization\freeze_validation_selection.py
if errorlevel 1 goto :failed
echo Validation selection frozen. The one-time final test is now unlocked.
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
