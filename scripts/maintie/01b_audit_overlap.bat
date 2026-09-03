@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"
echo ======================================================================
echo   MAINTIE OVERLAP / BIO REPRESENTATION AUDIT - NO TRAINING
echo ======================================================================
call conda run --no-capture-output -n avimaint-ie-classical python scripts\00b_overlap_audit.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
