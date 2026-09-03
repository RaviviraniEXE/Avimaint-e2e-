@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie\avimaint_dss"
call conda run -n avimaint-spert python services\spert_query_service.py --project-root "%~dp0\..\..\legacy_import\maintenance-ie"
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" pause
exit /b %RESULT%
