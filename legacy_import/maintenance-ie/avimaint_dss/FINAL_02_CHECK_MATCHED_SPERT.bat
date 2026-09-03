@echo off
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%PROJECT_ROOT%\..\..") do set "REPO_ROOT=%%~fI"
call conda run --no-capture-output -n avimaint-spert python -u services\spert_query_service.py --project-root "%PROJECT_ROOT%" --spert-root "%REPO_ROOT%\external\spert" --check-only
exit /b %ERRORLEVEL%
