@echo off
setlocal EnableExtensions
title AviMaint SpERT service
cd /d "%~dp0"

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%PROJECT_ROOT%\..\..") do set "REPO_ROOT=%%~fI"

set "SPERT_ROOT=%REPO_ROOT%\external\spert"
set "SPERT_ENV=avimaint-spert"

echo ============================================================
echo AviMaint SpERT query service
echo Project root: %PROJECT_ROOT%
echo SpERT source: %SPERT_ROOT%
echo Environment : %SPERT_ENV%
echo ============================================================
echo.

if not exist "%SPERT_ROOT%\spert\models.py" (
    echo ERROR: Official SpERT source was not found:
    echo %SPERT_ROOT%\spert\models.py
    pause
    exit /b 2
)

call conda run --no-capture-output -n %SPERT_ENV% python -u services\spert_query_service.py ^
  --project-root "%PROJECT_ROOT%" ^
  --spert-root "%SPERT_ROOT%"

set "RESULT=%ERRORLEVEL%"
echo.
echo SpERT service stopped with exit code %RESULT%.
pause
exit /b %RESULT%