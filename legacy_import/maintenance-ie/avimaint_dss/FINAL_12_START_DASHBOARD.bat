@echo off
setlocal EnableExtensions
title AviMaint-DSS V7.2.1 R4 - all services
cd /d "%~dp0"

echo Starting AviMaint-DSS from one controller window...
echo Separate model processes run in the background because they use different conda environments.
echo.

where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: conda was not found. Open Anaconda Prompt and run this file again.
  pause
  exit /b 2
)

call conda run --no-capture-output -n avimaint-dash python -u tools\runtime_supervisor.py %*
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" (
  echo AviMaint-DSS stopped with error code %RESULT%.
  echo Check the runtime_logs folder for service details.
  pause
)
exit /b %RESULT%
