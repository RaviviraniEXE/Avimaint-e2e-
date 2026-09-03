@echo off
setlocal
pushd "%~dp0"
python scripts\verify\project_readiness.py --compile
if errorlevel 1 (
  echo Readiness check failed. See the messages above.
  pause
  exit /b 1
)
echo.
echo Project is ready. Open FULL_EXECUTION_GUIDE.md and begin with environment setup.
pause
popd
