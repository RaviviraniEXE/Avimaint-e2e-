@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_normalization_gpu.ps1"
if errorlevel 1 (
  echo Normalization environment setup failed.
  exit /b 1
)
echo Normalization GPU environment setup completed.
