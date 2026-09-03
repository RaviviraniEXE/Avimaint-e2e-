@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_gpu_models.ps1"
if errorlevel 1 pause
exit /b %ERRORLEVEL%
