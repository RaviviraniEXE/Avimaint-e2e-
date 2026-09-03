@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\setup\setup_one.ps1" dashboard
if errorlevel 1 pause
exit /b %ERRORLEVEL%
