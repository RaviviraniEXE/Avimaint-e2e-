@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0clone_official_spert.ps1" %*
exit /b %ERRORLEVEL%
