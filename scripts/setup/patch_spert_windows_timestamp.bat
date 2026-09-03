@echo off
setlocal EnableExtensions
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0patch_spert_windows_timestamp.ps1"
exit /b %ERRORLEVEL%
