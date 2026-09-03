@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp006_predict_existing_spert.ps1" -ExportName "spert_core"
if errorlevel 1 pause
exit /b %ERRORLEVEL%
