@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp006_train_and_test_spert.ps1" -ExportName "spert_core"
if errorlevel 1 pause
exit /b %ERRORLEVEL%

