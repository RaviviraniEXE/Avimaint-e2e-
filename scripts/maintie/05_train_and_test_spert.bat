@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp005_train_and_test_spert.ps1"
exit /b %ERRORLEVEL%
