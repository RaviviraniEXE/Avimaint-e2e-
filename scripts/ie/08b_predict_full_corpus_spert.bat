@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp008b_predict_full_corpus_spert.ps1"
exit /b %ERRORLEVEL%
