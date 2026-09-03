@echo off
title AviMaint SpERT service
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
echo Starting SpERT service for project root: %PROJECT_ROOT%
python services\spert_query_service.py --project-root "%PROJECT_ROOT%"
pause

