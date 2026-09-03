@echo off
title AviMaint verified normalized semantic SpERT
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "MAINT_IE_ROOT=%%~fI"
set "LOCK=%~dp0runtime_model_lock.json"
if not exist "%LOCK%" (
  echo runtime_model_lock.json missing
  exit /b 2
)

powershell -NoProfile -Command ^
 "$l=Get-Content -Raw '%LOCK%'|ConvertFrom-Json; if($l.normalized_spert.enabled -and $l.normalized_spert.verified_representation){exit 0}else{Write-Host ('Semantic SpERT disabled: ' + $l.normalized_spert.reason); exit 3}"
if errorlevel 1 exit /b 0

call conda run --no-capture-output -n spert python -u services\normalized_spert_query_service.py ^
  --project-root "%MAINT_IE_ROOT%" --lock "%LOCK%" ^
  --host 127.0.0.1 --port 8767 --cpu
exit /b %ERRORLEVEL%
