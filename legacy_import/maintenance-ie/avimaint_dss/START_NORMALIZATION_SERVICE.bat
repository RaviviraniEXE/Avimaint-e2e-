@echo off
title AviMaint guarded ByT5 service
cd /d "%~dp0"
set "LOCK=%~dp0runtime_model_lock.json"
if not exist "%LOCK%" (
  echo runtime_model_lock.json missing
  exit /b 2
)

powershell -NoProfile -Command ^
 "$l=Get-Content -Raw '%LOCK%'|ConvertFrom-Json; if($l.byt5.enabled){exit 0}else{Write-Host ('ByT5 disabled: ' + $l.byt5.reason); exit 3}"
if errorlevel 1 exit /b 0

call conda run --no-capture-output -n avimaint-normalization python -u services\normalization_query_service.py ^
  --lock "%LOCK%" --host 127.0.0.1 --port 8766 --device cpu
exit /b %ERRORLEVEL%
