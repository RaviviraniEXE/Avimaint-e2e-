@echo off
setlocal EnableExtensions EnableDelayedExpansion
title AviMaint-DSS - Phase 5 frontend and reviewed backend
cd /d "%~dp0"

set "DASH_ENV=avimaint-dash"
set "RAW_HEALTH=http://127.0.0.1:8765/health"
set "NORM_HEALTH=http://127.0.0.1:8766/health"
set "SEM_HEALTH=http://127.0.0.1:8767/health"
set "MODEL_LOCK=%~dp0runtime_model_lock.json"

echo ============================================================
echo  AviMaint-DSS - PHASES 1-4 V7.2 final-reviewed backend
echo  RQ4/RQ5 branch : TRUE-RAW matched SpERT (:8765)
echo  Normalization  : locked guarded ByT5 (:8766)
echo  Diagnose IE    : normalized semantic SpERT (:8767, CPU)
echo  Reranker       : presentation-only
echo  REST API       : FastAPI on :8780
echo  Phase 5 UI     : React frontend at http://127.0.0.1:8780/
echo ============================================================
echo.

if not exist "%~dp0runtime_model_lock.json" (
  echo ERROR: runtime_model_lock.json is missing.
  echo Re-run APPLY_PHASES1_4_V7_2_FINAL_REVIEW.ps1.
  pause
  exit /b 2
)

echo [1/8] Checking matched TRUE-RAW RQ4 SpERT...
powershell -NoProfile -Command ^
 "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 2 '%RAW_HEALTH%'; $ok=($r.status -eq 'ready' -and [int]$r.entity_types -eq 9 -and [int]$r.relation_types -eq 11 -and $r.query_case_normalization -eq 'none_true_raw')} catch {}; if($ok){exit 0}else{exit 1}"
if not errorlevel 1 goto :RAW_READY

echo     Starting matched TRUE-RAW SpERT...
powershell -NoProfile -Command ^
 "try {$seen=@{}; $conns=@(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction Stop); foreach($c in $conns){$procId=[int]$c.OwningProcess; if(-not $seen.ContainsKey($procId)){$seen[$procId]=$true; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue}}} catch {}"
timeout /t 2 /nobreak >nul
start "AviMaint MATCHED RAW SpERT - keep open" cmd /k call "%~dp0FINAL_03_START_MATCHED_SPERT.bat"
for /L %%N in (1,1,60) do (
  powershell -NoProfile -Command ^
   "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 2 '%RAW_HEALTH%'; $ok=($r.status -eq 'ready' -and [int]$r.entity_types -eq 9 -and [int]$r.relation_types -eq 11 -and $r.query_case_normalization -eq 'none_true_raw')} catch {}; if($ok){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 goto :RAW_READY
  <nul set /p="."
  timeout /t 3 /nobreak >nul
)
echo.
echo ERROR: matched TRUE-RAW SpERT did not become ready.
pause
exit /b 2

:RAW_READY
echo.
echo [2/8] RQ4 SpERT identity:
powershell -NoProfile -Command ^
 "$r=Invoke-RestMethod -TimeoutSec 3 '%RAW_HEALTH%'; ConvertTo-Json -InputObject $r -Depth 5"

echo.
echo [3/8] Checking guarded ByT5 normalization service...
powershell -NoProfile -Command ^
 "$l=Get-Content -Raw '%MODEL_LOCK%'|ConvertFrom-Json; if($l.byt5.enabled){exit 0}else{Write-Host ('    ByT5 disabled by model lock: ' + $l.byt5.reason); exit 3}"
if errorlevel 3 goto :SEM_CHECK
powershell -NoProfile -Command ^
 "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 2 '%NORM_HEALTH%'; $lock=(Get-Content -Raw '%MODEL_LOCK%'|ConvertFrom-Json); $ok=($r.status -eq 'ready' -and $r.role -eq 'operational_byt5_normalizer' -and $r.model -eq $lock.byt5.model_path)} catch {}; if($ok){exit 0}else{exit 1}"
if not errorlevel 1 goto :NORM_READY
powershell -NoProfile -Command ^
 "try {$conns=@(Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction Stop); foreach($c in $conns){Stop-Process -Id ([int]$c.OwningProcess) -Force -ErrorAction SilentlyContinue}} catch {}"
start "AviMaint guarded ByT5 - keep open" cmd /k call "%~dp0START_NORMALIZATION_SERVICE.bat"
for /L %%N in (1,1,90) do (
  powershell -NoProfile -Command ^
   "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 2 '%NORM_HEALTH%'; $lock=(Get-Content -Raw '%MODEL_LOCK%'|ConvertFrom-Json); $ok=($r.status -eq 'ready' -and $r.role -eq 'operational_byt5_normalizer' -and $r.model -eq $lock.byt5.model_path)} catch {}; if($ok){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 goto :NORM_READY
  <nul set /p="."
  timeout /t 2 /nobreak >nul
)
echo.
echo WARNING: normalization service unavailable. Diagnose will safely fall back to raw SpERT.
goto :SEM_CHECK

:NORM_READY
echo.
echo     Normalization service ready:
powershell -NoProfile -Command ^
 "$r=Invoke-RestMethod -TimeoutSec 3 '%NORM_HEALTH%'; ConvertTo-Json -InputObject $r -Depth 5"

:SEM_CHECK
echo.
echo [4/8] Checking optional verified normalized semantic SpERT...
powershell -NoProfile -Command ^
 "$l=Get-Content -Raw '%MODEL_LOCK%'|ConvertFrom-Json; if($l.normalized_spert.enabled -and $l.normalized_spert.verified_representation){exit 0}else{Write-Host ('    Semantic SpERT disabled by model lock: ' + $l.normalized_spert.reason); exit 3}"
if errorlevel 3 goto :RERANK
powershell -NoProfile -Command ^
 "$ok=$false; try {$lock=(Get-Content -Raw '%MODEL_LOCK%' | ConvertFrom-Json); $r=Invoke-RestMethod -TimeoutSec 3 '%SEM_HEALTH%'; $ok=($r.status -eq 'ready' -and $r.role -eq 'normalized_semantic_spert' -and $r.representation -eq 'normalized_operational' -and $r.weights_sha256 -eq $lock.normalized_spert.weight_sha256)} catch {}; if($ok){exit 0}else{exit 1}"
if not errorlevel 1 goto :SEM_READY
powershell -NoProfile -Command ^
 "try {$conns=@(Get-NetTCPConnection -LocalPort 8767 -State Listen -ErrorAction Stop); foreach($c in $conns){Stop-Process -Id ([int]$c.OwningProcess) -Force -ErrorAction SilentlyContinue}} catch {}"
start "AviMaint NORMALIZED SpERT - keep open" cmd /k call "%~dp0START_NORMALIZED_SPERT_SERVICE.bat"
for /L %%N in (1,1,90) do (
  powershell -NoProfile -Command ^
   "$ok=$false; try {$lock=(Get-Content -Raw '%MODEL_LOCK%' | ConvertFrom-Json); $r=Invoke-RestMethod -TimeoutSec 3 '%SEM_HEALTH%'; $ok=($r.status -eq 'ready' -and $r.role -eq 'normalized_semantic_spert' -and $r.representation -eq 'normalized_operational' -and $r.weights_sha256 -eq $lock.normalized_spert.weight_sha256)} catch {}; if($ok){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 goto :SEM_READY
  <nul set /p="."
  timeout /t 2 /nobreak >nul
)
echo.
echo INFO: verified normalized semantic SpERT is unavailable/disabled. Diagnose will safely use the validated raw SpERT.
goto :RERANK

:SEM_READY
echo.
echo [5/8] Normalized SpERT identity:
powershell -NoProfile -Command ^
 "$r=Invoke-RestMethod -TimeoutSec 3 '%SEM_HEALTH%'; ConvertTo-Json -InputObject $r -Depth 5"

:RERANK
echo.
echo [6/8] Checking reranker...
set "CUDA_VISIBLE_DEVICES=-1"
call conda run --no-capture-output -n %DASH_ENV% python -u check_reranker.py
if errorlevel 1 (
  echo ERROR: reranker check failed.
  pause
  exit /b 3
)


echo.
echo [7/8] Checking frontend-ready REST API on :8780...
powershell -NoProfile -Command ^
 "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 3 'http://127.0.0.1:8780/api/v1/health'; $ok=($r.status -eq 'ready' -and $r.api_version -eq '1.0.1' -and $r.rq4_base -eq 'structure' -and $r.candidate_split -eq 'train' -and $r.frontend.ready -eq $true -and $r.frontend.version -eq '5.0.0')} catch {}; if($ok){exit 0}else{exit 1}"
if not errorlevel 1 goto :API_READY
powershell -NoProfile -Command ^
 "try {$seen=@{}; $conns=@(Get-NetTCPConnection -LocalPort 8780 -State Listen -ErrorAction Stop); foreach($c in $conns){$procId=[int]$c.OwningProcess; if(-not $seen.ContainsKey($procId)){$seen[$procId]=$true; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue}}} catch {}"
timeout /t 1 /nobreak >nul
start "AviMaint-DSS API - keep open" cmd /k call "%~dp0START_API_BACKEND.bat"
for /L %%N in (1,1,90) do (
  powershell -NoProfile -Command ^
   "$ok=$false; try {$r=Invoke-RestMethod -TimeoutSec 3 'http://127.0.0.1:8780/api/v1/health'; $ok=($r.status -eq 'ready' -and $r.api_version -eq '1.0.1' -and $r.rq4_base -eq 'structure' -and $r.candidate_split -eq 'train' -and $r.frontend.ready -eq $true -and $r.frontend.version -eq '5.0.0')} catch {}; if($ok){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 goto :API_READY
  <nul set /p="."
  timeout /t 2 /nobreak >nul
)
echo.
echo ERROR: FastAPI backend did not become ready.
pause
exit /b 4

:API_READY
echo.
echo     API ready: http://127.0.0.1:8780
echo     Swagger : http://127.0.0.1:8780/docs

if /I "%~1"=="--legacy-streamlit" goto :LEGACY_STREAMLIT

echo.
echo [8/8] Opening the Phase 5 frontend...
echo     http://127.0.0.1:8780/
echo.
start "" "http://127.0.0.1:8780/"
echo AviMaint-DSS is running. Keep this window and the service windows open.
echo Press any key only when you want to close this launcher window.
pause >nul
exit /b 0

:LEGACY_STREAMLIT
echo.
echo [8/8] Starting the legacy Streamlit comparison UI...
echo     http://localhost:8502
echo.
call conda run --no-capture-output -n %DASH_ENV% python -m streamlit run app.py --server.port 8502
set "RESULT=%ERRORLEVEL%"
echo.
echo Dashboard stopped with exit code %RESULT%.
pause
exit /b %RESULT%
