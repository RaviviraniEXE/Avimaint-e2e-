@echo off
cd /d "%~dp0"
call conda run --no-capture-output -n avimaint-dash python -u final_verify.py
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n avimaint-dash python -u check_v721_hybrid_runtime.py
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n avimaint-dash python -u check_phase1_dual_pipeline.py
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n avimaint-dash python -u check_phase4_api.py
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n avimaint-dash python -u check_phase5_views.py
if errorlevel 1 exit /b 1
call conda run --no-capture-output -n avimaint-dash python -u check_phase5_frontend.py
if errorlevel 1 exit /b 1
if exist "..\..\..\outputs\frozen\final_rq4_rq5\SHA256SUMS.txt" (echo FINAL RQ4/RQ5 FREEZE PRESENT) else (echo RQ4/RQ5 freeze not created yet)
echo V7.2.1 R4 HYBRID RUNTIME AND PHASE 5 FRONTEND VERIFIED
