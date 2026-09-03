@echo off
title AviMaint-DSS API
cd /d "%~dp0"
echo ============================================================
echo  AviMaint-DSS FastAPI backend
echo  API     : http://127.0.0.1:8780
echo  Swagger : http://127.0.0.1:8780/docs
echo ============================================================
echo.
set "CUDA_VISIBLE_DEVICES=-1"
call conda run --no-capture-output -n avimaint-dash python -m uvicorn api_server:app ^
  --host 127.0.0.1 --port 8780
echo.
echo API stopped.
pause
