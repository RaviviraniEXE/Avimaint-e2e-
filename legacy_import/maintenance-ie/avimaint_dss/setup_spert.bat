@echo off
title AviMaint SpERT service (isolated env)
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

echo ============================================================
echo   SpERT service - isolated environment (avoids the
echo   transformers version conflict with the reranker)
echo   Project root: %PROJECT_ROOT%
echo ============================================================
echo.
if not exist ".venv_spert\Scripts\python.exe" (
  echo [1/2] Creating isolated SpERT environment (.venv_spert)...
  python -m venv .venv_spert
  call .venv_spert\Scripts\activate.bat
  python -m pip install --upgrade pip
  python -m pip install -r requirements-spert.txt
) else (
  call .venv_spert\Scripts\activate.bat
)
echo.
echo [2/2] Starting SpERT service on http://127.0.0.1:8765 ...
echo       Leave this window open. The dashboard connects automatically.
python services\spert_query_service.py --project-root "%PROJECT_ROOT%"
pause

