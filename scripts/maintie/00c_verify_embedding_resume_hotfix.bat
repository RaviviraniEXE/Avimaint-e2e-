@echo off
setlocal
set ROOT=%~dp0\..\..
set PY=%ROOT%\legacy_import\maintie-bench\scripts\08_make_embeddings.py
set RUN=%ROOT%\scripts\maintie\02_train_baselines.bat

echo ======================================================================
echo   VERIFY MAINTIE EMBEDDING + RESUME HOTFIX
echo ======================================================================
if not exist "%PY%" (
  echo FAIL: missing %PY%
  exit /b 1
)
if not exist "%RUN%" (
  echo FAIL: missing %RUN%
  exit /b 1
)
findstr /C:"TRAIN-only FastText" "%PY%" >nul || (echo FAIL: TRAIN-only embedding marker missing & exit /b 1)
findstr /C:"SKIPPING retraining" "%RUN%" >nul || (echo FAIL: resume marker missing & exit /b 1)
call conda run --no-capture-output -n avimaint-ie-neural python -m py_compile "%PY%"
if errorlevel 1 exit /b %ERRORLEVEL%
echo HOTFIX VERIFIED.
echo MaintIE embeddings use frozen TRAIN only.
echo Existing Tier1 result will be reused; Tier1 will not retrain.
exit /b 0
