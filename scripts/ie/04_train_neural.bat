@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."
set "RAW=%ROOT%\data\aviation\raw\Aircraft_Annotation_DataFile.csv"
set "NORM=%ROOT%\data\aviation\processed\normalized_corpus.csv"

echo.
echo ======================================================================
echo   RQ3 FULL-SCHEMA NEURAL IE
echo   Tier 2: BiLSTM-CRF + neural RE
echo   Tier 3: DistilBERT NER + transformer RE
echo   Schema : FULL 9 entities / 11 relations
echo   Split  : frozen TRAIN / DEV / TEST
echo   Live   : epoch + batch progress, losses, DEV F1, patience, ETA
echo ======================================================================
echo.
echo [GPU PRECHECK] Requiring NVIDIA CUDA for all neural IE training...
call "%ROOT%\scripts\ie\00_verify_neural_gpu.bat"
if errorlevel 1 (
  echo ERROR: CUDA preflight failed. Neural training is intentionally blocked to prevent silent CPU fallback.
  exit /b 20
)
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo.

pushd "%ROOT%"
echo [PRECHECK] Auditing the frozen split before neural training...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\ie\audit_frozen_split.py
if errorlevel 1 (
  echo ERROR: frozen split audit failed.
  popd
  exit /b 1
)
popd

if not exist "%RAW%" (
  echo ERROR: canonical raw corpus is missing:
  echo   %RAW%
  exit /b 2
)
if not exist "%NORM%" (
  echo ERROR: canonical normalized corpus is missing:
  echo   %NORM%
  exit /b 2
)

pushd "%ROOT%\legacy_import\maintenance-ie"
if not exist outputs\splits.json (
  echo ERROR: final frozen split missing. Run scripts\ie\02_freeze_split.bat first.
  popd
  exit /b 3
)

if not exist outputs\embeddings\domain_ft.model (
  echo [PREP] Domain FastText embeddings are missing; training them once now...
  echo [PREP] Raw source : %RAW%
  echo [PREP] Norm source: %NORM%
  call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\08_make_embeddings.py --raw "%RAW%" --norm "%NORM%"
  if errorlevel 1 (
    echo ERROR: FastText preparation failed.
    popd
    exit /b 4
  )
) else (
  echo [PREP] Reusing existing domain FastText: outputs\embeddings\domain_ft.model
)

echo.
echo [TRAIN 1/2] Live Tier-2/Tier-3 training and compact evaluation...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\05_train_eval.py --tiers 2 3 --require-frozen-split --run-id aviation_neural
if errorlevel 1 (
  echo ERROR: full neural training/evaluation failed.
  popd
  exit /b 5
)

echo.
echo [REPORT 2/2] Building bootstrap tables and saving thesis models...
echo              Model internals retain the same epoch progress and early stopping output.
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\09_report.py --tiers 2 3 --bootstrap 1000 --save-models --run-id aviation_neural_final
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
  echo ERROR: final neural report failed with exit code %RESULT%.
  exit /b %RESULT%
)
exit /b 0
