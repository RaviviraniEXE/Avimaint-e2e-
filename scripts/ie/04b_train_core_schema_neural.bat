@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."
set "RAW=%ROOT%\data\aviation\raw\Aircraft_Annotation_DataFile.csv"
set "NORM=%ROOT%\data\aviation\processed\normalized_corpus.csv"

echo.
echo ======================================================================
echo   RQ2 CORE-SCHEMA NEURAL IE
echo   Tier 2A: BiLSTM-CRF NER ^(char-CNN + domain FastText^)
echo   Tier 2B: Neural relation classifier
echo   Tier 3A: DistilBERT token-classification NER
echo   Tier 3B: DistilBERT relation classifier
echo   Schema : CORE 8 entities / 10 relations
echo   Split  : frozen TRAIN / DEV / TEST - TEST never used for stopping
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
  echo Run/finalize the full-corpus normalization artifact before Tier 2.
  exit /b 2
)

pushd "%ROOT%\legacy_import\maintenance-ie"
if not exist outputs\splits.json (
  echo ERROR: final frozen split missing. Run scripts\ie\02_freeze_split.bat first.
  popd
  exit /b 3
)

if not exist outputs\embeddings\domain_ft.model (
  echo.
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
echo [TRAIN] Starting CORE neural experiment...
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\05_train_eval.py --tiers 2 3 --require-frozen-split --run-id core_neural --gold-glob "outputs/gold_core/*.jsonl" --schema-path config/schema_core.yaml --report-name ie_results_core_neural
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
  echo.
  echo ERROR: core neural IE run failed with exit code %RESULT%.
  echo The JSONL trace preserves completed epochs for diagnosis.
  exit /b %RESULT%
)

echo.
echo ======================================================================
echo   CORE NEURAL RUN COMPLETE
echo   Results : legacy_import\maintenance-ie\outputs\reports\ie_results_core_neural.json
echo   Log     : legacy_import\maintenance-ie\outputs\reports\ie_results_core_neural_log.csv
echo   Manifest: legacy_import\maintenance-ie\outputs\reports\ie_results_core_neural_latest_training_manifest.json
echo   Trace   : legacy_import\maintenance-ie\outputs\reports\training_logs\
echo ======================================================================
exit /b 0
