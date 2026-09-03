@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"

echo ======================================================================
echo   MAINTIE BASELINES - RESUME-SAFE / FULL GOLD EVALUATION
echo   Tier1: CRF + LogReg
echo   Tier2: BiLSTM-CRF + Neural RE
echo   Tier3: DistilBERT NER + Transformer RE
echo   BIO models are evaluated against ORIGINAL full MaintIE spans.
echo   Existing stable completed tiers are reused; no needless retraining.
echo   LIVE PROGRESS BARS: FastText / Tier2 NER+RE / Tier3 NER+RE
echo ======================================================================

if exist "outputs\reports\ie_results__maintie_tier1.json" (
  echo [Tier1] stable result already exists - SKIPPING retraining.
  echo         outputs\reports\ie_results__maintie_tier1.json
) else (
  call conda run --no-capture-output -n avimaint-ie-classical python scripts\05_train_eval.py --tiers 1 --tune --run-id maintie_tier1
  if errorlevel 1 goto :failed
)

if exist "outputs\reports\ie_results__maintie_neural.json" (
  echo.
  echo [Tier2/Tier3] stable neural result already exists - SKIPPING retraining.
  echo               outputs\reports\ie_results__maintie_neural.json
  popd
  exit /b 0
)

echo.
echo [GPU PREFLIGHT] avimaint-ie-neural
call conda run --no-capture-output -n avimaint-ie-neural python -c "import torch, tqdm; print('torch=',torch.__version__); print('tqdm=',tqdm.__version__); print('CUDA build=',torch.version.cuda); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); assert torch.cuda.is_available(), 'CUDA REQUIRED: refusing silent CPU fallback'; x=torch.randn((512,512),device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA compute test=PASS')"
if errorlevel 1 goto :failed

echo.
echo [EMBEDDINGS] MaintIE frozen TRAIN split only
call conda run --no-capture-output -n avimaint-ie-neural python scripts\08_make_embeddings.py
if errorlevel 1 goto :failed

call conda run --no-capture-output -n avimaint-ie-neural python scripts\05_train_eval.py --tiers 2 3 --run-id maintie_neural
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%

:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
