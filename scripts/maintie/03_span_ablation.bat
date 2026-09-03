@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"

echo ======================================================================
echo   MAINTIE BIO-vs-SPAN NER ABLATION

echo   Reuses already-trained Tier2/Tier3 BIO TEST predictions.

echo   Trains ONLY the two new span-NER ablation heads.

echo ======================================================================

call conda run --no-capture-output -n avimaint-ie-neural python -c "import torch; print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); assert torch.cuda.is_available(), 'CUDA REQUIRED for span ablation'"
if errorlevel 1 goto :failed

call conda run --no-capture-output -n avimaint-ie-neural python scripts\11_span_ner.py --encoders bilstm transformer --max-span 10 --epochs 30 --reuse-bio-dir outputs\predictions\maintie_neural --require-reuse-bio
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%

:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
