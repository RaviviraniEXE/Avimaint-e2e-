@echo off
setlocal
pushd "%~dp0\..\.."
echo Rechecking selected ByT5 system with protected-token and numeric grounding safeguards
conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_gold.yaml --split validation --system selective_byt5
if errorlevel 1 goto :failed
conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_gold.yaml --split validation --system selective_byt5
if errorlevel 1 goto :failed
echo Selected-system validation recheck completed.
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
