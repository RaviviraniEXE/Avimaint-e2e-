@echo off
setlocal
pushd "%~dp0..\.."
conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization train --config configs/normalization/byt5_gold.yaml
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
