@echo off
setlocal
pushd "%~dp0..\.."
conda run -n avimaint-normalization python -m avimaint.normalization train --config configs/normalization/byt5_silver_then_gold.yaml
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
