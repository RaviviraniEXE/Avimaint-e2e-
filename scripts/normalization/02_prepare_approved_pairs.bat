@echo off
setlocal
pushd "%~dp0..\.."
conda run -n avimaint-normalization python -m avimaint.normalization prepare --config configs/normalization/data.yaml
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
