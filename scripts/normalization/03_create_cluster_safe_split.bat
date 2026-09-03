@echo off
setlocal
pushd "%~dp0..\.."
conda run -n avimaint-normalization python -m avimaint.normalization split --config configs/normalization/split.yaml
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
