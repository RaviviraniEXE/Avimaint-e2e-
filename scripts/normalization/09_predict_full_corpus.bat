@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0\..\.."
set SYSTEMS=raw selective_byt5
if /I "%~1"=="ALL" set SYSTEMS=raw rules byt5 selective_byt5 rules_then_byt5
for %%S in (%SYSTEMS%) do (
  echo Full-corpus inference for %%S
  call conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict-corpus --config configs/normalization/byt5_gold.yaml --system %%S
  if errorlevel 1 goto :failed
)
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
