@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0\..\.."
for %%S in (rules rules_then_byt5) do (
  echo Re-running corrected validation prediction and evaluation for %%S
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
)
echo Corrected rule-system validation rerun complete.
popd
exit /b 0

:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
