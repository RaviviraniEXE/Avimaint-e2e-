@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0..\.."
for %%S in (raw rules byt5 rules_then_byt5) do (
  echo Running validation prediction and evaluation for %%S
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
)
echo Primary comparison complete: raw, rules, ByT5, rules-then-ByT5.
echo Optional ablations: run 05b_run_validation_ablations.bat.
popd
exit /b 0

:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
