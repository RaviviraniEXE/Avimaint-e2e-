@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0\..\.."
call conda run --no-capture-output -n avimaint-normalization python -u scripts\normalization\build_expert_sensitivity_set.py
if errorlevel 1 goto :failed
for %%S in (raw rules byt5 selective_byt5 rules_then_byt5) do (
  echo Expert-expanded sensitivity evaluation for %%S
  call conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_expert_sensitivity.yaml --split sensitivity --system %%S
  if errorlevel 1 goto :failed
  call conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_expert_sensitivity.yaml --split sensitivity --system %%S
  if errorlevel 1 goto :failed
)
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
