@echo off
setlocal EnableDelayedExpansion
pushd "%~dp0\..\.."
for %%S in (most_frequent_replacement selective_byt5) do (
  echo Running optional validation ablation for %%S
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_gold.yaml --split validation --system %%S
  if errorlevel 1 goto :failed
)
popd
exit /b 0
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
