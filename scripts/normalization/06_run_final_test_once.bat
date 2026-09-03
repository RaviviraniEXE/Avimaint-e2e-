@echo off
setlocal EnableDelayedExpansion
if /I not "%~1"=="CONFIRM_TEST" (
  echo Final test is locked. First select the final configuration using validation only.
  echo Run: scripts\normalization\06_run_final_test_once.bat CONFIRM_TEST
  exit /b 2
)
pushd "%~dp0..\.."
if not exist "outputs\normalization\selection\normalization_selection_manifest.json" (
  echo Missing frozen validation selection manifest.
  echo Run: scripts\normalization\05e_freeze_validation_selection.bat
  popd
  exit /b 3
)
conda run --no-capture-output -n avimaint-normalization python -u scripts\normalization\freeze_validation_selection.py --verify
if errorlevel 1 goto :failed
for %%S in (raw most_frequent_replacement rules byt5 selective_byt5 rules_then_byt5) do (
  echo Running final test prediction and evaluation for %%S
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization predict --config configs/normalization/byt5_gold.yaml --split test --system %%S
  if errorlevel 1 goto :failed
  conda run --no-capture-output -n avimaint-normalization python -u -m avimaint.normalization evaluate --config configs/normalization/byt5_gold.yaml --split test --system %%S
  if errorlevel 1 goto :failed
)
popd
exit /b 0

:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
