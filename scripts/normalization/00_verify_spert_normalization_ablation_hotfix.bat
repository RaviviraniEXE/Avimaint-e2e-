@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY NORMALIZATION -^> FROZEN SpERT HOTFIX
echo ======================================================================

for %%F in (
  "scripts\ie\project_normalization_to_gold.py"
  "scripts\ie\prepare_normalization_spert_ablation.py"
  "scripts\ie\evaluate_normalization_spert_ablation.py"
  "scripts\ie\run_normalization_spert_ablation.ps1"
  "scripts\normalization\11_compare_downstream_spert_frozen.bat"
  "scripts\ie\06_predict_existing_spert.ps1"
  "legacy_import\maintenance-ie\outputs\spert\test.json"
  "outputs\normalization\full_corpus\raw.csv"
  "outputs\normalization\full_corpus\rules.csv"
  "outputs\normalization\full_corpus\byt5.csv"
  "outputs\normalization\full_corpus\selective_byt5.csv"
  "outputs\normalization\full_corpus\rules_then_byt5.csv"
) do (
  if not exist %%F (
    echo MISSING: %%~F
    popd
    exit /b 2
  )
)

call conda run --no-capture-output -n avimaint-spert python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0))"
if errorlevel 1 (
  echo ERROR: avimaint-spert CUDA verification failed.
  popd
  exit /b 3
)

call conda run --no-capture-output -n avimaint-ie-classical python -m py_compile scripts\ie\project_normalization_to_gold.py scripts\ie\prepare_normalization_spert_ablation.py scripts\ie\evaluate_normalization_spert_ablation.py
if errorlevel 1 (
  echo ERROR: Python hotfix compile check failed.
  popd
  exit /b 4
)

echo.
echo HOTFIX PRECHECK PASSED.
echo Next:
echo   scripts\normalization\11_compare_downstream_spert_frozen.bat
popd
exit /b 0
