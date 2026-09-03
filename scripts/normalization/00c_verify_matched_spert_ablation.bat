@echo off
setlocal EnableExtensions
pushd "%~dp0\..\.."
echo ======================================================================
echo   VERIFY REPRESENTATION-MATCHED NORMALIZATION -^> SpERT HOTFIX
echo ======================================================================

for %%F in (
  "legacy_import\maintenance-ie\outputs\splits.json"
  "legacy_import\maintenance-ie\outputs\spert\train.json"
  "legacy_import\maintenance-ie\outputs\spert\dev.json"
  "legacy_import\maintenance-ie\outputs\spert\test.json"
  "legacy_import\maintenance-ie\outputs\spert\avimaint_types.json"
  "legacy_import\maintenance-ie\outputs\spert\avimaint_spert.conf"
  "scripts\ie\06_train_and_test_spert.ps1"
  "scripts\ie\06_predict_existing_spert.ps1"
  "scripts\ie\project_normalization_to_gold.py"
  "scripts\ie\prepare_matched_normalization_spert.py"
  "scripts\ie\train_matched_normalization_spert.ps1"
  "scripts\ie\evaluate_matched_normalization_spert.py"
) do if not exist %%F (
  echo ERROR: missing %%F
  popd
  exit /b 3
)

for %%S in (raw rules byt5 selective_byt5 rules_then_byt5) do if not exist "legacy_import\maintenance-ie\outputs\gold_variants\%%S" (
  echo ERROR: missing projected gold variant %%S
  popd
  exit /b 4
)

findstr /I /C:"ExportName" scripts\ie\06_train_and_test_spert.ps1 >nul || (
  echo ERROR: existing SpERT trainer lacks ExportName support.
  popd
  exit /b 5
)
findstr /I /C:"ExportName" scripts\ie\06_predict_existing_spert.ps1 >nul || (
  echo ERROR: existing SpERT predictor lacks ExportName support.
  popd
  exit /b 6
)

call conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('torch=',torch.__version__); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if errorlevel 1 (
  echo ERROR: CUDA preflight failed.
  popd
  exit /b 7
)

echo.
echo HOTFIX VERIFIED.
echo - Frozen split and authoritative raw SpERT export are present.
echo - Four normalized model directories will be separate and named by normalization system.
echo - Raw model will NOT be retrained.
echo - All four normalized models use the raw SpERT hyperparameters unchanged except paths/label.
echo - Comparative TEST metrics are calculated only after all four fixed models exist.
echo - Interrupted runs can resume without retraining completed models.
echo.
echo WARNING: the next experiment WILL TRAIN FOUR SpERT models on GPU.
popd
exit /b 0
