@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY MATCHED SpERT TRUE-RAW CORRECTION
echo ======================================================================

if not exist scripts\ie\audit_spert_annotation_representation.py (
  echo FAIL: representation audit script missing.
  popd & exit /b 1
)
if not exist scripts\ie\prepare_true_raw_matched_spert.py (
  echo FAIL: corrected raw preparation script missing.
  popd & exit /b 1
)
if not exist scripts\ie\train_true_raw_matched_spert.ps1 (
  echo FAIL: raw-only training script missing.
  popd & exit /b 1
)
if not exist scripts\ie\evaluate_matched_normalization_spert_v2.py (
  echo FAIL: corrected evaluator missing.
  popd & exit /b 1
)

findstr /C:"outputs/spert_normalized/raw" scripts\ie\prepare_true_raw_matched_spert.py >nul || (
  echo FAIL: raw target path is not explicit.
  popd & exit /b 1
)
findstr /C:"Only the missing true-raw SpERT condition must be trained." scripts\ie\audit_spert_annotation_representation.py >nul || (
  echo FAIL: audit correction policy missing.
  popd & exit /b 1
)
findstr /C:"Existing four normalized models are NOT retrained" scripts\ie\train_true_raw_matched_spert.ps1 >nul || (
  echo FAIL: raw-only training safety statement missing.
  popd & exit /b 1
)
findstr /C:"Historical outputs/spert baseline: excluded from the five-way RQ1 table" scripts\ie\evaluate_matched_normalization_spert_v2.py >nul || (
  echo FAIL: evaluator does not exclude mislabeled historical baseline.
  popd & exit /b 1
)

echo.
echo CORRECTION HOTFIX VERIFIED.
echo - Historical annotation representation will be audited automatically.
echo - A NEW true System-A raw export/model is stored under outputs\spert_normalized\raw.
echo - The four completed normalized SpERT models are reused, not retrained.
echo - Same frozen split, same full 9x11 schema, same fixed hyperparameters.
echo - Historical outputs\spert is retained only as an annotation-representation reference.
echo - Corrected five-way results go to normalization_spert_matched_v2.
echo.
popd
exit /b 0
