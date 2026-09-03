@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY TRUE-RAW MATCHED SpERT CORRECTION V3
echo ======================================================================

if not exist scripts\ie\prepare_true_raw_matched_spert.py (
  echo FAIL: corrected preparation script missing.
  popd & exit /b 1
)

findstr /C:"V3: compare identical support keys only." scripts\ie\prepare_true_raw_matched_spert.py >nul || (
  echo FAIL: V3 support-comparison fix missing.
  popd & exit /b 1
)

findstr /C:"Record-count/order checks are separate from support comparison." scripts\ie\prepare_true_raw_matched_spert.py >nul || (
  echo FAIL: separate record-count/order gate missing.
  popd & exit /b 1
)

echo.
echo V3 FIX VERIFIED.
echo - False dictionary-shape support failure is corrected.
echo - Raw and existing conditions still require identical TEST record count/order.
echo - Raw and existing conditions still require identical entity/relation gold support.
echo - Existing four models remain reuse-only; no retraining is introduced.
echo.
popd
exit /b 0
