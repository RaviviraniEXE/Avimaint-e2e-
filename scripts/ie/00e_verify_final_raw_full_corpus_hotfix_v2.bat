@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY FINAL RAW SpERT FULL-CORPUS HOTFIX V2
echo ======================================================================

findstr /C:"--representation raw" scripts\ie\08_prepare_full_corpus.bat >nul || (
  echo FAIL: preparation BAT does not explicitly request raw.
  popd & exit /b 1
)
findstr /C:"gold token parity" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: gold representation parity gate is missing.
  popd & exit /b 1
)
findstr /C:"allowed_field_separators" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: source field-boundary audit is missing.
  popd & exit /b 1
)
findstr /C:"2026-08-28_12-27-06_489728\final_model" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (
  echo FAIL: selected Raw checkpoint is not pinned.
  popd & exit /b 1
)
if not exist scripts\ie\verify_full_corpus_spert.py (
  echo FAIL: strict full-corpus verifier is missing.
  popd & exit /b 1
)

echo.
echo V2 HOTFIX VERIFIED.
echo - normalized_corpus.csv::raw is used as the established combined RAW IE view.
echo - PROBLEM and ACTION source wording must be preserved exactly.
echo - Only the established inter-field separator may differ.
echo - All 1,600 manually reviewed gold token sequences must match the RAW view.
echo - Full-corpus inference remains pinned to the selected Raw full 9x11 SpERT checkpoint.
echo - No training is invoked.
echo.
popd
exit /b 0
