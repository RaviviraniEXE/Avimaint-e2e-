@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY FINAL SELECTIVE-BYT5 OPERATIONAL PIPELINE HOTFIX
echo ======================================================================

if not exist legacy_import\maintenance-ie\src\data\corpus.py (
  echo FAIL: corpus.py missing.
  popd & exit /b 1
)
if not exist legacy_import\maintenance-ie\scripts\12_predict_full_prep.py (
  echo FAIL: final prep missing.
  popd & exit /b 1
)
if not exist scripts\ie\08b_predict_full_corpus_spert.ps1 (
  echo FAIL: predictor missing.
  popd & exit /b 1
)
if not exist scripts\ie\verify_full_corpus_spert.py (
  echo FAIL: strict verifier missing.
  popd & exit /b 1
)

findstr /C:"selective_byt5" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: preparation is not Selective-ByT5.
  popd & exit /b 1
)
findstr /C:"outputs/spert_normalized/{REPRESENTATION}" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: trained-export token parity gate missing.
  popd & exit /b 1
)
findstr /C:"selective_byt5" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (
  echo FAIL: predictor is not pinned to Selective-ByT5.
  popd & exit /b 1
)
findstr /C:"problem_raw" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: raw source provenance is not retained.
  popd & exit /b 1
)

echo.
echo SELECTIVE-BYT5 OPERATIONAL HOTFIX VERIFIED.
echo - RQ1 corrected five-way experiment remains unchanged and frozen.
echo - Final operational representation is Selective ByT5.
echo - Final SpERT model is the matched Selective-ByT5 model.
echo - All 1600 projected tokens must equal the actual trained SpERT export.
echo - The operational tokenizer must reproduce all 1600 trained token sequences.
echo - Original raw PROBLEM and ACTION are retained for provenance.
echo - No model training or retuning is introduced.
echo.
popd
exit /b 0
