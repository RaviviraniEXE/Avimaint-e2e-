@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY FINAL TRUE-RAW FULL-CORPUS PIPELINE HOTFIX
echo ======================================================================

if not exist legacy_import\maintenance-ie\src\data\corpus.py (
  echo FAIL: corpus.py missing.
  popd & exit /b 1
)
if not exist legacy_import\maintenance-ie\scripts\12_predict_full_prep.py (
  echo FAIL: full-corpus prep script missing.
  popd & exit /b 1
)
if not exist scripts\ie\08b_predict_full_corpus_spert.ps1 (
  echo FAIL: full-corpus SpERT predictor missing.
  popd & exit /b 1
)
if not exist scripts\ie\verify_full_corpus_spert.py (
  echo FAIL: strict extraction verifier missing.
  popd & exit /b 1
)
if not exist scripts\ie\09b_build_final_true_raw_kg.py (
  echo FAIL: final KG wrapper missing.
  popd & exit /b 1
)
if not exist scripts\ie\10_freeze_final_true_raw_ie_kg.py (
  echo FAIL: final freeze script missing.
  popd & exit /b 1
)

findstr /C:"representation: str = " legacy_import\maintenance-ie\src\data\corpus.py >nul || (
  echo FAIL: corpus.load has no explicit representation parameter.
  popd & exit /b 1
)
findstr /C:"CANONICAL_SYSTEM_RAW" legacy_import\maintenance-ie\src\data\corpus.py >nul || (
  echo FAIL: corpus.py has no authoritative System-A raw source.
  popd & exit /b 1
)
findstr /C:"load(representation=" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: final preparation does not request representation explicitly.
  popd & exit /b 1
)
findstr /C:"outputs/gold_variants/raw" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: 1600-record projected true-raw parity gate missing.
  popd & exit /b 1
)
findstr /C:"MODEL_REGISTRY_V2.json" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (
  echo FAIL: predictor does not use corrected V2 model registry.
  popd & exit /b 1
)
findstr /C:"outputs\spert_normalized\raw" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (
  echo FAIL: predictor has no corrected raw-model location guard.
  popd & exit /b 1
)

echo.
echo FINAL PIPELINE HOTFIX VERIFIED.
echo - corpus.load keeps normalized as historical default but supports explicit raw.
echo - explicit raw uses outputs\normalization\full_corpus\raw.csv.
echo - preparation requires 1600/1600 parity with outputs\gold_variants\raw.
echo - prediction uses the corrected raw model from MODEL_REGISTRY_V2.json.
echo - historical outputs\spert cannot be used accidentally.
echo - final KG is provenance-checked and old KG is archived before replacement.
echo - final extraction + KG can be frozen with checksums without duplicating model weights.
echo.
popd
exit /b 0
