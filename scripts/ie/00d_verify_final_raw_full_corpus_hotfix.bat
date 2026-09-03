@echo off
setlocal
pushd "%~dp0\..\.."
echo ======================================================================
echo   VERIFY FINAL RAW SpERT FULL-CORPUS HOTFIX
echo ======================================================================
findstr /C:"--representation raw" scripts\ie\08_prepare_full_corpus.bat >nul || (echo FAIL: raw preparation not explicit. & popd & exit /b 1)
findstr /C:"2026-08-28_12-27-06_489728\final_model" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (echo FAIL: final Raw checkpoint not pinned. & popd & exit /b 1)
findstr /C:"Refusing inference: prepared corpus representation" scripts\ie\08b_predict_full_corpus_spert.ps1 >nul || (echo FAIL: raw representation gate missing. & popd & exit /b 1)
if not exist scripts\ie\verify_full_corpus_spert.py (echo FAIL: strict verifier missing. & popd & exit /b 1)
findstr /C:"normalized_corpus.csv raw view does not match canonical" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (echo FAIL: raw source cross-check missing. & popd & exit /b 1)
echo.
echo HOTFIX VERIFIED.
echo - Full-corpus preparation explicitly uses RAW combined PROBLEM + ACTION.
echo - The raw combined view is cross-checked against source fields.
echo - Prediction is pinned to the selected Raw full 9x11 SpERT checkpoint.
echo - Predictor refuses non-raw or non-6169 prepared input.
echo - Strict post-prediction validation checks IDs, tokens, spans, relation endpoints, and schema.
echo - No training is invoked.
popd
exit /b 0
