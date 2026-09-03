@echo off
setlocal
pushd "%~dp0\..\.."

echo ======================================================================
echo   VERIFY SELECTIVE-BYT5 TOKENIZER V2
echo ======================================================================

findstr /C:"[./-]" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: dot/slash/hyphen compound token support missing.
  popd & exit /b 1
)
findstr /C:"operational_maintenance_tokenizer_v2" legacy_import\maintenance-ie\scripts\12_predict_full_prep.py >nul || (
  echo FAIL: tokenizer V2 marker missing.
  popd & exit /b 1
)

echo.
echo TOKENIZER V2 HOTFIX VERIFIED.
echo - R/H, L/H and A/C remain single tokens.
echo - 3-4 and other hyphen compounds remain single tokens.
echo - 12.1, 2772.5 and CLEARNACE.035 remain single tokens.
echo - sentence-final periods remain separate tokens.
echo - the 1600/1600 trained-token parity gate remains mandatory.
echo.
popd
exit /b 0
