@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
echo ======================================================================
echo   PREPARE FINAL 6169 SELECTIVE-BYT5 CORPUS FOR MATCHED SpERT
echo   Raw PROBLEM/ACTION will be retained for provenance
echo ======================================================================
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\12_predict_full_prep.py --expected-records 6169 --expected-gold-records 1600
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
