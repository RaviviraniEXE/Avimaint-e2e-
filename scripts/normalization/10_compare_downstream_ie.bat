@echo off
setlocal
pushd "%~dp0\..\.."
if not exist legacy_import\maintenance-ie\outputs\splits.json (
  echo ERROR: representative aviation IE split is not frozen. Import random annotations first.
  popd
  exit /b 3
)
call conda run -n avimaint-ie-classical python scripts\ie\project_normalization_to_gold.py --min-coverage 0.97 --systems raw rules byt5 selective_byt5 rules_then_byt5
if errorlevel 1 goto :failed
pushd legacy_import\maintenance-ie
call conda run -n avimaint-ie-classical python scripts\07_normalization_ie.py
set RESULT=%ERRORLEVEL%
popd
popd
exit /b %RESULT%
:failed
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
