@echo off
setlocal
if "%~2"=="" (
  echo Usage: 00_import_additional_annotations.bat path-to-Label-Studio-export.json batch_name
  echo Use a name beginning with random_ for an unbiased pilot/round batch.
  exit /b 2
)
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
call conda run -n avimaint-ie-classical python scripts\02_import_gold.py --export "%~1" --name "%~2"
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
