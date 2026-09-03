@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
if not exist outputs\splits.json (
  echo ERROR: final frozen split missing. Import random gold and run 02_freeze_split.bat.
  popd
  exit /b 3
)
call conda run -n avimaint-ie-classical python scripts\06_export_spert.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
