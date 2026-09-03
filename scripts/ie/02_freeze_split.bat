@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
if exist outputs\splits.json (
  echo Frozen aviation split already exists. It will not be overwritten.
) else (
  if not exist outputs\gold\pilot.jsonl (
    echo ERROR: pilot.jsonl is missing. Run scripts\ie\01_import_annotations.bat first.
    popd
    exit /b 3
  )
  if not exist outputs\gold\round1.jsonl (
    echo ERROR: round1.jsonl is missing. Run scripts\ie\01_import_annotations.bat first.
    popd
    exit /b 3
  )
  call conda run -n avimaint-ie-classical python scripts\03_freeze_test.py --random-files outputs\gold\pilot.jsonl outputs\gold\round1.jsonl
)
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" exit /b %RESULT%
pushd "%~dp0\..\.."
call conda run -n avimaint-ie-classical python scripts\ie\audit_frozen_split.py
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
