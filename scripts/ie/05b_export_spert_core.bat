@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
call conda run -n avimaint-ie-classical python scripts\06_export_spert.py --gold-glob "outputs/gold_core/*.jsonl" --schema-path config/schema_core.yaml --output-dir outputs/spert_core
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" pause
exit /b %RESULT%

