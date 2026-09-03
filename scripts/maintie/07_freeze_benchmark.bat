@echo off
setlocal
cd /d "%~dp0..\..\"

set "SRC=legacy_import\maintie-bench\outputs"
set "DST=outputs\frozen\maintie_benchmark"

if not exist "%SRC%\reports\final_benchmark\FINAL_MAINTIE_MANIFEST.json" (
  echo ERROR: final benchmark manifest missing.
  echo Run scripts\maintie\06_finalize_benchmark_report.bat first.
  exit /b 1
)

echo ======================================================================
echo   FREEZE MAINTIE BENCHMARK EVIDENCE
echo ======================================================================

if exist "%DST%" (
  echo ERROR: "%DST%" already exists.
  echo Refusing to overwrite an existing frozen benchmark.
  exit /b 1
)

mkdir "%DST%\reports" >nul
mkdir "%DST%\spert" >nul
mkdir "%DST%\predictions" >nul

xcopy "%SRC%\reports\final_benchmark" "%DST%\reports\final_benchmark\" /E /I /Y >nul
copy "%SRC%\reports\ie_results__maintie_tier1.json" "%DST%\reports\" >nul
copy "%SRC%\reports\ie_results__maintie_tier1_manifest.json" "%DST%\reports\" >nul
copy "%SRC%\reports\ie_results__maintie_neural.json" "%DST%\reports\" >nul
copy "%SRC%\reports\ie_results__maintie_neural_manifest.json" "%DST%\reports\" >nul
copy "%SRC%\reports\maintie_overlap_audit.json" "%DST%\reports\" >nul
if exist "%SRC%\reports\tables\span_ner_ablation.csv" copy "%SRC%\reports\tables\span_ner_ablation.csv" "%DST%\reports\" >nul
copy "%SRC%\reports\spert_test.json" "%DST%\reports\" >nul

copy "%SRC%\spert\train.json" "%DST%\spert\" >nul
copy "%SRC%\spert\dev.json" "%DST%\spert\" >nul
copy "%SRC%\spert\test.json" "%DST%\spert\" >nul
copy "%SRC%\spert\predictions_test.json" "%DST%\spert\" >nul
copy "%SRC%\spert\avimaint_types.json" "%DST%\spert\" >nul
copy "%SRC%\spert\avimaint_spert.conf" "%DST%\spert\" >nul

if exist "%SRC%\predictions\maintie_tier1" xcopy "%SRC%\predictions\maintie_tier1" "%DST%\predictions\maintie_tier1\" /E /I /Y >nul
if exist "%SRC%\predictions\maintie_neural" xcopy "%SRC%\predictions\maintie_neural" "%DST%\predictions\maintie_neural\" /E /I /Y >nul
if exist "%SRC%\predictions\span_ablation" xcopy "%SRC%\predictions\span_ablation" "%DST%\predictions\span_ablation\" /E /I /Y >nul

powershell -NoProfile -Command "Get-ChildItem '%DST%' -File -Recurse | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Get-FileHash -Algorithm SHA256 | ForEach-Object { '{0}  {1}' -f $_.Hash,$_.FullName } | Out-File -Encoding utf8 '%DST%\SHA256SUMS.txt'"
if errorlevel 1 exit /b %errorlevel%

echo.
echo MAINTIE BENCHMARK FROZEN -> %DST%
echo Do not tune against this frozen TEST result.
endlocal
