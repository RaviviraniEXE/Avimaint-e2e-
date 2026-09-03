@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."
set "RAW=%ROOT%\data\aviation\raw\Aircraft_Annotation_DataFile.csv"
set "NORM=%ROOT%\data\aviation\processed\normalized_corpus.csv"

echo ======================================================================
echo   VERIFY NEURAL CORPUS-PATH HOTFIX
echo ======================================================================
if not exist "%RAW%" (
  echo FAIL: missing %RAW%
  exit /b 2
)
if not exist "%NORM%" (
  echo FAIL: missing %NORM%
  exit /b 2
)

echo [OK] Raw corpus found.
echo [OK] Normalized corpus found.

pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\08_make_embeddings.py --raw "%RAW%" --norm "%NORM%" --check-only
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
  echo FAIL: corpus loader preflight failed with exit code %RESULT%.
  exit /b %RESULT%
)

echo.
echo PATCH VERIFIED SUCCESSFULLY.
echo Re-run:
echo   scripts\ie\04b_train_core_schema_neural.bat
exit /b 0
