@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo ======================================================================
echo   VERIFY CLASSICAL IE LIVE-PROGRESS HOTFIX
echo ======================================================================

pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run --no-capture-output -n avimaint-ie-classical python -u -m py_compile scripts\05_train_eval.py scripts\09_report.py scripts\10_significance.py src\progress.py src\models\crf_ner.py src\models\relation_logreg.py
if errorlevel 1 goto :failed_legacy
call conda run --no-capture-output -n avimaint-ie-classical python -u -c "from src.progress import LiveProgress; from src.models.crf_ner import CRFTagger; from src.models.relation_logreg import RelationClassifier; print('Imports OK: progress + CRF + LogReg')"
if errorlevel 1 goto :failed_legacy
popd

pushd "%ROOT%"
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\ie\audit_frozen_split.py
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" goto :failed

echo.
echo PATCH VERIFIED SUCCESSFULLY.
echo Next core command: scripts\ie\03b_train_core_schema_classical.bat
exit /b 0

:failed_legacy
set RESULT=%ERRORLEVEL%
popd

:failed
echo.
echo ERROR: verification failed with exit code %RESULT%.
pause
exit /b %RESULT%
