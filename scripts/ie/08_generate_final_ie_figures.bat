@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo.
echo ======================================================================
echo   GENERATE FINAL AVIATION IE FIGURES FROM EXISTING RESULTS ONLY
echo   NO MODEL TRAINING / NO RETRAINING
echo   Core 8/10 + Full 9/11 + Core-vs-Full comparison
echo   Old outputs\reports\figures are preserved
echo ======================================================================
echo.

pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run --no-capture-output -n avimaint-ie-classical python -u scripts\13_generate_final_figures_existing.py
set RESULT=%ERRORLEVEL%
popd

if not "%RESULT%"=="0" goto :failed

echo.
echo ======================================================================
echo   FINAL FIGURE GENERATION COMPLETE - NO TRAINING WAS PERFORMED
echo   Figures : legacy_import\maintenance-ie\outputs\reports\final_figures\
echo   Tables  : legacy_import\maintenance-ie\outputs\reports\final_tables\
echo   Manifest: legacy_import\maintenance-ie\outputs\reports\FINAL_IE_FIGURE_MANIFEST.json
echo ======================================================================
exit /b 0

:failed
echo.
echo ERROR: final IE figure generation failed with exit code %RESULT%.
echo IMPORTANT: no model training was started by this script.
pause
exit /b %RESULT%
