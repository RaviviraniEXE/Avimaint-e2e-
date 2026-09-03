@echo off
setlocal
set ROOT=%~dp0\..\..
set BENCH=%ROOT%\legacy_import\maintie-bench

echo ======================================================================
echo   VERIFY MAINTIE BENCHMARK METHODOLOGY + SAFETY HOTFIX

echo ======================================================================

findstr /C:"original_full_spans_and_relations" "%BENCH%\scripts\05_train_eval.py" >nul || goto :failed
findstr /C:"reuse-bio-dir" "%BENCH%\scripts\11_span_ner.py" >nul || goto :failed
findstr /C:"BIO REPRESENTATION AUDIT" "%BENCH%\scripts\00b_overlap_audit.py" >nul || goto :failed
findstr /C:"CUDA REQUIRED" "%~dp0\02_train_baselines.bat" >nul || goto :failed

echo [1/2] Static patch markers: PASS
call conda run -n avimaint-ie-classical python -m py_compile "%BENCH%\scripts\05_train_eval.py" "%BENCH%\scripts\00b_overlap_audit.py"
if errorlevel 1 goto :failed
call conda run -n avimaint-ie-neural python -m py_compile "%BENCH%\scripts\11_span_ner.py"
if errorlevel 1 goto :failed
echo [2/2] Python syntax: PASS

echo.
echo HOTFIX VERIFIED.
echo MaintIE BIO baselines will be scored against FULL original spans.
echo Stable result files and TEST predictions will be preserved.
echo Span ablation will reuse baseline BIO predictions and avoid duplicate BIO retraining.
exit /b 0

:failed
echo HOTFIX VERIFICATION FAILED.
exit /b 1
