@echo off
setlocal EnableExtensions
set "ROOT=%~dp0..\.."

echo.
echo ======================================================================
echo   AviMaint SpERT Windows Compatibility Verification
echo ======================================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\setup\patch_spert_windows_timestamp.ps1" -VerifyOnly
if errorlevel 1 goto :failed

echo.
echo WINDOWS TIMESTAMP PATCH VERIFIED.
echo The official SpERT run directory no longer contains ':' characters.
echo.
exit /b 0

:failed
echo.
echo PATCH VERIFICATION FAILED.
echo Run scripts\setup\patch_spert_windows_timestamp.bat and retry.
exit /b 1
