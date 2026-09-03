@echo off
setlocal
cd /d "%~dp0..\.."
conda run --no-capture-output -n avimaint-spert python -u scripts\setup\patch_spert_safetensors_compat.py
exit /b %ERRORLEVEL%
