@echo off
title AviMaint dashboard
cd /d "%~dp0"
echo Activating conda env "avimaint-dash" and starting the dashboard ...
call conda activate avimaint-dash

REM NOTE: `conda activate` can report a nonzero exit code even on success,
REM so we do NOT gate on errorlevel. If you haven't created the env yet,
REM run setup_dashboard_conda.bat first (streamlit won't be found otherwise).

python -m streamlit run app.py
pause

