@echo off
REM Starts the AviMaint-DSS dashboard. Double-click or run from a terminal.
cd /d "%~dp0"
python -m streamlit run app.py
pause

