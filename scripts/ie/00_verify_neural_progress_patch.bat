@echo off
setlocal EnableExtensions
set "ROOT=%~dp0\..\.."

echo ======================================================================
echo   VERIFY NEURAL IE LIVE-PROGRESS + EARLY-STOPPING PATCH
echo ======================================================================

pushd "%ROOT%"
call conda run --no-capture-output -n avimaint-ie-neural python -u scripts\ie\audit_frozen_split.py
if errorlevel 1 goto :failed_root
popd

pushd "%ROOT%\legacy_import\maintenance-ie"
call conda run --no-capture-output -n avimaint-ie-neural python -u -c "from src.progress import EpochProgress; from src.models.bilstm_crf import BiLSTMCRF; from src.models.relation_bilstm import NeuralRelationClassifier; from src.models.transformer_ie import TransformerNER, TransformerRE; import torch; print('PATCH IMPORTS OK'); print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
set RESULT=%ERRORLEVEL%
popd
if not "%RESULT%"=="0" goto :failed

echo.
echo PATCH VERIFIED SUCCESSFULLY.
echo Next command:
echo   scripts\ie\04b_train_core_schema_neural.bat
exit /b 0

:failed_root
set RESULT=%ERRORLEVEL%
popd
:failed
echo PATCH VERIFICATION FAILED with exit code %RESULT%.
pause
exit /b %RESULT%
