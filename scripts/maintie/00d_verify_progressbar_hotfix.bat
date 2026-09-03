@echo off
setlocal
pushd "%~dp0\..\..\legacy_import\maintie-bench"

echo ======================================================================
echo   MAINTIE PROGRESS-BAR HOTFIX VERIFY - NO TRAINING
echo ======================================================================

echo [1/3] Python syntax...
call conda run -n avimaint-ie-neural python -m py_compile src\models\bilstm_crf.py src\models\relation_bilstm.py src\models\transformer_ie.py src\models\span_ner.py src\models\embeddings.py scripts\08_make_embeddings.py
if errorlevel 1 goto :failed

echo [2/3] tqdm availability...
call conda run --no-capture-output -n avimaint-ie-neural python -c "import tqdm; print('tqdm=',tqdm.__version__); print('tqdm import=PASS')"
if errorlevel 1 goto :failed

echo [3/3] Progress markers...
findstr /C:"Tier2 NER epoch" src\models\bilstm_crf.py >nul || goto :failed
findstr /C:"Tier2 RE  epoch" src\models\relation_bilstm.py >nul || goto :failed
findstr /C:"Tier3 NER epoch" src\models\transformer_ie.py >nul || goto :failed
findstr /C:"Tier3 RE  epoch" src\models\transformer_ie.py >nul || goto :failed
findstr /C:"FastText TRAIN-only" src\models\embeddings.py >nul || goto :failed
findstr /C:"BiLSTM SPAN epoch" src\models\span_ner.py >nul || goto :failed
findstr /C:"Transformer SPAN epoch" src\models\span_ner.py >nul || goto :failed

echo.
echo PROGRESS-BAR HOTFIX VERIFIED.
echo NO TRAINING WAS PERFORMED.
echo Tier1 stable result remains reusable.
echo Run scripts\maintie\02_train_baselines.bat
popd
exit /b 0

:failed
echo.
echo ERROR: progress-bar hotfix verification failed.
popd
exit /b 1
