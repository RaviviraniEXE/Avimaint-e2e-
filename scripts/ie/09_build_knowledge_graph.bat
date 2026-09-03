@echo off
setlocal
if "%~1"=="" (
  echo Usage: 09_build_knowledge_graph.bat path-to-full-corpus-predictions.json
  exit /b 2
)
pushd "%~dp0\..\..\legacy_import\maintenance-ie"
call conda run -n avimaint-ie-classical python scripts\13_build_kg.py --pred "%~1" --tokens outputs\kg\full_index.jsonl --name aviation
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
