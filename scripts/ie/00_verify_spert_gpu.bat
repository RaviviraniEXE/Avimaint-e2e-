@echo off
setlocal EnableExtensions
set "ENV=avimaint-spert"
set "ROOT=%~dp0..\.."

echo.
echo ======================================================================
echo   AviMaint SpERT GPU Preflight
echo ======================================================================
echo.

call conda run --no-capture-output -n %ENV% python -u -c "import sys,torch,transformers; print('torch=',torch.__version__); print('torch CUDA build=',torch.version.cuda); print('transformers=',transformers.__version__); print('cuda available=',torch.cuda.is_available()); print('GPU count=',torch.cuda.device_count()); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU'); sys.exit(20) if not torch.cuda.is_available() else None; a=torch.randn((1024,1024),device='cuda'); b=a@a; torch.cuda.synchronize(); print('CUDA compute test=PASS')"
if errorlevel 1 goto :failed

if not exist "%ROOT%\external\spert\spert.py" (
  echo WARNING: official SpERT source is not installed yet.
  echo Run scripts\setup\clone_official_spert.bat before training.
) else (
  echo Official SpERT source=FOUND
)

echo.
echo GPU PREFLIGHT PASSED.
echo.
exit /b 0

:failed
echo.
echo GPU PREFLIGHT FAILED.
echo Run scripts\setup\fix_spert_cuda.bat and retry.
exit /b 1
