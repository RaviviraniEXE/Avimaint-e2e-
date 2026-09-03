@echo off
setlocal EnableExtensions
set "ENV=avimaint-ie-neural"

echo.
echo ======================================================================
echo   AviMaint IE Neural - GPU Preflight
 echo  This check MUST pass before BiLSTM/Transformer training.
echo ======================================================================
echo.

call conda run --no-capture-output -n %ENV% python -u -c "import sys,torch; print('torch=',torch.__version__); print('torch CUDA build=',torch.version.cuda); print('cuda available=',torch.cuda.is_available()); print('GPU count=',torch.cuda.device_count()); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU'); sys.exit(20) if not torch.cuda.is_available() else None; a=torch.randn((2048,2048),device='cuda'); b=a@a; torch.cuda.synchronize(); print('CUDA compute test=PASS'); print('peak allocated MB=',round(torch.cuda.max_memory_allocated()/1024/1024,1))"
if errorlevel 1 (
  echo.
  echo ERROR: avimaint-ie-neural cannot use CUDA.
  echo Run: scripts\setup\fix_ie_neural_cuda.bat
  exit /b 20
)

echo.
nvidia-smi

echo.
echo GPU PREFLIGHT PASSED.
exit /b 0
