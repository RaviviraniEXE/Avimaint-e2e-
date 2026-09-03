@echo off
setlocal EnableExtensions
set "ENV=avimaint-ie-neural"

echo.
echo ======================================================================
echo   AviMaint IE Neural - CUDA Repair
echo   Target: PyTorch 2.5.1 + CUDA 12.1 for NVIDIA GPU
echo ======================================================================
echo.

echo [1/4] Current PyTorch state...
call conda run --no-capture-output -n %ENV% python -u -c "import torch; print('torch=',torch.__version__); print('torch CUDA build=',torch.version.cuda); print('cuda available=',torch.cuda.is_available()); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

echo.
echo [2/4] Reinstalling the CUDA-enabled PyTorch wheel...
call conda run --no-capture-output -n %ENV% python -m pip uninstall -y torch torchvision torchaudio
if errorlevel 1 goto :failed
call conda run --no-capture-output -n %ENV% python -m pip install --no-cache-dir --force-reinstall torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :failed

echo.
echo [3/4] Verifying CUDA and allocating a real tensor on the RTX GPU...
call conda run --no-capture-output -n %ENV% python -u -c "import sys,torch; print('torch=',torch.__version__); print('torch CUDA build=',torch.version.cuda); print('cuda available=',torch.cuda.is_available()); print('GPU count=',torch.cuda.device_count()); print('device=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU'); sys.exit(20) if not torch.cuda.is_available() else None; x=torch.randn((1024,1024),device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA tensor test=PASS'); print('allocated_MB=',round(torch.cuda.memory_allocated()/1024/1024,1)); print('total_VRAM_GB=',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))"
if errorlevel 1 goto :failed

echo.
echo [4/4] NVIDIA process view...
nvidia-smi

echo.
echo ======================================================================
echo   CUDA REPAIR COMPLETE
 echo  Next: scripts\ie\00_verify_neural_gpu.bat
 echo  Then: scripts\ie\04b_train_core_schema_neural.bat
 echo ======================================================================
exit /b 0

:failed
echo.
echo ERROR: CUDA repair failed. Do not start neural IE training.
exit /b 1
