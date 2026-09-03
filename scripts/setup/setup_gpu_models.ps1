$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "Open Anaconda Prompt; conda is not available." }

# Fetch the official archived upstream source and record its exact commit.
& (Join-Path $PSScriptRoot "clone_official_spert.ps1")

# Normalization has its own fully pinned CUDA requirements file.
& (Join-Path $PSScriptRoot "setup_normalization_gpu.ps1")

foreach ($Key in @("ie-neural", "spert")) {
    & (Join-Path $PSScriptRoot "setup_one.ps1") $Key
    $Name = if ($Key -eq "ie-neural") { "avimaint-ie-neural" } else { "avimaint-spert" }
    & conda run -n $Name python -m pip install --force-reinstall `
        torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    & conda run -n $Name python -c "import torch; print('$Name', torch.__version__, 'CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
}
Write-Host "GPU model environments are ready and remain isolated."
