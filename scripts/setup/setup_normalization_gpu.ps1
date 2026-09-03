$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvironmentFile = Join-Path $ProjectRoot "envs\normalization\environment.yml"
$RequirementsFile = Join-Path $ProjectRoot "envs\normalization\requirements-gpu-cu121.txt"
$EnvironmentName = "avimaint-normalization"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda is not available. Open Anaconda Prompt or initialize Conda for PowerShell."
}

$KnownEnvironments = (conda env list --json | ConvertFrom-Json).envs
$Exists = $KnownEnvironments | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName }
if ($Exists) {
    conda env update --name $EnvironmentName --file $EnvironmentFile --prune
}
else {
    conda env create --file $EnvironmentFile
}

conda run --name $EnvironmentName python -m pip install --upgrade pip
conda run --name $EnvironmentName python -m pip install --requirement $RequirementsFile
conda run --name $EnvironmentName python -m pip install --editable $ProjectRoot --no-deps
conda run --name $EnvironmentName python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
