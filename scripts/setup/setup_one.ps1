param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("core", "normalization", "ie-classical", "ie-neural", "spert", "retrieval", "dashboard", "dev")]
    [string]$EnvironmentKey
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvironmentFile = Join-Path $ProjectRoot "envs\$EnvironmentKey\environment.yml"
$RequirementsFile = Join-Path $ProjectRoot "envs\$EnvironmentKey\requirements.txt"
$EnvironmentName = (Select-String -Path $EnvironmentFile -Pattern "^name:\s*(.+)$").Matches.Groups[1].Value.Trim()

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda is not available. Open Anaconda Prompt or initialize Conda for PowerShell."
}

$KnownEnvironments = (conda env list --json | ConvertFrom-Json).envs
$Exists = $KnownEnvironments | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName }

if ($Exists) {
    Write-Host "Updating $EnvironmentName"
    conda env update --name $EnvironmentName --file $EnvironmentFile --prune
}
else {
    Write-Host "Creating $EnvironmentName"
    conda env create --file $EnvironmentFile
}

conda run --name $EnvironmentName python -m pip install --upgrade pip
conda run --name $EnvironmentName python -m pip install --requirement $RequirementsFile
conda run --name $EnvironmentName python -m pip install --editable $ProjectRoot --no-deps
conda run --name $EnvironmentName python -m avimaint.cli doctor --environment $EnvironmentName

