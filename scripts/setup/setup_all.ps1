$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SetupOne = Join-Path $ProjectRoot "scripts\setup\setup_one.ps1"
$CloneSpERT = Join-Path $ProjectRoot "scripts\setup\clone_official_spert.ps1"
$EnvironmentKeys = @(
    "core",
    "normalization",
    "ie-classical",
    "ie-neural",
    "spert",
    "retrieval",
    "dashboard",
    "dev"
)

foreach ($EnvironmentKey in $EnvironmentKeys) {
    & $SetupOne $EnvironmentKey
}

& $CloneSpERT

Write-Host "All AviMaint environments are installed."
