$ErrorActionPreference = "Stop"
$EnvironmentNames = @(
    "avimaint-core",
    "avimaint-normalization",
    "avimaint-ie-classical",
    "avimaint-ie-neural",
    "avimaint-spert",
    "avimaint-retrieval",
    "avimaint-dashboard",
    "avimaint-dev"
)

foreach ($EnvironmentName in $EnvironmentNames) {
    Write-Host "Verifying $EnvironmentName"
    conda run --name $EnvironmentName python -m avimaint.cli doctor --environment $EnvironmentName
    conda run --name $EnvironmentName python -m pip check
}

