param(
    [switch]$Train,
    [switch]$EvaluateTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectRoot

$DataConfig = "configs/normalization/data.yaml"
$SplitConfig = "configs/normalization/split.yaml"
$ModelConfig = "configs/normalization/byt5_gold.yaml"

conda run -n avimaint-normalization python -m avimaint.normalization audit --config $DataConfig
Write-Host "Review data/aviation/interim/normalization_manual_review.csv before continuing."

if ($Train) {
    conda run -n avimaint-normalization python -m avimaint.normalization prepare --config $DataConfig
    conda run -n avimaint-normalization python -m avimaint.normalization split --config $SplitConfig
    conda run -n avimaint-normalization python -m avimaint.normalization train --config $ModelConfig
    conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split validation --system raw
    conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split validation --system most_frequent_replacement
    conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split validation --system rules
    conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split validation --system byt5
    conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split validation --system selective_byt5
}

if ($EvaluateTest) {
    Write-Host "Test evaluation is final. Confirm the configuration was selected using validation only."
    foreach ($System in @("raw", "most_frequent_replacement", "rules", "byt5", "selective_byt5", "rules_then_byt5")) {
        conda run -n avimaint-normalization python -m avimaint.normalization predict --config $ModelConfig --split test --system $System
        conda run -n avimaint-normalization python -m avimaint.normalization evaluate --config $ModelConfig --split test --system $System
    }
}
