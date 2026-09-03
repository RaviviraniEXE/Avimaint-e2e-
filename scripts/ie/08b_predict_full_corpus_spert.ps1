$ErrorActionPreference = "Stop"

$System = "selective_byt5"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IE = Join-Path $Root "legacy_import\maintenance-ie"
$SpERT = Join-Path $Root "external\spert"

$RegistryPath = Join-Path $IE "outputs\reports\normalization_spert_matched_v2\MODEL_REGISTRY_V2.json"
$PrepManifest = Join-Path $IE "outputs\kg\full_corpus_manifest.json"

if (-not (Test-Path $RegistryPath)) {
    throw "Corrected V2 model registry missing: $RegistryPath"
}
if (-not (Test-Path $PrepManifest)) {
    throw "Full-corpus preparation manifest missing: $PrepManifest"
}

$Registry = Get-Content -Raw $RegistryPath | ConvertFrom-Json
$Entry = $Registry.$System
if (-not $Entry) {
    throw "MODEL_REGISTRY_V2.json has no '$System' entry."
}

$ModelRel = [string]$Entry.final_model_path
if (-not $ModelRel) {
    throw "Registry '$System' entry has no final_model_path."
}

$Model = Join-Path $Root $ModelRel
if (-not (Test-Path $Model)) {
    throw "Registered Selective-ByT5 final_model missing: $Model"
}

$ExpectedRoot = (
    Resolve-Path (
        Join-Path $IE "outputs\spert_normalized\selective_byt5"
    )
).Path
$ResolvedModel = (Resolve-Path $Model).Path

if (-not $ResolvedModel.StartsWith(
    $ExpectedRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw (
        "Refusing wrong model. Registry model is outside " +
        "outputs\spert_normalized\selective_byt5: $ResolvedModel"
    )
}

$Export = Join-Path $IE "outputs\spert_normalized\selective_byt5"
$Types = Join-Path $Export "avimaint_types.json"
$Dataset = Join-Path $IE "outputs\kg\full_corpus_spert.json"
$Index = Join-Path $IE "outputs\kg\full_index.jsonl"
$Pred = Join-Path $IE "outputs\kg\predictions_full.json"
$ExtractionManifest = Join-Path $IE "outputs\kg\FINAL_FULL_CORPUS_SPERT_MANIFEST.json"

$Prep = Get-Content -Raw $PrepManifest | ConvertFrom-Json
if ($Prep.representation -ne $System) {
    throw (
        "Refusing inference: prepared representation=" +
        "'$($Prep.representation)', expected '$System'."
    )
}
if (
    $Prep.representation_parity.status -ne "pass" -or
    [int]$Prep.representation_parity.projected_equals_trained_export -ne 1600 -or
    [int]$Prep.representation_parity.operational_tokenizer_matches -ne 1600 -or
    [int]$Prep.representation_parity.mismatches -ne 0
) {
    throw "Refusing inference: 1600/1600 trained-representation parity did not pass."
}
if (
    [int]$Prep.prepared_records -ne 6169 -or
    [int]$Prep.unique_identifiers -ne 6169
) {
    throw "Refusing inference: prepared corpus is not 6169 unique records."
}

& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if ($LASTEXITCODE -ne 0) {
    throw "SpERT CUDA preflight failed."
}

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\setup\patch_spert_safetensors_compat.py")
if ($LASTEXITCODE -ne 0) {
    throw "SpERT safetensors compatibility patch failed."
}

Write-Host "========================================================================"
Write-Host "  FINAL OPERATIONAL 6169 SELECTIVE-BYT5 -> MATCHED SpERT"
Write-Host "  NO TRAINING / NO RETUNING"
Write-Host "========================================================================"
Write-Host "[REPRESENTATION] selective_byt5"
Write-Host "[RAW SOURCE]     preserved in full_index.jsonl"
Write-Host "[MODEL]          $ResolvedModel"
Write-Host "[RECORDS]        6169"
Write-Host ""

if (Test-Path $Pred) {
    Remove-Item -Force $Pred
}
if (Test-Path $ExtractionManifest) {
    Remove-Item -Force $ExtractionManifest
}

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") predict `
  --dataset_path $Dataset `
  --types_path $Types `
  --model_path $ResolvedModel `
  --tokenizer_path $ResolvedModel `
  --predictions_path $Pred `
  --model_type spert `
  --eval_batch_size 1 `
  --max_span_size 10 `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "Full-corpus Selective-ByT5 SpERT prediction failed."
}

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\ie\verify_full_corpus_spert.py") `
  --dataset $Dataset `
  --predictions $Pred `
  --index $Index `
  --types $Types `
  --model $ResolvedModel `
  --prep-manifest $PrepManifest `
  --model-registry $RegistryPath `
  --representation selective_byt5 `
  --model-system selective_byt5 `
  --expected-records 6169 `
  --expected-entity-types 9 `
  --expected-relation-types 11 `
  --manifest $ExtractionManifest

if ($LASTEXITCODE -ne 0) {
    throw "Final Selective-ByT5 extraction verification failed."
}

Write-Host ""
Write-Host "FINAL OPERATIONAL PREDICTIONS VERIFIED:"
Write-Host "  $Pred"
Write-Host "MANIFEST:"
Write-Host "  $ExtractionManifest"
