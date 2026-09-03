param([string]$ExportName = "spert_core")
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IE = Join-Path $Root "legacy_import\maintenance-ie"
$SpERT = Join-Path $Root "external\spert"
$Export = Join-Path $IE ("outputs\" + $ExportName)
$Test = Join-Path $Export "test.json"
$Types = Join-Path $Export "avimaint_types.json"
$Pred = Join-Path $Export "predictions_test.json"

Write-Host ""
Write-Host "======================================================================"
Write-Host "  RECOVER EXISTING SpERT MODEL -> FROZEN TEST PREDICTION"
Write-Host "  No training will be repeated"
Write-Host "======================================================================"

& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('torch=',torch.__version__); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if ($LASTEXITCODE -ne 0) { throw "SpERT CUDA preflight failed." }

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\setup\patch_spert_safetensors_compat.py")
if ($LASTEXITCODE -ne 0) { throw "SpERT safetensors compatibility patch failed." }

$Model = Get-ChildItem (Join-Path $Export "save") -Recurse -Directory -Filter final_model | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Model) { throw "No existing SpERT final_model was found under $Export\save. Do NOT retrain until this is investigated." }
$Safe = Join-Path $Model.FullName "model.safetensors"
$Bin = Join-Path $Model.FullName "pytorch_model.bin"
if (-not (Test-Path $Safe) -and -not (Test-Path $Bin)) { throw "Existing final_model has no model.safetensors or pytorch_model.bin: $($Model.FullName)" }

Write-Host ""
Write-Host "[MODEL] Reusing completed training artifact"
Write-Host "  model=$($Model.FullName)"
if (Test-Path $Safe) {
    $WeightsName = "model.safetensors"
} else {
    $WeightsName = "pytorch_model.bin"
}
Write-Host "  weights=$WeightsName"
Write-Host "[TEST] Frozen TEST prediction only"
if (Test-Path $Pred) { Remove-Item -Force $Pred }

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") predict `
  --dataset_path $Test `
  --types_path $Types `
  --model_path $Model.FullName --tokenizer_path $Model.FullName `
  --predictions_path $Pred `
  --model_type spert --eval_batch_size 1 --max_span_size 10 --seed 42
if ($LASTEXITCODE -ne 0) { throw "SpERT prediction command failed with exit code $LASTEXITCODE." }

& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\ie\verify_spert_predictions.py") --dataset $Test --predictions $Pred
if ($LASTEXITCODE -ne 0) { throw "SpERT prediction artifact validation failed. The run is NOT complete." }

Write-Host ""
Write-Host "RECOVERY COMPLETE - TRAINING WAS NOT REPEATED"
Write-Host "Frozen-test predictions: $Pred"
