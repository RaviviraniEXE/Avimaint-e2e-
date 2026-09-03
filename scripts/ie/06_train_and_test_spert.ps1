param([string]$ExportName = "spert")
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IE = Join-Path $Root "legacy_import\maintenance-ie"
$SpERT = Join-Path $Root "external\spert"
$Provenance = Join-Path $SpERT "UPSTREAM_PROVENANCE.json"

Write-Host ""
Write-Host "======================================================================"
Write-Host "  AviMaint SpERT GPU preflight"
Write-Host "  Environment: avimaint-spert"
Write-Host "  CUDA is mandatory; CPU fallback is disabled"
Write-Host "======================================================================"
& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('torch=',torch.__version__); print('CUDA build=',torch.version.cuda); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None; x=torch.randn((1024,1024),device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA compute test=PASS')"
if ($LASTEXITCODE -ne 0) { throw "SpERT CUDA preflight failed. Run scripts\setup\fix_spert_cuda.bat. Training was NOT started." }
if (-not (Test-Path (Join-Path $SpERT ".git")) -or -not (Test-Path $Provenance)) { throw "Official SpERT is not installed. Run scripts\setup\clone_official_spert.bat first." }

Write-Host ""
Write-Host "[COMPAT] Verifying/applying Windows-safe SpERT run-directory timestamp..."
& (Join-Path $Root "scripts\setup\patch_spert_windows_timestamp.ps1")
if ($LASTEXITCODE -ne 0) { throw "SpERT Windows compatibility patch failed. Training was NOT started." }
Write-Host "[COMPAT] Verifying/applying safetensors checkpoint compatibility..."
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\setup\patch_spert_safetensors_compat.py")
if ($LASTEXITCODE -ne 0) { throw "SpERT safetensors compatibility patch failed. Training was NOT started." }

$Export = Join-Path $IE ("outputs\" + $ExportName)
$Config = Join-Path $Export "avimaint_spert.conf"
$Test = Join-Path $Export "test.json"
$Pred = Join-Path $Export "predictions_test.json"
if (-not (Test-Path $Config)) { throw "SpERT export/config is missing: $Config. Run the corresponding 05/05b export script first." }

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Write-Host ""
Write-Host "[TRAIN] Official SpERT on CUDA"
Write-Host "  export=$Export"
Write-Host "  config=$Config"
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") train --config $Config
if ($LASTEXITCODE -ne 0) { throw "SpERT training failed with exit code $LASTEXITCODE." }
$Model = Get-ChildItem (Join-Path $Export "save") -Recurse -Directory -Filter final_model | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Model) { throw "SpERT final_model was not created." }

Write-Host ""
Write-Host "[TEST] Frozen TEST prediction"
Write-Host "  model=$($Model.FullName)"
if (Test-Path $Pred) { Remove-Item -Force $Pred }
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") predict `
  --dataset_path $Test `
  --types_path (Join-Path $Export "avimaint_types.json") `
  --model_path $Model.FullName --tokenizer_path $Model.FullName `
  --predictions_path $Pred `
  --model_type spert --eval_batch_size 1 --max_span_size 10 --seed 42
if ($LASTEXITCODE -ne 0) { throw "SpERT frozen-test prediction failed with exit code $LASTEXITCODE." }
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\ie\verify_spert_predictions.py") --dataset $Test --predictions $Pred
if ($LASTEXITCODE -ne 0) { throw "SpERT prediction artifact validation failed. The run is NOT complete." }
Write-Host ""
Write-Host "Frozen-test predictions: $Pred"
Write-Host "SpERT CUDA run complete."
