$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Bench = Join-Path $Root "legacy_import\maintie-bench"
$SpERT = Join-Path $Root "external\spert"
$Provenance = Join-Path $SpERT "UPSTREAM_PROVENANCE.json"
if (-not (Test-Path (Join-Path $SpERT ".git")) -or -not (Test-Path $Provenance)) { throw "Official SpERT is not installed. Run scripts\setup\clone_official_spert.bat first." }
& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if ($LASTEXITCODE -ne 0) { throw "MaintIE SpERT CUDA preflight failed." }
& (Join-Path $Root "scripts\setup\patch_spert_windows_timestamp.ps1")
if ($LASTEXITCODE -ne 0) { throw "SpERT Windows timestamp patch failed." }
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\setup\patch_spert_safetensors_compat.py")
if ($LASTEXITCODE -ne 0) { throw "SpERT safetensors compatibility patch failed." }
$Export = Join-Path $Bench "outputs\spert"
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") train --config (Join-Path $Export "avimaint_spert.conf")
if ($LASTEXITCODE -ne 0) { throw "MaintIE SpERT training failed." }
$Model = Get-ChildItem (Join-Path $Export "save") -Recurse -Directory -Filter final_model | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Model) { throw "MaintIE SpERT final_model was not created." }
$Test = Join-Path $Export "test.json"
$Pred = Join-Path $Export "predictions_test.json"
if (Test-Path $Pred) { Remove-Item -Force $Pred }
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $SpERT "spert.py") predict `
  --dataset_path $Test --types_path (Join-Path $Export "avimaint_types.json") `
  --model_path $Model.FullName --tokenizer_path $Model.FullName `
  --predictions_path $Pred --model_type spert `
  --eval_batch_size 1 --max_span_size 10 --seed 42
if ($LASTEXITCODE -ne 0) { throw "MaintIE SpERT prediction command failed." }
& conda run --no-capture-output -n avimaint-spert python -u (Join-Path $Root "scripts\ie\verify_spert_predictions.py") --dataset $Test --predictions $Pred
if ($LASTEXITCODE -ne 0) { throw "MaintIE SpERT prediction artifact validation failed." }
Write-Host "MaintIE frozen-test predictions verified: $Pred"
