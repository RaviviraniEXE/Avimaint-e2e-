param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IE = Join-Path $Root "legacy_import\maintenance-ie"
$Base = Join-Path $IE "outputs\spert_normalized"
$TrainDriver = Join-Path $Root "scripts\ie\06_train_and_test_spert.ps1"
$PredictDriver = Join-Path $Root "scripts\ie\06_predict_existing_spert.ps1"
$VerifyPred = Join-Path $Root "scripts\ie\verify_spert_predictions.py"
$LogDir = Join-Path $Base "_run_logs"
$Systems = @("rules", "byt5", "selective_byt5", "rules_then_byt5")

if (-not (Test-Path (Join-Path $Base "PREP_MANIFEST.json"))) {
    throw "Matched SpERT exports are not prepared. Run prepare_matched_normalization_spert.py first."
}
if (-not (Test-Path $TrainDriver)) { throw "Missing established SpERT training driver: $TrainDriver" }
if (-not (Test-Path $PredictDriver)) { throw "Missing established no-retrain SpERT predictor: $PredictDriver" }
if (-not (Select-String -Path $TrainDriver -Pattern "ExportName" -Quiet)) {
    throw "06_train_and_test_spert.ps1 does not expose -ExportName; nested normalized model directories cannot be selected safely."
}
if (-not (Select-String -Path $PredictDriver -Pattern "ExportName" -Quiet)) {
    throw "06_predict_existing_spert.ps1 does not expose -ExportName."
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "========================================================================"
Write-Host "  REPRESENTATION-MATCHED NORMALIZATION -> SpERT TRAINING"
Write-Host "  Four NEW normalized models; raw model is NOT retrained"
Write-Host "  Same frozen 1275/100/225 membership and same FULL 9x11 config"
Write-Host "  Comparative TEST metrics are withheld until ALL four are complete"
Write-Host "========================================================================"

& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('torch=',torch.__version__); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed. Training was NOT started." }

$modelMap = @{}

foreach ($System in $Systems) {
    $Export = Join-Path $Base $System
    $ExportName = "spert_normalized\$System"
    $Config = Join-Path $Export "avimaint_spert.conf"
    $Test = Join-Path $Export "test.json"
    $Pred = Join-Path $Export "predictions_test.json"
    $Save = Join-Path $Export "save"
    $LogPath = Join-Path $LogDir ($System + ".log")

    if (-not (Test-Path $Config)) { throw "Missing prepared config for $System : $Config" }
    if (-not (Test-Path $Test)) { throw "Missing prepared frozen test for $System : $Test" }

    $ExistingModel = $null
    if (Test-Path $Save) {
        $ExistingModel = Get-ChildItem $Save -Recurse -Directory -Filter final_model |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }

    if (-not $ExistingModel) {
        Write-Host ""
        Write-Host "------------------------------------------------------------------------"
        Write-Host ("  TRAIN + PREDICT: " + $System)
        Write-Host "------------------------------------------------------------------------"
        Write-Host ("  export=" + $Export)
        Write-Host "  TEST metrics will NOT be calculated here."

        $PreviousPreference = $ErrorActionPreference
        $Code = 999
        try {
            $ErrorActionPreference = "Continue"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $TrainDriver -ExportName $ExportName 2>&1 |
                Tee-Object -FilePath $LogPath |
                ForEach-Object { Write-Host $_ }
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousPreference
        }
        if ($Code -ne 0) { throw "SpERT training/prediction failed for $System with exit code $Code. See $LogPath" }
    }
    else {
        Write-Host ""
        Write-Host "------------------------------------------------------------------------"
        Write-Host ("  RESUME: " + $System)
        Write-Host "------------------------------------------------------------------------"
        Write-Host ("  Existing final_model found: " + $ExistingModel.FullName)
        if (-not (Test-Path $Pred)) {
            Write-Host "  Prediction artifact missing; recovering prediction WITHOUT retraining."
            $PreviousPreference = $ErrorActionPreference
            $Code = 999
            try {
                $ErrorActionPreference = "Continue"
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $PredictDriver -ExportName $ExportName 2>&1 |
                    Tee-Object -FilePath $LogPath -Append |
                    ForEach-Object { Write-Host $_ }
                $Code = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $PreviousPreference
            }
            if ($Code -ne 0) { throw "Prediction recovery failed for $System with exit code $Code. See $LogPath" }
        }
        else {
            Write-Host "  predictions_test.json already exists; model will NOT be retrained."
        }
    }

    & conda run --no-capture-output -n avimaint-spert python -u $VerifyPred --dataset $Test --predictions $Pred
    if ($LASTEXITCODE -ne 0) { throw "$System prediction verification failed. Do not continue to comparative evaluation." }

    $FinalModel = Get-ChildItem $Save -Recurse -Directory -Filter final_model |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $FinalModel) { throw "$System has no final_model after run." }
    $modelMap[$System] = $FinalModel.FullName
    Write-Host ("  MODEL SAVED AS " + $System + " -> " + $FinalModel.FullName)
}

$modelPathJson = Join-Path $Base "MODEL_PATHS.json"
$modelMap | ConvertTo-Json | Set-Content -Encoding UTF8 $modelPathJson

Write-Host ""
Write-Host "========================================================================"
Write-Host "  ALL FOUR NORMALIZED SpERT MODELS + TEST PREDICTIONS ARE COMPLETE"
Write-Host "  No comparative TEST metric has been used to retune any condition."
Write-Host ("  Model-path registry -> " + $modelPathJson)
Write-Host "========================================================================"
