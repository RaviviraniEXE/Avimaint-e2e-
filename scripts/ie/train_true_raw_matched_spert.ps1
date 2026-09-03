param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IE = Join-Path $Root "legacy_import\maintenance-ie"
$Base = Join-Path $IE "outputs\spert_normalized"
$Export = Join-Path $Base "raw"
$ExportName = "spert_normalized\raw"
$TrainDriver = Join-Path $Root "scripts\ie\06_train_and_test_spert.ps1"
$PredictDriver = Join-Path $Root "scripts\ie\06_predict_existing_spert.ps1"
$VerifyPred = Join-Path $Root "scripts\ie\verify_spert_predictions.py"
$LogDir = Join-Path $Base "_run_logs"
$LogPath = Join-Path $LogDir "raw_v2_correction.log"

if (-not (Test-Path (Join-Path $Base "PREP_MANIFEST_V2.json"))) {
    throw "Corrected raw export is not prepared."
}
if (-not (Test-Path $TrainDriver)) { throw "Missing SpERT training driver: $TrainDriver" }
if (-not (Test-Path $PredictDriver)) { throw "Missing SpERT prediction driver: $PredictDriver" }
if (-not (Select-String -Path $TrainDriver -Pattern "ExportName" -Quiet)) {
    throw "Training driver does not expose -ExportName."
}
if (-not (Select-String -Path $PredictDriver -Pattern "ExportName" -Quiet)) {
    throw "Prediction driver does not expose -ExportName."
}

$Config = Join-Path $Export "avimaint_spert.conf"
$Test = Join-Path $Export "test.json"
$Pred = Join-Path $Export "predictions_test.json"
$Save = Join-Path $Export "save"
if (-not (Test-Path $Config)) { throw "Missing raw config: $Config" }
if (-not (Test-Path $Test)) { throw "Missing raw frozen TEST: $Test" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "========================================================================"
Write-Host "  CORRECTED MATCHED NORMALIZATION -> SpERT"
Write-Host "  TRAIN ONLY THE MISSING TRUE SYSTEM-A RAW CONDITION"
Write-Host "  Existing four normalized models are NOT retrained"
Write-Host "========================================================================"

& conda run --no-capture-output -n avimaint-spert python -u -c "import sys,torch; print('torch=',torch.__version__); print('CUDA available=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); sys.exit(20) if not torch.cuda.is_available() else None"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed. Training was NOT started." }

$ExistingModel = $null
if (Test-Path $Save) {
    $ExistingModel = Get-ChildItem $Save -Recurse -Directory -Filter final_model |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

if (-not $ExistingModel) {
    Write-Host ""
    Write-Host "------------------------------------------------------------------------"
    Write-Host "  TRAIN + PREDICT: true raw"
    Write-Host "------------------------------------------------------------------------"
    Write-Host ("  export=" + $Export)
    Write-Host "  Fixed architecture/hyperparameters; no retuning."

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
    if ($Code -ne 0) {
        throw "True-raw SpERT training/prediction failed with exit code $Code. See $LogPath"
    }
}
else {
    Write-Host ("Existing corrected raw final_model found: " + $ExistingModel.FullName)
    Write-Host "Raw model will NOT be retrained."
    if (-not (Test-Path $Pred)) {
        Write-Host "Prediction artifact missing; recovering prediction WITHOUT retraining."
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
        if ($Code -ne 0) { throw "Raw prediction recovery failed with exit code $Code." }
    }
}

& conda run --no-capture-output -n avimaint-spert python -u $VerifyPred --dataset $Test --predictions $Pred
if ($LASTEXITCODE -ne 0) { throw "Corrected raw prediction verification failed." }

$FinalModel = Get-ChildItem $Save -Recurse -Directory -Filter final_model |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $FinalModel) { throw "Corrected raw final_model not found after run." }

$Registry = @{
    raw = $FinalModel.FullName
}
$Registry | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Base "RAW_MODEL_PATH_V2.json")

Write-Host ""
Write-Host "========================================================================"
Write-Host "  TRUE RAW MODEL COMPLETE"
Write-Host ("  MODEL SAVED AS raw -> " + $FinalModel.FullName)
Write-Host "  Existing four normalized models were not retrained."
Write-Host "========================================================================"
