param()

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IeRoot = Join-Path $Root "legacy_import\maintenance-ie"
$SpertDir = Join-Path $IeRoot "outputs\spert"
$Prepared = Join-Path $IeRoot "outputs\normalization_spert_ablation\prepared"
$PredDir = Join-Path $IeRoot "outputs\normalization_spert_ablation\predictions"
$LogDir = Join-Path $IeRoot "outputs\normalization_spert_ablation\logs"
$Predictor = Join-Path $Root "scripts\ie\06_predict_existing_spert.ps1"

$Systems = @("raw", "rules", "byt5", "selective_byt5", "rules_then_byt5")
$ExportName = "spert"   # IMPORTANT: full 9-entity / 11-relation aviation SpERT export

$LiveTest = Join-Path $SpertDir "test.json"
$LivePred = Join-Path $SpertDir "predictions_test.json"
$BackupDir = Join-Path $IeRoot "outputs\normalization_spert_ablation\_runtime_backup"

if (-not (Test-Path $Predictor)) {
    throw "Missing existing no-retrain SpERT predictor: $Predictor"
}
if (-not (Test-Path $LiveTest)) {
    throw "Missing authoritative full-schema SpERT test.json: $LiveTest"
}
if (-not (Test-Path (Join-Path $SpertDir "save"))) {
    throw "Missing full-schema SpERT save directory: $(Join-Path $SpertDir 'save')"
}

# Safety check: this hotfix relies on the existing predictor supporting
# -ExportName so we can explicitly select outputs\spert (FULL schema), rather
# than any default such as spert_core.
if (-not (Select-String -Path $Predictor -Pattern "ExportName" -Quiet)) {
    throw "Existing predictor does not expose -ExportName. Do not run this ablation until scripts\ie\06_predict_existing_spert.ps1 is inspected."
}

$FullModel = Get-ChildItem (Join-Path $SpertDir "save") -Recurse -Directory -Filter final_model |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $FullModel) {
    throw "No existing FULL-schema SpERT final_model found under $SpertDir\save. No training will be started."
}

New-Item -ItemType Directory -Force -Path $PredDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

$BackupTest = Join-Path $BackupDir "test.json.original"
$BackupPred = Join-Path $BackupDir "predictions_test.json.original"
$HadPrediction = Test-Path $LivePred

Copy-Item -Force $LiveTest $BackupTest
if ($HadPrediction) {
    Copy-Item -Force $LivePred $BackupPred
}

Write-Host ""
Write-Host "========================================================================"
Write-Host "  FIVE-WAY NORMALIZATION -> SAME FROZEN FULL SpERT"
Write-Host "  NO TRAINING. NO RETUNING. Existing prediction script only."
Write-Host "========================================================================"
Write-Host ("[FULL MODEL] " + $FullModel.FullName)
Write-Host ("[EXPORT]     " + $ExportName)
Write-Host ""

try {
    foreach ($System in $Systems) {
        $VariantTest = Join-Path $Prepared ($System + "_test.json")
        if (-not (Test-Path $VariantTest)) {
            throw "Missing prepared test variant: $VariantTest"
        }

        Write-Host ""
        Write-Host "------------------------------------------------------------------------"
        Write-Host ("  PREDICT: " + $System)
        Write-Host "------------------------------------------------------------------------"

        Copy-Item -Force $VariantTest $LiveTest
        if (Test-Path $LivePred) {
            Remove-Item -Force $LivePred
        }

        $LogPath = Join-Path $LogDir ($System + ".log")

        # Windows PowerShell 5.1 wraps text written by a native child process to
        # STDERR as NativeCommandError when the parent redirects 2>&1. Libraries
        # such as transformers emit harmless FutureWarning messages to STDERR.
        # With ErrorActionPreference=Stop this used to abort the ablation even
        # though SpERT itself had not failed. Temporarily use Continue ONLY for
        # the child process stream, then rely on the real process exit code.
        $PreviousPreference = $ErrorActionPreference
        $Code = 999
        try {
            $ErrorActionPreference = "Continue"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Predictor -ExportName $ExportName 2>&1 |
                Tee-Object -FilePath $LogPath |
                ForEach-Object { Write-Host $_ }
            $Code = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousPreference
        }

        if ($Code -ne 0) {
            throw "Existing SpERT predictor failed for $System with exit code $Code. See $LogPath"
        }
        if (-not (Test-Path $LivePred)) {
            throw "Predictor returned exit code 0 but no predictions_test.json exists for $System"
        }

        $Destination = Join-Path $PredDir ($System + "_predictions.json")
        Copy-Item -Force $LivePred $Destination
        Write-Host ("  saved -> " + $Destination)
    }
}
finally {
    Write-Host ""
    Write-Host "[RESTORE] Restoring authoritative SpERT test/prediction artifacts..."
    if (Test-Path $BackupTest) {
        Copy-Item -Force $BackupTest $LiveTest
    }
    if ($HadPrediction -and (Test-Path $BackupPred)) {
        Copy-Item -Force $BackupPred $LivePred
    }
    elseif (Test-Path $LivePred) {
        Remove-Item -Force $LivePred
    }
    Write-Host "[RESTORE] done"
}

Write-Host ""
Write-Host "All five frozen FULL-model prediction artifacts created."
