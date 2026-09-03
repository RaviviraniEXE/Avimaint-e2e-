param([switch]$VerifyOnly)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpERT = Join-Path $Root "external\spert"
$Trainer = Join-Path $SpERT "spert\trainer.py"
$PatchDir = Join-Path $SpERT "AVIMAINT_PATCHES"
$PatchManifest = Join-Path $PatchDir "windows_safe_run_key.json"
$Backup = Join-Path $PatchDir "trainer.py.upstream_backup"
$Provenance = Join-Path $SpERT "UPSTREAM_PROVENANCE.json"

if (-not (Test-Path $Trainer)) {
    throw "Official SpERT trainer was not found: $Trainer. Run scripts\setup\clone_official_spert.bat first."
}

$Old = "run_key = str(datetime.datetime.now()).replace(' ', '_')"
$New = "run_key = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # AviMaint Windows-safe compatibility patch"
$Text = [System.IO.File]::ReadAllText($Trainer)

if ($Text.Contains($New)) {
    Write-Host "[SPERT WINDOWS PATCH] already applied"
    Write-Host "  trainer=$Trainer"
    Write-Host "  run_key format=YYYY-MM-DD_HH-MM-SS_microseconds"
    exit 0
}

# Accept a previously applied equivalent safe patch as valid.
if ($Text -match "run_key\s*=\s*datetime\.datetime\.now\(\)\.strftime\([^\r\n]*%H-%M-%S") {
    Write-Host "[SPERT WINDOWS PATCH] equivalent Windows-safe run_key already present"
    Write-Host "  trainer=$Trainer"
    exit 0
}

if (-not $Text.Contains($Old)) {
    throw "Could not find the expected upstream SpERT run_key statement. Refusing to patch an unknown source layout."
}

if ($VerifyOnly) {
    throw "SpERT still uses a Windows-incompatible timestamp containing ':'. Run scripts\setup\patch_spert_windows_timestamp.bat or rerun the SpERT launcher after installing the hotfix."
}

New-Item -ItemType Directory -Force -Path $PatchDir | Out-Null
if (-not (Test-Path $Backup)) {
    Copy-Item -LiteralPath $Trainer -Destination $Backup
}

$BeforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Trainer).Hash.ToLowerInvariant()
$Patched = $Text.Replace($Old, $New)
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($Trainer, $Patched, $Utf8NoBom)
$AfterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Trainer).Hash.ToLowerInvariant()

$Commit = $null
if (Test-Path (Join-Path $SpERT ".git")) {
    try { $Commit = (& git -C $SpERT rev-parse HEAD).Trim() } catch { $Commit = $null }
}

$Manifest = [ordered]@{
    patch = "windows_safe_run_key"
    applied_utc = (Get-Date).ToUniversalTime().ToString("o")
    upstream_commit = $Commit
    file = "spert/trainer.py"
    original_sha256 = $BeforeHash
    patched_sha256 = $AfterHash
    upstream_statement = $Old
    replacement_statement = $New
    reason = "Official SpERT derives save/log directory names from str(datetime.datetime.now()), which contains ':'; ':' is invalid in Windows directory names. The compatibility patch changes only the run-directory timestamp format."
    experiment_semantics_changed = $false
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $PatchManifest

# Record the compatibility patch in the project's existing provenance file.
if (Test-Path $Provenance) {
    try {
        $P = Get-Content -Raw $Provenance | ConvertFrom-Json
        if ($P.PSObject.Properties.Name -contains "source_modified_by_avimaint") {
            $P.source_modified_by_avimaint = $true
        } else {
            $P | Add-Member -NotePropertyName source_modified_by_avimaint -NotePropertyValue $true
        }
        $PatchInfo = [ordered]@{
            id = "windows_safe_run_key"
            file = "spert/trainer.py"
            semantic_change = $false
            manifest = "AVIMAINT_PATCHES/windows_safe_run_key.json"
        }
        if ($P.PSObject.Properties.Name -contains "compatibility_patches") {
            $Existing = @($P.compatibility_patches) | Where-Object { $_.id -ne "windows_safe_run_key" }
            $P.compatibility_patches = @($Existing) + @($PatchInfo)
        } else {
            $P | Add-Member -NotePropertyName compatibility_patches -NotePropertyValue @($PatchInfo)
        }
        $P | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $Provenance
    } catch {
        Write-Warning "Could not update UPSTREAM_PROVENANCE.json, but the compatibility patch itself was applied: $($_.Exception.Message)"
    }
}

Write-Host "[SPERT WINDOWS PATCH] APPLIED"
Write-Host "  trainer=$Trainer"
Write-Host "  backup=$Backup"
Write-Host "  manifest=$PatchManifest"
Write-Host "  run_key format=YYYY-MM-DD_HH-MM-SS_microseconds"
exit 0
