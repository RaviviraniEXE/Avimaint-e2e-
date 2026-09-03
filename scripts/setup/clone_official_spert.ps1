param(
    [string]$Ref = "master",
    [switch]$Refresh
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Target = Join-Path $Root "external\spert"
$Remote = "https://github.com/lavis-nlp/spert.git"
$Readme = Join-Path $Target "README.md"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows, reopen Anaconda Prompt, and run this script again."
}

if (Test-Path (Join-Path $Target ".git")) {
    $ConfiguredRemote = (& git -C $Target remote get-url origin).Trim()
    if ($ConfiguredRemote -ne $Remote) {
        throw "external\spert exists but origin is '$ConfiguredRemote', not the official '$Remote'."
    }
    $Dirty = & git -C $Target status --porcelain
    if ($Dirty) {
        throw "external\spert has local modifications. Preserve or discard them explicitly before refreshing."
    }
    if ($Refresh) {
        & git -C $Target fetch --prune origin
        if ($LASTEXITCODE -ne 0) { throw "Could not fetch official SpERT." }
    }
} else {
    if (Test-Path $Target) {
        Get-ChildItem $Target -Force | Where-Object { $_.Name -ne "README.md" } | ForEach-Object {
            throw "external\spert must be empty before cloning; unexpected item: $($_.FullName)"
        }
        if (Test-Path $Readme) { Remove-Item $Readme }
    }
    & git clone --origin origin --branch master --single-branch $Remote $Target
    if ($LASTEXITCODE -ne 0) { throw "Official SpERT clone failed." }
}

if ($Ref -eq "master") {
    & git -C $Target checkout master
    if ($Refresh) { & git -C $Target merge --ff-only origin/master }
} else {
    & git -C $Target checkout --detach $Ref
}
if ($LASTEXITCODE -ne 0) { throw "Could not check out SpERT ref '$Ref'." }

$Commit = (& git -C $Target rev-parse HEAD).Trim()
$Branch = (& git -C $Target rev-parse --abbrev-ref HEAD).Trim()
$Provenance = [ordered]@{
    upstream = $Remote
    requested_ref = $Ref
    resolved_commit = $Commit
    branch = $Branch
    repository_archived_utc = "2025-04-02"
    cloned_or_verified_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_modified_by_avimaint = $false
}
$Provenance | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Target "UPSTREAM_PROVENANCE.json")
Write-Host "Official SpERT ready at commit $Commit"
Write-Host "Provenance: $Target\UPSTREAM_PROVENANCE.json"
