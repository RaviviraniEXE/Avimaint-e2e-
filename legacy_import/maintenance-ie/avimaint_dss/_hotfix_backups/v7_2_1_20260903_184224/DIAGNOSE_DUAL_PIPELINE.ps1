param(
    [string]$RawSpERT = "http://127.0.0.1:8765",
    [string]$Normalizer = "http://127.0.0.1:8766",
    [string]$SemanticSpERT = "http://127.0.0.1:8767"
)
$ErrorActionPreference = "Stop"

$lockPath = Join-Path $PSScriptRoot "runtime_model_lock.json"
$lock = $null
if (Test-Path -LiteralPath $lockPath) {
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
}

Write-Host "=== RAW RQ4 SPERT ==="
Invoke-RestMethod "$RawSpERT/health" | ConvertTo-Json -Depth 10

Write-Host ""
Write-Host "=== BYT5 NORMALIZER ==="
if ($lock -and -not $lock.byt5.enabled) {
    Write-Host ("DISABLED BY MODEL LOCK: " + $lock.byt5.reason)
} else {
    try { Invoke-RestMethod "$Normalizer/health" | ConvertTo-Json -Depth 10 }
    catch { Write-Host "UNAVAILABLE" }
}

Write-Host ""
Write-Host "=== NORMALIZED SEMANTIC SPERT ==="
if ($lock -and (-not $lock.normalized_spert.enabled -or -not $lock.normalized_spert.verified_representation)) {
    Write-Host ("DISABLED BY MODEL LOCK: " + $lock.normalized_spert.reason)
} else {
    try { Invoke-RestMethod "$SemanticSpERT/health" | ConvertTo-Json -Depth 10 }
    catch { Write-Host "UNAVAILABLE" }
}

$queries = @(
    "L/H MAG EXC RPM DROP DURING RUN UP",
    "#2 intake gasket leaking",
    "R/H ENG #4 CYL LOW COMPRESSION 20/80 PSI",
    "#3 rocker cover leaking and #4 intake gasket leaking"
)

foreach ($q in $queries) {
    Write-Host ""
    Write-Host ("=" * 96)
    Write-Host "USER: $q"

    $normText = $null
    if (-not $lock -or $lock.byt5.enabled) {
        try {
            $b = @{text=$q} | ConvertTo-Json -Compress
            $n = Invoke-RestMethod -Method Post -Uri "$Normalizer/normalize" -ContentType "application/json" -Body $b
            Write-Host "BYT5 MODEL INPUT: $($n.model_input)"
            Write-Host "NORMALIZED CANDIDATE: $($n.normalized)"
            $normText = [string]$n.normalized
        } catch {
            Write-Host "NORMALIZATION UNAVAILABLE"
        }
    }

    $rawInput = $q.ToUpperInvariant()
    $bRaw = @{text=$rawInput} | ConvertTo-Json -Compress
    $raw = Invoke-RestMethod -Method Post -Uri "$RawSpERT/predict" -ContentType "application/json" -Body $bRaw
    Write-Host ""
    Write-Host "RAW RQ4 ENTITIES:"
    $raw.entities | Format-Table type,text,score -AutoSize
    Write-Host "RAW RQ4 RELATIONS:"
    $raw.relations | Format-Table type,head,tail,score -AutoSize

    if ($normText -and (-not $lock -or ($lock.normalized_spert.enabled -and $lock.normalized_spert.verified_representation))) {
        try {
            $bSem = @{text=$normText} | ConvertTo-Json -Compress
            $sem = Invoke-RestMethod -Method Post -Uri "$SemanticSpERT/predict" -ContentType "application/json" -Body $bSem
            Write-Host ""
            Write-Host "NORMALIZED SEMANTIC ENTITIES:"
            $sem.entities | Format-Table type,text,score -AutoSize
            Write-Host "NORMALIZED SEMANTIC RELATIONS:"
            $sem.relations | Format-Table type,head,tail,score -AutoSize
        } catch {
            Write-Host "NORMALIZED SEMANTIC SPERT UNAVAILABLE"
        }
    }
}
