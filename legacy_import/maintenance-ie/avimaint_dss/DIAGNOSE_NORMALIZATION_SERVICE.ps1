param([string]$Url = "http://127.0.0.1:8766")
$ErrorActionPreference = "Stop"

Write-Host "=== EXPERT RULES -> GUARDED BYT5 HEALTH ==="
Invoke-RestMethod "$Url/health" | ConvertTo-Json -Depth 10

$queries = @(
    "L/H MAG EXC RPM DROP DURING RUN UP",
    "#2 intake gasket leaking",
    "R/H ENG #4 CYL LOW COMPRESSION 20/80 PSI",
    "FWD ENG BAFFLE ASSY LOOSE"
)
foreach ($q in $queries) {
    Write-Host ""
    Write-Host "INPUT: $q"
    $body = @{text=$q} | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$Url/normalize" -ContentType "application/json" -Body $body |
        ConvertTo-Json -Depth 10
}
