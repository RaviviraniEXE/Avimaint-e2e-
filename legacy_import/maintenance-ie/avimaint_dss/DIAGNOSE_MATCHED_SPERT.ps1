param(
    [string]$Url = "http://127.0.0.1:8765"
)

$ErrorActionPreference = "Stop"

Write-Host "=== HEALTH ==="
Invoke-RestMethod "$Url/health" | ConvertTo-Json -Depth 10

$queries = @(
    "#2 intake gasket leaking",
    "#3 rocker cover leaking and #4 intake gasket leaking",
    "#4 cylinder low compression",
    "left magneto excessive rpm drop during run up"
)

foreach ($q in $queries) {
    Write-Host ""
    Write-Host "=== QUERY ==="
    Write-Host $q
    $body = @{text=$q} | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$Url/predict" -ContentType "application/json" -Body $body |
        ConvertTo-Json -Depth 20
}
