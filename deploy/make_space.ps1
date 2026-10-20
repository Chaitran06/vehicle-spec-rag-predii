# Windows version of make_space.sh.
#   powershell -ExecutionPolicy Bypass -File deploy\make_space.ps1 ..\f150-spec-finder-space
# Run `specrag ingest; specrag export-index` first so index\ exists.
param([Parameter(Mandatory = $true)][string]$Dest)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not (Test-Path (Join-Path $Root "index\vectors.npz"))) {
    throw "index\vectors.npz missing - run: specrag ingest; specrag export-index"
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item (Join-Path $Root "deploy\huggingface\Dockerfile") (Join-Path $Dest "Dockerfile") -Force
Copy-Item (Join-Path $Root "deploy\huggingface\README.md") (Join-Path $Dest "README.md") -Force
Copy-Item (Join-Path $Root "requirements.txt") $Dest -Force
foreach ($d in "specrag", "api", "frontend", "index") {
    $target = Join-Path $Dest $d
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item (Join-Path $Root $d) $target -Recurse
}
Get-ChildItem $Dest -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Write-Host "Space folder ready at $Dest"
