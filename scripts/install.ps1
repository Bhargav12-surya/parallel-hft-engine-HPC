param([string]$Python = "py", [switch]$Science)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required; pass -Python with a suitable launcher.' }
}
$extras = if ($Science) { '.[dev,science]' } else { '.[dev]' }
& .\.venv\Scripts\python.exe -m pip install -e $extras
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe scripts/build.py --test
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
Write-Host 'Ready: .\.venv\Scripts\hftlab.exe'
