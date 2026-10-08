param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string[]]$WorkerCommand
)
$ErrorActionPreference = 'Stop'
$Repository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repository
& .\.venv\Scripts\python.exe tools\competition_supervisor.py --run-id $RunId --minimum-free-gib 20 -- @WorkerCommand
exit $LASTEXITCODE

