param(
    [Parameter(Mandatory=$true)][string]$StatePath,
    [int]$RefreshSeconds = 5
)

$ErrorActionPreference = 'Stop'
while ($true) {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $overall = $state.overall_percent
    $stage = $state.stage_percent
    $overallNumeric = $overall -is [double] -or $overall -is [int] -or $overall -is [decimal]
    $stageNumeric = $stage -is [double] -or $stage -is [int] -or $stage -is [decimal]
    if ($overallNumeric) {
        Write-Progress -Id 1 -Activity 'OVERALL PIPELINE' -Status "$overall% · $($state.stage)" -PercentComplete ([double]$overall)
    } else {
        Write-Progress -Id 1 -Activity 'OVERALL PIPELINE' -Status "INDETERMINATE · $($state.stage)" -PercentComplete 0
    }
    $counter = if ($null -ne $state.done -and $null -ne $state.total) { "$($state.done)/$($state.total)" } else { 'counter unavailable' }
    if ($stageNumeric) {
        Write-Progress -Id 2 -ParentId 1 -Activity 'CURRENT STAGE' -Status "$($state.current_operation) · $counter · $stage%" -PercentComplete ([double]$stage)
    } else {
        Write-Progress -Id 2 -ParentId 1 -Activity 'CURRENT STAGE' -Status "$($state.current_operation) · $counter · INDETERMINATE" -PercentComplete 0
    }
    if ($state.status -ne 'RUNNING') {
        Write-Progress -Id 2 -ParentId 1 -Activity 'CURRENT STAGE' -Completed
        Write-Progress -Id 1 -Activity 'OVERALL PIPELINE' -Completed
        $state | ConvertTo-Json -Depth 10
        break
    }
    Start-Sleep -Seconds $RefreshSeconds
}
