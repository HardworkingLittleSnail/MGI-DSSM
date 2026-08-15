$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Seeds = @("7", "17", "27", "37", "47", "57", "67", "77", "87", "97")
$OutputRoot = Join-Path $ProjectRoot "outputs\comparison_models_three_datasets_10seeds_200ep"
$Datasets = @("nasa", "calce", "tju")

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$TranscriptPath = Join-Path $OutputRoot "training_transcript.log"
Start-Transcript -Path $TranscriptPath -Append | Out-Null

foreach ($Dataset in $Datasets) {
    Write-Output "[$(Get-Date -Format s)] START physics-dual-loss/$Dataset"
    & python "Compare-Models\run_physics_dual_loss.py" `
        --dataset $Dataset `
        --seeds $Seeds `
        --device cpu `
        --max-epochs 200 `
        --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "physics-dual-loss/$Dataset failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format s)] DONE physics-dual-loss/$Dataset"
}

foreach ($Dataset in $Datasets) {
    Write-Output "[$(Get-Date -Format s)] START sg-dits/$Dataset"
    & python "Compare-Models\run_sg_dits.py" `
        --dataset $Dataset `
        --seeds $Seeds `
        --device cpu `
        --max-epochs 200 `
        --samples 5 `
        --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "sg-dits/$Dataset failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format s)] DONE sg-dits/$Dataset"
}

$Marker = Join-Path $OutputRoot "TRAINING_COMPLETE.txt"
"Completed all 60 runs at $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $Marker
Write-Output "[$(Get-Date -Format s)] ALL TRAINING COMPLETE"
Stop-Transcript | Out-Null
