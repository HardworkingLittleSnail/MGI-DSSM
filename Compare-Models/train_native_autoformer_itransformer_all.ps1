param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [string]$OutputRoot = "outputs\patchformer_native_10seeds_200ep"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Seeds = @("7", "17", "27", "37", "47", "57", "67", "77", "87", "97")
$Datasets = @("nasa", "calce", "tju")
$Models = @("autoformer", "itransformer")

foreach ($Model in $Models) {
    Write-Output "[$(Get-Date -Format s)] START $Model"
    & python "Compare-Models\run_autoformer_itransformer.py" `
        --model $Model `
        --datasets $Datasets `
        --seeds $Seeds `
        --device $Device `
        --max-epochs 200 `
        --data-version version3 `
        --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "$Model failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format s)] DONE $Model"
}

$ResolvedOutput = Join-Path $ProjectRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null
"Completed Autoformer and iTransformer on NASA/CALCE/TJU at $(Get-Date -Format o)" |
    Set-Content -Encoding UTF8 (Join-Path $ResolvedOutput "TRAINING_COMPLETE.txt")
Write-Output "[$(Get-Date -Format s)] ALL TRAINING COMPLETE"
