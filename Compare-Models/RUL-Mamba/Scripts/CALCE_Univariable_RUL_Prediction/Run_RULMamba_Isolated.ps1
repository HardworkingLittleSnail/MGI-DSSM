param(
    [string]$PythonExe = 'D:\ChenLi\Anaconda\anaconda\envs\etsformer\python.exe',
    [string]$BaseConfig = '',
    [string]$ModelConfig = '',
    [int[]]$Seeds = @(7, 17, 27, 37, 47, 57, 67, 77, 87, 97),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$TrainScript = Join-Path $ProjectRoot 'Scripts\CALCE_Univariable_RUL_Prediction\Train_CALCE_Univariable.py'
if (-not $BaseConfig) {
    $BaseConfig = Join-Path $ProjectRoot 'Configs\CALCE\Univariable\Base_RULMamba_Native.yaml'
}
if (-not $ModelConfig) {
    $ModelConfig = Join-Path $ProjectRoot 'Configs\CALCE\Univariable\RULMamba_Native.yaml'
}
$ModelConfigData = Get-Content -LiteralPath $ModelConfig -Raw
$LogsMatch = [regex]::Match($ModelConfigData, '(?m)^\s*logs_dir:\s*(.+?)\s*$')
if ($LogsMatch.Success) {
    $ConfiguredLogsDir = $LogsMatch.Groups[1].Value.Trim()
} else {
    $ConfiguredLogsDir = 'Logs/CALCE/Univariable/RULMamba_Native_Protocol'
}
$ConsoleLogDir = Join-Path $ProjectRoot ($ConfiguredLogsDir -replace '/', '\')
$ConsoleLogDir = Join-Path $ConsoleLogDir 'Console'
New-Item -ItemType Directory -Force -Path $ConsoleLogDir | Out-Null

$Batteries = @('CS2_35', 'CS2_36', 'CS2_37', 'CS2_38')

Push-Location $ProjectRoot
try {
    foreach ($Battery in $Batteries) {
        foreach ($Seed in $Seeds) {
            $ConsoleLog = Join-Path $ConsoleLogDir ("{0}_seed_{1}.console.log" -f $Battery, $Seed)
            $Arguments = @(
                $TrainScript,
                '--config', $BaseConfig,
                '--model-config', $ModelConfig,
                '--batteries', $Battery,
                '--seeds', $Seed,
                '--defer-summary'
            )
            if ($Force) {
                $Arguments += '--force'
            }

            # Windows PowerShell converts every native stderr line (including
            # harmless Python warnings) into an ErrorRecord. Temporarily use
            # Continue so stderr is logged without terminating the runner;
            # the native exit code below remains the actual failure signal.
            $PreviousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $ConsoleLog
            $RunExitCode = $LASTEXITCODE
            $ErrorActionPreference = $PreviousErrorAction
            if ($RunExitCode -ne 0) {
                throw "CALCE run failed: battery=$Battery seed=$Seed; see $ConsoleLog"
            }
        }
    }

    # All 40 artifacts now exist. A final no-training pass reads them and
    # rebuilds all_results.csv/json and Summary.json across batteries/seeds.
    & $PythonExe $TrainScript `
        '--config' $BaseConfig `
        '--model-config' $ModelConfig
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to rebuild the aggregate CALCE summary.'
    }
}
finally {
    Pop-Location
}
