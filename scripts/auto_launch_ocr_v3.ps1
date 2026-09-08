param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 1440
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).Path
$componentStatus = Join-Path $workspaceRoot "work\train-component-v4\torch_run_status.json"
$ocrOutput = Join-Path $workspaceRoot "work\train-ocr-v3-shrink-warmstart"
$ocrConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_ocr_v3_shrink_warmstart.json"
$python = Join-Path $workspaceRoot ".venv-train\Scripts\python.exe"
$launcherLog = Join-Path $workspaceRoot "work\ocr-v3-launcher.log"
$stdoutLog = Join-Path $workspaceRoot "work\ocr-v3.stdout.log"
$stderrLog = Join-Path $workspaceRoot "work\ocr-v3.stderr.log"
$env:PYTHONPATH = Join-Path $workspaceRoot "src"

function Write-LauncherLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

foreach ($requiredPath in @($python, $ocrConfig)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "required file is missing: $requiredPath"
    }
}

Write-LauncherLog "waiting for component-v4 formal training completion"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        Write-LauncherLog "timeout waiting for component-v4; ocr-v3 was not launched"
        exit 2
    }
    $finished = $false
    if (Test-Path -LiteralPath $componentStatus -PathType Leaf) {
        try {
            $status = Get-Content -LiteralPath $componentStatus -Raw -Encoding UTF8 | ConvertFrom-Json
            $finished = (
                [bool]$status.finished_at_utc -and
                $status.status -in @("candidate_evidence_ready", "gate_failed")
            )
        } catch {
            $finished = $false
        }
    }
    if ($finished) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

$resume = Join-Path $ocrOutput "resume_checkpoint.pt"
if (Test-Path -LiteralPath $ocrOutput) {
    if (-not (Test-Path -LiteralPath $resume -PathType Leaf)) {
        $entries = @(Get-ChildItem -LiteralPath $ocrOutput -Force)
        if ($entries.Count -gt 0) {
            Write-LauncherLog "ocr-v3 output is non-empty without a resume checkpoint; refusing overwrite"
            exit 3
        }
    }
}

$arguments = @(
    "-m", "pcbparse.training.torch_train_cli",
    "--config", $ocrConfig,
    "--task", "ocr",
    "--output", $ocrOutput,
    "--workspace-root", $workspaceRoot,
    "--device", "cuda"
)
if (Test-Path -LiteralPath $resume -PathType Leaf) {
    $arguments += @("--resume-checkpoint", $resume)
    Write-LauncherLog "ocr-v3 will resume from its epoch-boundary checkpoint"
}
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $workspaceRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Write-LauncherLog "ocr-v3 launched (pid $($process.Id))"
