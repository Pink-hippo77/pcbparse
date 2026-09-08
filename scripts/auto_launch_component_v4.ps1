param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 720
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).Path
$pinStatus = Join-Path $workspaceRoot "work\train-pin-v5-boundary\torch_run_status.json"
$componentOutput = Join-Path $workspaceRoot "work\train-component-v4"
$componentConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_component_v4.json"
$python = Join-Path $workspaceRoot ".venv-train\Scripts\python.exe"
$launcherLog = Join-Path $workspaceRoot "work\component-v4-launcher.log"
$stdoutLog = Join-Path $workspaceRoot "work\component-v4.stdout.log"
$stderrLog = Join-Path $workspaceRoot "work\component-v4.stderr.log"
$env:PYTHONPATH = Join-Path $workspaceRoot "src"

function Write-LauncherLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

foreach ($requiredPath in @($python, $componentConfig)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "required file is missing: $requiredPath"
    }
}

Write-LauncherLog "waiting for pin-v5-boundary formal training completion"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        Write-LauncherLog "timeout waiting for pin-v5-boundary; component-v4 was not launched"
        exit 2
    }
    $finished = $false
    if (Test-Path -LiteralPath $pinStatus -PathType Leaf) {
        try {
            $status = Get-Content -LiteralPath $pinStatus -Raw -Encoding UTF8 | ConvertFrom-Json
            # A setup/runtime failure also has finished_at_utc, but is not a
            # completed formal training run.  Only the two terminal outcomes
            # reached after the configured epoch/early-stop policy release the
            # next GPU job.
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

$resume = Join-Path $componentOutput "resume_checkpoint.pt"
if (Test-Path -LiteralPath $componentOutput) {
    if (-not (Test-Path -LiteralPath $resume -PathType Leaf)) {
        $entries = @(Get-ChildItem -LiteralPath $componentOutput -Force)
        if ($entries.Count -gt 0) {
            Write-LauncherLog "component-v4 output is non-empty without a resume checkpoint; refusing overwrite"
            exit 3
        }
    }
}

$arguments = @(
    "-m", "pcbparse.training.torch_train_cli",
    "--config", $componentConfig,
    "--task", "component",
    "--output", $componentOutput,
    "--workspace-root", $workspaceRoot,
    "--device", "cuda"
)
if (Test-Path -LiteralPath $resume -PathType Leaf) {
    $arguments += @("--resume-checkpoint", $resume)
    Write-LauncherLog "component-v4 will resume from its epoch-boundary checkpoint"
}
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $workspaceRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Write-LauncherLog "component-v4 launched (pid $($process.Id))"
