param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 360
)

$ErrorActionPreference = "Stop"
$componentStatus = Join-Path $Workspace "work\train-component-v3\torch_run_status.json"
$python = Join-Path $Workspace ".venv-train\Scripts\python.exe"
$log = Join-Path $Workspace "work\pin-v2-launcher.log"
$env:PYTHONPATH = Join-Path $Workspace "src"
$env:PCB_OCR_MATERIALIZED_ROOT = Join-Path $Workspace "work\materialized\ocr-v2-512"

function Write-Log($message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Output $line
}

Write-Log "waiting for component v3 status: $componentStatus"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while (-not (Test-Path $componentStatus)) {
    if ((Get-Date) -gt $deadline) {
        Write-Log "timeout waiting for component training; aborting"
        exit 2
    }
    Start-Sleep -Seconds $PollSeconds
}
$finished = $false
while (-not $finished) {
    if ((Get-Date) -gt $deadline) {
        Write-Log "timeout waiting for component training completion; aborting"
        exit 2
    }
    try {
        $status = Get-Content $componentStatus -Raw -Encoding UTF8 | ConvertFrom-Json
        $finished = [bool]$status.finished_at_utc -or $status.status -eq "candidate_evidence_ready"
    } catch {
        $finished = $false
    }
    if (-not $finished) {
        Start-Sleep -Seconds $PollSeconds
    }
}
Write-Log "component v3 finished; launching pin v2 training"

$pinOutput = Join-Path $Workspace "work\train-pin-v2"
$pinOutLog = Join-Path $Workspace "work\pin-v2-watchdog-stdout.log"
$pinErrLog = Join-Path $Workspace "work\pin-v2-watchdog-stderr.log"

$arguments = @(
    "-m", "pcbparse.training.torch_train_cli",
    "--config", (Join-Path $Workspace "configs\train\dataset300_v2_pin_v2.json"),
    "--task", "pin",
    "--output", $pinOutput,
    "--workspace-root", $Workspace,
    "--device", "cuda"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -RedirectStandardOutput $pinOutLog -RedirectStandardError $pinErrLog -WindowStyle Hidden -PassThru
Write-Log "pin v2 launched (pid $($process.Id)); stdout -> $pinOutLog, stderr -> $pinErrLog"
