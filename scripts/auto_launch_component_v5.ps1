param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 1440
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).Path
$ocrStatus = Join-Path $workspaceRoot "work\train-ocr-v4-rare-prefix\torch_run_status.json"
$componentOutput = Join-Path $workspaceRoot "work\train-component-v5-fullmix"
$componentConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_component_v5_fullmix.json"
$python = Join-Path $workspaceRoot ".venv-train\Scripts\python.exe"
$launcherLog = Join-Path $workspaceRoot "work\component-v5-launcher.log"
$stdoutLog = Join-Path $workspaceRoot "work\component-v5.stdout.log"
$stderrLog = Join-Path $workspaceRoot "work\component-v5.stderr.log"
$pidFile = Join-Path $workspaceRoot "work\component-v5.pid.json"

# The upstream OCR-v4 run was preflighted and started before an additive config
# schema update. Its terminal status must retain this exact run hash; accepting
# the current resume-compatible hash here would let an unrelated rerun release
# the GPU queue.
$expectedOcrTrainingConfigSha256 = "8581f309ee1f342b9359ce0511ea5fb2e0fe74c45d22349a3b5fb04552318fa5"
$expectedComponentTrainingConfigSha256 = "093b0b500a1fc6f8c6efcabb6e4d49d88b4dcceaad00b1604e1ab1047f7f30d2"
$expectedResumeSchema = "pcbparse.torch_training_resume.v2"
$expectedSelectionPolicy = "all_gate_normalized_violation_v1"
$expectedPrimaryMetric = "bbox_map50"
$expectedPrimaryOperator = "gte"
$expectedGateThreshold = 0.25
$expectedSingleGateComparison = "legacy_primary_metric_direction_with_absolute_epsilon"
$env:PYTHONPATH = Join-Path $workspaceRoot "src"

function Write-LauncherLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

# The watcher has no interactive console in normal use.  Record unexpected
# terminating errors so a failed queue hand-off cannot look like a clean exit.
trap {
    try {
        Write-LauncherLog (
            "unhandled launcher error: {0}: {1}" -f `
                $_.Exception.GetType().Name, $_.Exception.Message
        )
    } catch {
        # Preserve the original error if the log itself is unavailable.
    }
    exit 99
}

function Test-ComponentAllGatePolicy($Lineage) {
    try {
        $selection = $Lineage.model_selection
        $gates = @($selection.gates)
        $ordering = @($selection.ordering)
        return (
            $selection.name -eq $expectedSelectionPolicy -and
            $selection.primary_metric -eq $expectedPrimaryMetric -and
            $selection.primary_operator -eq $expectedPrimaryOperator -and
            $selection.comparison.single_gate -eq $expectedSingleGateComparison -and
            [double]$selection.comparison.single_gate_absolute_epsilon -eq 1e-12 -and
            $ordering.Count -eq 4 -and
            $ordering[0] -eq "failed_gate_count_ascending" -and
            $ordering[1] -eq "maximum_normalized_violation_ascending" -and
            $ordering[2] -eq "total_normalized_violation_ascending" -and
            $ordering[3] -eq "primary_metric_best" -and
            $gates.Count -eq 1 -and
            $gates[0].metric -eq $expectedPrimaryMetric -and
            $gates[0].operator -eq $expectedPrimaryOperator -and
            [double]$gates[0].threshold -eq $expectedGateThreshold
        )
    } catch {
        return $false
    }
}

function Test-ComponentStatusLineage($Status, [bool]$RequireTopLevelFormalEvidence) {
    try {
        $valid = (
            $Status.task -eq "component" -and
            $Status.smoke -eq $false -and
            $Status.lineage.task -eq "component" -and
            $Status.lineage.smoke -eq $false -and
            $Status.lineage.formal_evidence -eq $true -and
            $Status.lineage.training_config_sha256 -eq $expectedComponentTrainingConfigSha256 -and
            $Status.lineage.resume.checkpoint_schema -eq $expectedResumeSchema -and
            (Test-ComponentAllGatePolicy $Status.lineage)
        )
        if ($RequireTopLevelFormalEvidence) {
            $valid = $valid -and $Status.formal_evidence -eq $true
        }
        return $valid
    } catch {
        return $false
    }
}

function Get-RecordedProcessState {
    # Four fail-closed states are intentional:
    # missing   - no prior PID metadata; a fresh launch may be considered
    # active    - the exact recorded Component process is still alive
    # inactive  - metadata is valid, but that exact process has exited
    # ambiguous - malformed/stale-incompatible metadata; human review required
    if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
        return "missing"
    }
    try {
        $pidState = Get-Content -LiteralPath $pidFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $recordedPid = [int]$pidState.pid
        $recordedOutput = [IO.Path]::GetFullPath([string]$pidState.output)
        if (
            $recordedPid -le 0 -or
            [string]$pidState.task -ne "component" -or
            [string]$pidState.training_config_sha256 -ne $expectedComponentTrainingConfigSha256 -or
            [string]$pidState.resume_schema -ne $expectedResumeSchema -or
            [string]$pidState.model_selection_policy -ne $expectedSelectionPolicy -or
            -not $recordedOutput.Equals(
                $componentOutput,
                [StringComparison]::OrdinalIgnoreCase
            )
        ) {
            return "ambiguous"
        }
        $recordedStart = [DateTimeOffset]::Parse([string]$pidState.started_at_utc).UtcDateTime
        $candidate = Get-Process -Id $recordedPid -ErrorAction SilentlyContinue
        if ($null -eq $candidate) {
            return "inactive"
        }
        if ($candidate.ProcessName -ne "python") {
            # The PID was reused by another executable, so the originally
            # recorded Component process is no longer active.
            return "inactive"
        }
        $actualStart = $candidate.StartTime.ToUniversalTime()
        if ([Math]::Abs(($actualStart - $recordedStart).TotalSeconds) -lt 5) {
            return "active"
        }
        return "inactive"
    } catch {
        return "ambiguous"
    }
}

foreach ($requiredPath in @($python, $componentConfig)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "required file is missing: $requiredPath"
    }
}
if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5"
}
if ($TimeoutMinutes -lt 1) {
    throw "TimeoutMinutes must be positive"
}

# Hold one workspace-scoped mutex for the full watcher lifetime. Status/PID
# checks protect an already-started trainer, while this closes the race where
# two watchers reach Start-Process before either has written PID/status data.
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $workspaceDigest = [BitConverter]::ToString(
        $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspaceRoot))
    ).Replace("-", "").Substring(0, 32)
} finally {
    $sha256.Dispose()
}
$mutexName = "Local\pcbparse-component-v5-$workspaceDigest"
$launchMutex = [Threading.Mutex]::new($false, $mutexName)
try {
    $mutexAcquired = $launchMutex.WaitOne(0)
} catch [Threading.AbandonedMutexException] {
    $mutexAcquired = $true
}
if (-not $mutexAcquired) {
    Write-LauncherLog "another component-v5 watcher is already active; refusing a duplicate watcher"
    $launchMutex.Dispose()
    exit 7
}

Write-LauncherLog "waiting for exact formal ocr-v4-rare-prefix terminal status"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        Write-LauncherLog "timeout waiting for ocr-v4-rare-prefix; component-v5 was not launched"
        exit 2
    }
    $finished = $false
    if (Test-Path -LiteralPath $ocrStatus -PathType Leaf) {
        try {
            $status = Get-Content -LiteralPath $ocrStatus -Raw -Encoding UTF8 | ConvertFrom-Json
            $finished = (
                [bool]$status.finished_at_utc -and
                $status.status -in @("candidate_evidence_ready", "gate_failed") -and
                $status.task -eq "ocr" -and
                $status.smoke -eq $false -and
                $status.formal_evidence -eq $true -and
                $status.lineage.task -eq "ocr" -and
                $status.lineage.smoke -eq $false -and
                $status.lineage.formal_evidence -eq $true -and
                $status.lineage.training_config_sha256 -eq $expectedOcrTrainingConfigSha256
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

$componentStatus = Join-Path $componentOutput "torch_run_status.json"
$existingStatus = $null
if (Test-Path -LiteralPath $componentStatus -PathType Leaf) {
    try {
        $existingStatus = Get-Content -LiteralPath $componentStatus -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-LauncherLog "component-v5 status is unreadable; refusing an ambiguous launch"
        exit 5
    }
    $hasFormalTerminalStatus = (
        [bool]$existingStatus.finished_at_utc -and
        $existingStatus.status -in @("candidate_evidence_ready", "gate_failed")
    )
    if ($hasFormalTerminalStatus) {
        if (Test-ComponentStatusLineage $existingStatus $true) {
            Write-LauncherLog "component-v5 already has a compatible formal terminal status; no launch needed"
            exit 0
        }
        Write-LauncherLog "component-v5 terminal status has incompatible task/config/v2-policy lineage; refusing launch"
        exit 5
    }
}

$recordedProcessState = Get-RecordedProcessState
if ($recordedProcessState -eq "active") {
    Write-LauncherLog "component-v5 already has an active recorded process; refusing a duplicate launch"
    exit 4
}
if ($recordedProcessState -eq "ambiguous") {
    Write-LauncherLog "component-v5 PID metadata is ambiguous; refusing a duplicate launch"
    exit 5
}
if ($null -ne $existingStatus -and $existingStatus.status -eq "running") {
    if ($recordedProcessState -ne "inactive") {
        Write-LauncherLog "component-v5 reports running without a provably inactive recorded process; refusing launch"
        exit 5
    }
    if (-not (Test-ComponentStatusLineage $existingStatus $false)) {
        Write-LauncherLog "component-v5 running status lacks exact task/config/v2 all-gate lineage; refusing launch"
        exit 5
    }
}
if (
    $null -ne $existingStatus -and
    $existingStatus.status -eq "failed" -and
    [string]$existingStatus.error.message -like "*predates the all-gate model-selection policy*"
) {
    Write-LauncherLog "component-v5 output contains a rejected v1 resume checkpoint; refusing retry"
    exit 6
}

$resume = Join-Path $componentOutput "resume_checkpoint.pt"
if (Test-Path -LiteralPath $componentOutput) {
    $entries = @(Get-ChildItem -LiteralPath $componentOutput -Force)
    if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $resume -PathType Leaf)) {
        Write-LauncherLog "component-v5 output is non-empty without a resume checkpoint; refusing overwrite"
        exit 3
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
    if ($null -eq $existingStatus -or -not (Test-ComponentStatusLineage $existingStatus $false)) {
        Write-LauncherLog "component-v5 resume metadata is not the exact task/config/v2 all-gate policy; refusing resume"
        exit 6
    }
    $arguments += @("--resume-checkpoint", $resume)
    Write-LauncherLog "component-v5 will resume from a verified v2 all-gate epoch checkpoint"
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $workspaceRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
$pidState = [ordered]@{
    pid = $process.Id
    started_at_utc = $process.StartTime.ToUniversalTime().ToString("o")
    task = "component"
    output = $componentOutput
    training_config_sha256 = $expectedComponentTrainingConfigSha256
    resume_schema = $expectedResumeSchema
    model_selection_policy = $expectedSelectionPolicy
}
$pidTemporary = "$pidFile.$PID.tmp"
try {
    $pidState | ConvertTo-Json | Set-Content -LiteralPath $pidTemporary -Encoding UTF8
    Move-Item -LiteralPath $pidTemporary -Destination $pidFile -Force
} finally {
    Remove-Item -LiteralPath $pidTemporary -Force -ErrorAction SilentlyContinue
}
Write-LauncherLog "component-v5 launched (pid $($process.Id))"
$launchMutex.ReleaseMutex()
$launchMutex.Dispose()
