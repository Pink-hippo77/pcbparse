param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 1440,
    [switch]$LaunchEnvironmentSmoke
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).Path
$componentStatus = Join-Path $workspaceRoot "work\train-component-v5-fullmix\torch_run_status.json"
$ocrOutput = Join-Path $workspaceRoot "work\train-ocr-v5-allgate"
$ocrConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_ocr_v5_allgate.json"
$python = Join-Path $workspaceRoot ".venv-train\Scripts\python.exe"
$pythonWorkingDirectory = Join-Path $workspaceRoot "src"
$launcherLog = Join-Path $workspaceRoot "work\ocr-v5-launcher.log"
$stdoutLog = Join-Path $workspaceRoot "work\ocr-v5.stdout.log"
$stderrLog = Join-Path $workspaceRoot "work\ocr-v5.stderr.log"
$pidFile = Join-Path $workspaceRoot "work\ocr-v5.pid.json"
$expectedComponentTrainingConfigSha256 = "093b0b500a1fc6f8c6efcabb6e4d49d88b4dcceaad00b1604e1ab1047f7f30d2"
$expectedOcrTrainingConfigSha256 = "80508a46009e160eef0f6c228aeccac152f148205ab7e917aaf2a0025c5ea229"
$expectedResumeSchema = "pcbparse.torch_training_resume.v2"
$expectedSelectionPolicy = "all_gate_normalized_violation_v1"
$expectedSingleGateComparison = "legacy_primary_metric_direction_with_absolute_epsilon"

function Write-LauncherLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

function New-DeduplicatedLaunchEnvironment {
    # PowerShell 7 materializes Start-Process environments with a
    # case-insensitive comparer on Windows.  Some hosts expose both Path and
    # PATH, so retain exactly one canonical key.  Module discovery comes from
    # the source working directory and never from a caller-controlled
    # PYTHONPATH (which can also be corrupted when a non-ASCII path is
    # marshaled by older hosts).
    $launchEnvironment = [hashtable]::new([StringComparer]::OrdinalIgnoreCase)
    $pathValue = $null
    foreach ($entry in [Environment]::GetEnvironmentVariables("Process").GetEnumerator()) {
        $name = [string]$entry.Key
        if ($name.Equals("PYTHONPATH", [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if ($name.Equals("Path", [StringComparison]::OrdinalIgnoreCase)) {
            if ($null -eq $pathValue -or $name -ceq "Path") {
                $pathValue = [string]$entry.Value
            }
            continue
        }
        if (-not $launchEnvironment.ContainsKey($name)) {
            $launchEnvironment[$name] = [string]$entry.Value
        }
    }
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable(
            "Path", [EnvironmentVariableTarget]::Machine
        )
    }
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        throw "cannot construct a canonical Path for the OCR-v5 child environment"
    }
    $launchEnvironment["Path"] = $pathValue
    if ($launchEnvironment.ContainsKey("PYTHONPATH")) {
        throw "OCR-v5 launch environment unexpectedly contains PYTHONPATH"
    }
    return $launchEnvironment
}

function Start-BoundPythonProcess(
    [object[]]$Arguments,
    [string]$StandardOutputPath,
    [string]$StandardErrorPath
) {
    $launchEnvironment = New-DeduplicatedLaunchEnvironment
    return Start-Process `
        -FilePath $python `
        -ArgumentList $Arguments `
        -WorkingDirectory $pythonWorkingDirectory `
        -RedirectStandardOutput $StandardOutputPath `
        -RedirectStandardError $StandardErrorPath `
        -WindowStyle Hidden `
        -UseNewEnvironment `
        -Environment $launchEnvironment `
        -PassThru
}

function Invoke-LaunchEnvironmentSmoke {
    # argparse --help proves that the exact child environment imports the
    # repository package from a Unicode workspace without creating training
    # output, checkpoints, or PID metadata.
    $token = [Guid]::NewGuid().ToString("N")
    $temporaryRoot = [IO.Path]::GetTempPath()
    $smokeStdout = Join-Path $temporaryRoot "pcbparse-ocr-v5-$token.stdout.log"
    $smokeStderr = Join-Path $temporaryRoot "pcbparse-ocr-v5-$token.stderr.log"
    try {
        $smokeArguments = @("-m", "pcbparse.training.torch_train_cli", "--help")
        $smokeProcess = Start-BoundPythonProcess `
            $smokeArguments $smokeStdout $smokeStderr
        $smokeProcess.WaitForExit()
        $smokeOutput = Get-Content -LiteralPath $smokeStdout -Raw -Encoding UTF8
        $smokeError = Get-Content -LiteralPath $smokeStderr -Raw -Encoding UTF8
        if ($smokeProcess.ExitCode -ne 0) {
            throw "launch-environment smoke child failed with exit $($smokeProcess.ExitCode): $smokeError"
        }
        if ($smokeOutput -notlike "*--config*" -or
            $smokeOutput -notlike "*--workspace-root*" -or
            $smokeOutput -notlike "*--task*") {
            throw "launch-environment smoke did not import the expected pcbparse CLI"
        }
        Write-Output "OCR_V5_LAUNCH_ENVIRONMENT_SMOKE_OK"
    } finally {
        Remove-Item -LiteralPath $smokeStdout -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $smokeStderr -Force -ErrorAction SilentlyContinue
    }
}

# Watchers normally have no console.  Preserve the exact terminal error instead
# of disappearing silently if an unforeseen PowerShell exception occurs.
trap {
    try {
        Write-LauncherLog (
            "unhandled launcher error: {0}: {1}" -f `
                $_.Exception.GetType().Name, $_.Exception.Message
        )
    } catch {
        # The original failure remains authoritative if even logging fails.
    }
    exit 99
}

function Test-OcrAllGatePolicy($Lineage) {
    try {
        $selection = $Lineage.model_selection
        $gates = @($selection.gates)
        $ordering = @($selection.ordering)
        return (
            $selection.name -eq $expectedSelectionPolicy -and
            $selection.primary_metric -eq "normalized_exact_match" -and
            $selection.primary_operator -eq "gte" -and
            $selection.comparison.multi_gate -eq "exact_lexicographic_key" -and
            $selection.comparison.single_gate -eq $expectedSingleGateComparison -and
            [double]$selection.comparison.single_gate_absolute_epsilon -eq 1e-12 -and
            $selection.violation_normalization -eq "raw_violation/max(abs(threshold),1e-12)" -and
            $ordering.Count -eq 4 -and
            $ordering[0] -eq "failed_gate_count_ascending" -and
            $ordering[1] -eq "maximum_normalized_violation_ascending" -and
            $ordering[2] -eq "total_normalized_violation_ascending" -and
            $ordering[3] -eq "primary_metric_best" -and
            $gates.Count -eq 4 -and
            $gates[0].metric -eq "normalized_exact_match" -and
            $gates[0].operator -eq "gte" -and
            [double]$gates[0].threshold -eq 0.65 -and
            $gates[1].metric -eq "detector_hmean" -and
            $gates[1].operator -eq "gte" -and
            [double]$gates[1].threshold -eq 0.45 -and
            $gates[2].metric -eq "recognition_designator_exact_match" -and
            $gates[2].operator -eq "gte" -and
            [double]$gates[2].threshold -eq 0.90 -and
            $gates[3].metric -eq "recognition_pinnumber_exact_match" -and
            $gates[3].operator -eq "gte" -and
            [double]$gates[3].threshold -eq 0.74
        )
    } catch {
        return $false
    }
}

function Test-OcrStatusLineage($Status, [bool]$RequireTopLevelFormalEvidence) {
    try {
        $valid = (
            $Status.task -eq "ocr" -and
            $Status.smoke -eq $false -and
            $Status.lineage.task -eq "ocr" -and
            $Status.lineage.smoke -eq $false -and
            $Status.lineage.formal_evidence -eq $true -and
            $Status.lineage.training_config_sha256 -eq $expectedOcrTrainingConfigSha256 -and
            $Status.lineage.resume.checkpoint_schema -eq $expectedResumeSchema -and
            (Test-OcrAllGatePolicy $Status.lineage)
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
    if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) {
        return "missing"
    }
    try {
        $pidState = Get-Content -LiteralPath $pidFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $recordedPid = [int]$pidState.pid
        $recordedOutput = [IO.Path]::GetFullPath([string]$pidState.output)
        if (
            $recordedPid -le 0 -or
            [string]$pidState.task -ne "ocr" -or
            [string]$pidState.training_config_sha256 -ne $expectedOcrTrainingConfigSha256 -or
            [string]$pidState.resume_schema -ne $expectedResumeSchema -or
            [string]$pidState.model_selection_policy -ne $expectedSelectionPolicy -or
            -not $recordedOutput.Equals(
                $ocrOutput,
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
            # The recorded PID was reused by another executable, so the
            # originally recorded OCR process is no longer active.
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

foreach ($requiredPath in @($python, $ocrConfig)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "required file is missing: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath $pythonWorkingDirectory -PathType Container)) {
    throw "required source directory is missing: $pythonWorkingDirectory"
}
if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5"
}
if ($TimeoutMinutes -lt 1) {
    throw "TimeoutMinutes must be positive"
}
if ($LaunchEnvironmentSmoke) {
    Invoke-LaunchEnvironmentSmoke
    exit 0
}

# Hold one workspace-scoped mutex for the full watcher lifetime.  Status/PID
# checks protect an already-started trainer, while this closes the smaller race
# where two watcher processes reach Start-Process before either has written PID
# or running-status metadata.
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $workspaceDigest = [BitConverter]::ToString(
        $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspaceRoot))
    ).Replace("-", "").Substring(0, 32)
} finally {
    $sha256.Dispose()
}
$mutexName = "Local\pcbparse-ocr-v5-$workspaceDigest"
$launchMutex = [Threading.Mutex]::new($false, $mutexName)
try {
    $mutexAcquired = $launchMutex.WaitOne(0)
} catch [Threading.AbandonedMutexException] {
    $mutexAcquired = $true
}
if (-not $mutexAcquired) {
    Write-LauncherLog "another ocr-v5 watcher is already active; refusing a duplicate watcher"
    $launchMutex.Dispose()
    exit 7
}

Write-LauncherLog "waiting for component-v5-fullmix formal training completion"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ($true) {
    if ((Get-Date) -gt $deadline) {
        Write-LauncherLog "timeout waiting for component-v5-fullmix; ocr-v5 was not launched"
        exit 2
    }
    $finished = $false
    if (Test-Path -LiteralPath $componentStatus -PathType Leaf) {
        try {
            $status = Get-Content -LiteralPath $componentStatus -Raw -Encoding UTF8 | ConvertFrom-Json
            $finished = (
                [bool]$status.finished_at_utc -and
                $status.status -in @("candidate_evidence_ready", "gate_failed") -and
                $status.task -eq "component" -and
                $status.smoke -eq $false -and
                $status.formal_evidence -eq $true -and
                $status.lineage.task -eq "component" -and
                $status.lineage.smoke -eq $false -and
                $status.lineage.formal_evidence -eq $true -and
                $status.lineage.training_config_sha256 -eq $expectedComponentTrainingConfigSha256 -and
                $status.lineage.resume.checkpoint_schema -eq $expectedResumeSchema -and
                $status.lineage.model_selection.name -eq $expectedSelectionPolicy
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

$ocrStatus = Join-Path $ocrOutput "torch_run_status.json"
$existingStatus = $null
if (Test-Path -LiteralPath $ocrStatus -PathType Leaf) {
    try {
        $existingStatus = Get-Content -LiteralPath $ocrStatus -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-LauncherLog "ocr-v5 status is unreadable; refusing an ambiguous launch"
        exit 5
    }
    $hasFormalTerminalStatus = (
        [bool]$existingStatus.finished_at_utc -and
        $existingStatus.status -in @("candidate_evidence_ready", "gate_failed")
    )
    if ($hasFormalTerminalStatus) {
        if (Test-OcrStatusLineage $existingStatus $true) {
            Write-LauncherLog "ocr-v5 already has a compatible formal terminal status; no launch needed"
            exit 0
        }
        Write-LauncherLog "ocr-v5 terminal status has incompatible task/config/v2-policy lineage; refusing launch"
        exit 5
    }
}

$recordedProcessState = Get-RecordedProcessState
if ($recordedProcessState -eq "active") {
    Write-LauncherLog "ocr-v5 already has an active recorded process; refusing a duplicate launch"
    exit 4
}
if ($recordedProcessState -eq "ambiguous") {
    Write-LauncherLog "ocr-v5 PID metadata is ambiguous; refusing a duplicate launch"
    exit 5
}
if ($null -ne $existingStatus -and $existingStatus.status -eq "running") {
    if ($recordedProcessState -ne "inactive") {
        Write-LauncherLog "ocr-v5 reports running without a provably inactive recorded process; refusing launch"
        exit 5
    }
    if (-not (Test-OcrStatusLineage $existingStatus $false)) {
        Write-LauncherLog "ocr-v5 running status lacks exact task/config/v2 all-gate lineage; refusing launch"
        exit 5
    }
}
if (
    $null -ne $existingStatus -and
    $existingStatus.status -eq "failed" -and
    [string]$existingStatus.error.message -like "*predates the all-gate model-selection policy*"
) {
    Write-LauncherLog "ocr-v5 output contains a rejected v1 resume checkpoint; refusing retry"
    exit 6
}

$resume = Join-Path $ocrOutput "resume_checkpoint.pt"
if (Test-Path -LiteralPath $ocrOutput) {
    $entries = @(Get-ChildItem -LiteralPath $ocrOutput -Force)
    if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $resume -PathType Leaf)) {
        Write-LauncherLog "ocr-v5 output is non-empty without a resume checkpoint; refusing overwrite"
        exit 3
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
    if ($null -eq $existingStatus -or -not (Test-OcrStatusLineage $existingStatus $false)) {
        Write-LauncherLog "ocr-v5 resume metadata is not the required v2 all-gate policy; refusing resume"
        exit 6
    }
    $arguments += @("--resume-checkpoint", $resume)
    Write-LauncherLog "ocr-v5 will resume from a verified v2 all-gate epoch checkpoint"
}

$process = Start-BoundPythonProcess $arguments $stdoutLog $stderrLog
$pidState = [ordered]@{
    pid = $process.Id
    started_at_utc = $process.StartTime.ToUniversalTime().ToString("o")
    task = "ocr"
    output = $ocrOutput
    training_config_sha256 = $expectedOcrTrainingConfigSha256
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
Write-LauncherLog "ocr-v5 launched (pid $($process.Id))"
$launchMutex.ReleaseMutex()
$launchMutex.Dispose()
