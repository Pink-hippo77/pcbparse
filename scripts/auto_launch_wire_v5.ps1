param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 1440,
    [switch]$LaunchEnvironmentSmoke
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path -LiteralPath $Workspace).Path
$sourceRoot = Join-Path $workspaceRoot "src"
$workRoot = Join-Path $workspaceRoot "work"
$ocrConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_ocr_v5_allgate.json"
$ocrStatus = Join-Path $workspaceRoot "work\train-ocr-v5-allgate\torch_run_status.json"
$wireConfig = Join-Path $workspaceRoot "configs\train\dataset300_v2_wire_v5_fresh_allgate.json"
$wireOutput = Join-Path $workspaceRoot "work\train-wire-v5-fresh-allgate"
$python = Join-Path $workspaceRoot ".venv-train\Scripts\python.exe"
$launcherLog = Join-Path $workRoot "wire-v5-launcher.log"
$stdoutLog = Join-Path $workRoot "wire-v5.stdout.log"
$stderrLog = Join-Path $workRoot "wire-v5.stderr.log"
$pidFile = Join-Path $workRoot "wire-v5.pid.json"

# These are semantic hashes produced by pcbparse.training.config, not merely
# byte hashes.  The byte hashes additionally pin the reviewed files so a queue
# hand-off never starts after an unnoticed edit or path substitution.
$expectedOcrTrainingConfigSha256 = "80508a46009e160eef0f6c228aeccac152f148205ab7e917aaf2a0025c5ea229"
$expectedOcrConfigFileSha256 = "833752c16936f9bb02ccec7d9344907d40719636cb219bbbcf156d024095718b"
$expectedWireTrainingConfigSha256 = "e9b4003e3352a3ab4ea647fc1f3bec281e0cb4463fe3147d3851a28b1639716a"
$expectedWireConfigFileSha256 = "e2a1b78b1f766d3075582d8960c2c03871d01ac16f7a1fe1fef56c66bfc5afee"
$expectedResumeSchema = "pcbparse.torch_training_resume.v2"
$expectedSelectionPolicy = "all_gate_normalized_violation_v1"
$expectedSingleGateComparison = "legacy_primary_metric_direction_with_absolute_epsilon"
$pidSchema = "pcbparse.training.launch_pid.v1"
$launchEnvironmentMode = "new_case_insensitive_deduplicated_no_pythonpath_v1"

if (-not (Test-Path -LiteralPath $workRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $workRoot | Out-Null
}

function Write-LauncherLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

trap {
    try {
        Write-LauncherLog (
            "unhandled launcher error: {0}: {1}" -f `
                $_.Exception.GetType().Name, $_.Exception.Message
        )
    } catch {
        # Preserve the original terminating error if logging also fails.
    }
    exit 99
}

function Get-SemanticConfigSha256([string]$ConfigPath) {
    $program = @"
import sys
from pcbparse.training.config import load_training_config, training_config_sha256
print(training_config_sha256(load_training_config(sys.argv[1])))
"@
    # Import from the repository's src working directory.  Do not let a
    # caller-provided PYTHONPATH decide which pcbparse package is hashed.
    $savedPythonPath = [Environment]::GetEnvironmentVariable(
        "PYTHONPATH", [EnvironmentVariableTarget]::Process
    )
    [Environment]::SetEnvironmentVariable(
        "PYTHONPATH", $null, [EnvironmentVariableTarget]::Process
    )
    Push-Location -LiteralPath $sourceRoot
    try {
        $lines = @(& $python -c $program $ConfigPath)
        $semanticProcessExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
        [Environment]::SetEnvironmentVariable(
            "PYTHONPATH", $savedPythonPath, [EnvironmentVariableTarget]::Process
        )
    }
    if ($semanticProcessExitCode -ne 0 -or $lines.Count -ne 1) {
        throw "cannot compute semantic training-config SHA-256: $ConfigPath"
    }
    $value = ([string]$lines[0]).Trim().ToLowerInvariant()
    if ($value -notmatch "^[0-9a-f]{64}$") {
        throw "semantic training-config SHA-256 is malformed: $ConfigPath"
    }
    return $value
}

function New-DeduplicatedLaunchEnvironment {
    # Start-Process on Windows materializes a case-insensitive environment.
    # Some hosts expose both Path and PATH, which otherwise throws before the
    # child exists.  Rebuild exactly one key per case-folded name, canonicalize
    # Path, and deliberately omit PYTHONPATH: module discovery comes only from
    # $sourceRoot as the child working directory.
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
        throw "cannot construct a canonical Path for the Wire-v5 child environment"
    }
    $launchEnvironment["Path"] = $pathValue
    if ($launchEnvironment.ContainsKey("PYTHONPATH")) {
        throw "Wire-v5 launch environment unexpectedly contains PYTHONPATH"
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
        -WorkingDirectory $sourceRoot `
        -RedirectStandardOutput $StandardOutputPath `
        -RedirectStandardError $StandardErrorPath `
        -WindowStyle Hidden `
        -UseNewEnvironment `
        -Environment $launchEnvironment `
        -PassThru
}

function Invoke-LaunchEnvironmentSmoke {
    # This executes only argparse --help.  It proves that the exact child
    # environment can import pcbparse from a non-ASCII workspace without
    # creating a run directory, PID record, checkpoint, or trainer process.
    $token = [Guid]::NewGuid().ToString("N")
    $temporaryRoot = [IO.Path]::GetTempPath()
    $smokeStdout = Join-Path $temporaryRoot "pcbparse-wire-v5-$token.stdout.log"
    $smokeStderr = Join-Path $temporaryRoot "pcbparse-wire-v5-$token.stderr.log"
    try {
        $smokeArguments = @(
            "-m", "pcbparse.training.torch_train_cli", "--help"
        )
        $smokeProcess = Start-BoundPythonProcess `
            $smokeArguments $smokeStdout $smokeStderr
        $smokeProcess.WaitForExit()
        $smokeOutput = (
            Get-Content -LiteralPath $smokeStdout -Raw -Encoding UTF8
        )
        $smokeError = (
            Get-Content -LiteralPath $smokeStderr -Raw -Encoding UTF8
        )
        if ($smokeProcess.ExitCode -ne 0) {
            throw "launch-environment smoke child failed with exit $($smokeProcess.ExitCode): $smokeError"
        }
        if ($smokeOutput -notlike "*--config*" -or
            $smokeOutput -notlike "*--workspace-root*" -or
            $smokeOutput -notlike "*--task*") {
            throw "launch-environment smoke did not import the expected pcbparse CLI"
        }
        Write-Output "WIRE_V5_LAUNCH_ENVIRONMENT_SMOKE_OK"
    } finally {
        Remove-Item -LiteralPath $smokeStdout -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $smokeStderr -Force -ErrorAction SilentlyContinue
    }
}

function Assert-ConfigBinding(
    [string]$ConfigPath,
    [string]$ExpectedSemanticSha256,
    [string]$ExpectedFileSha256
) {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "required training config is missing: $ConfigPath"
    }
    $actualFileSha256 = (
        Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualFileSha256 -ne $ExpectedFileSha256) {
        throw "training-config file SHA-256 differs from the reviewed queue binding: $ConfigPath"
    }
    $actualSemanticSha256 = Get-SemanticConfigSha256 $ConfigPath
    if ($actualSemanticSha256 -ne $ExpectedSemanticSha256) {
        throw "semantic training-config SHA-256 differs from the reviewed queue binding: $ConfigPath"
    }
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

function Test-OcrGateResult(
    $Gate,
    [string]$Metric,
    [double]$Threshold
) {
    try {
        $value = [double]$Gate.value
        return (
            $Gate.metric -eq $Metric -and
            $Gate.operator -eq "gte" -and
            [double]$Gate.threshold -eq $Threshold -and
            $Gate.passed -eq $true -and
            -not [double]::IsNaN($value) -and
            -not [double]::IsInfinity($value) -and
            $value -ge $Threshold
        )
    } catch {
        return $false
    }
}

function Test-OcrStatusBaseLineage($Status) {
    try {
        $compatible = @($Status.lineage.resume_compatible_training_config_sha256s)
        return (
            $Status.task -eq "ocr" -and
            $Status.smoke -eq $false -and
            $Status.lineage.task -eq "ocr" -and
            $Status.lineage.smoke -eq $false -and
            $Status.lineage.formal_evidence -eq $true -and
            $Status.lineage.training_config_sha256 -eq $expectedOcrTrainingConfigSha256 -and
            $compatible -contains $expectedOcrTrainingConfigSha256 -and
            $Status.lineage.resume.checkpoint_schema -eq $expectedResumeSchema -and
            (Test-OcrAllGatePolicy $Status.lineage)
        )
    } catch {
        return $false
    }
}

function Test-OcrPassedTerminal($Status) {
    try {
        $gates = @($Status.gates)
        return (
            [bool]$Status.finished_at_utc -and
            $Status.status -eq "candidate_evidence_ready" -and
            $Status.registerable -eq $true -and
            $Status.formal_evidence -eq $true -and
            $Status.gates_passed -eq $true -and
            (Test-OcrStatusBaseLineage $Status) -and
            $gates.Count -eq 4 -and
            (Test-OcrGateResult $gates[0] "normalized_exact_match" 0.65) -and
            (Test-OcrGateResult $gates[1] "detector_hmean" 0.45) -and
            (Test-OcrGateResult $gates[2] "recognition_designator_exact_match" 0.90) -and
            (Test-OcrGateResult $gates[3] "recognition_pinnumber_exact_match" 0.74)
        )
    } catch {
        return $false
    }
}

function Get-OcrQueueState {
    if (-not (Test-Path -LiteralPath $ocrStatus -PathType Leaf)) {
        return "pending"
    }
    try {
        $status = Get-Content -LiteralPath $ocrStatus -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return "ambiguous"
    }
    if (Test-OcrPassedTerminal $status) {
        return "passed"
    }
    if ([bool]$status.finished_at_utc) {
        # Any terminal OCR-v5 result other than the exact four-gate success is
        # final rejection, not a condition that may release this queue later.
        return "rejected"
    }
    if ($status.status -eq "running" -and (Test-OcrStatusBaseLineage $status)) {
        return "pending"
    }
    return "ambiguous"
}

function Test-WireAllGatePolicy($Lineage) {
    try {
        $selection = $Lineage.model_selection
        $gates = @($selection.gates)
        $ordering = @($selection.ordering)
        return (
            $selection.name -eq $expectedSelectionPolicy -and
            $selection.primary_metric -eq "skeleton_f1" -and
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
            $gates.Count -eq 1 -and
            $gates[0].metric -eq "skeleton_f1" -and
            $gates[0].operator -eq "gte" -and
            [double]$gates[0].threshold -eq 0.55
        )
    } catch {
        return $false
    }
}

function Test-WireStatusLineage($Status, [bool]$RequireTopLevelFormalEvidence) {
    try {
        $compatible = @($Status.lineage.resume_compatible_training_config_sha256s)
        $valid = (
            $Status.task -eq "wire" -and
            $Status.smoke -eq $false -and
            $Status.lineage.task -eq "wire" -and
            $Status.lineage.smoke -eq $false -and
            $Status.lineage.formal_evidence -eq $true -and
            $Status.lineage.training_config_sha256 -eq $expectedWireTrainingConfigSha256 -and
            $compatible -contains $expectedWireTrainingConfigSha256 -and
            $Status.lineage.resume.checkpoint_schema -eq $expectedResumeSchema -and
            (Test-WireAllGatePolicy $Status.lineage)
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
        $recordedConfig = [IO.Path]::GetFullPath([string]$pidState.training_config_path)
        $recordedUpstream = [IO.Path]::GetFullPath([string]$pidState.upstream_status_path)
        if (
            $pidState.schema -ne $pidSchema -or
            $recordedPid -le 0 -or
            [string]$pidState.task -ne "wire" -or
            [string]$pidState.upstream_task -ne "ocr" -or
            [string]$pidState.training_config_sha256 -ne $expectedWireTrainingConfigSha256 -or
            [string]$pidState.training_config_file_sha256 -ne $expectedWireConfigFileSha256 -or
            [string]$pidState.upstream_training_config_sha256 -ne $expectedOcrTrainingConfigSha256 -or
            [string]$pidState.resume_schema -ne $expectedResumeSchema -or
            [string]$pidState.model_selection_policy -ne $expectedSelectionPolicy -or
            [string]$pidState.launch_environment_mode -ne $launchEnvironmentMode -or
            [string]$pidState.launch_working_directory -ne $sourceRoot -or
            [string]$pidState.pythonpath_mode -ne "working_directory_src_only" -or
            -not $recordedOutput.Equals($wireOutput, [StringComparison]::OrdinalIgnoreCase) -or
            -not $recordedConfig.Equals($wireConfig, [StringComparison]::OrdinalIgnoreCase) -or
            -not $recordedUpstream.Equals($ocrStatus, [StringComparison]::OrdinalIgnoreCase)
        ) {
            return "ambiguous"
        }
        $recordedStart = [DateTimeOffset]::Parse([string]$pidState.started_at_utc).UtcDateTime
        $candidate = Get-Process -Id $recordedPid -ErrorAction SilentlyContinue
        if ($null -eq $candidate) {
            return "inactive"
        }
        if ($candidate.ProcessName -ne "python") {
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

foreach ($requiredPath in @($python, $ocrConfig, $wireConfig)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "required file is missing: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "required source working directory is missing: $sourceRoot"
}
if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5"
}
if ($TimeoutMinutes -lt 1) {
    throw "TimeoutMinutes must be positive"
}
Assert-ConfigBinding `
    $ocrConfig `
    $expectedOcrTrainingConfigSha256 `
    $expectedOcrConfigFileSha256
Assert-ConfigBinding `
    $wireConfig `
    $expectedWireTrainingConfigSha256 `
    $expectedWireConfigFileSha256
if ($LaunchEnvironmentSmoke) {
    Invoke-LaunchEnvironmentSmoke
    exit 0
}

$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $workspaceDigest = [BitConverter]::ToString(
        $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspaceRoot))
    ).Replace("-", "").Substring(0, 32)
} finally {
    $sha256.Dispose()
}
$mutexName = "Local\pcbparse-wire-v5-$workspaceDigest"
$launchMutex = [Threading.Mutex]::new($false, $mutexName)
$mutexAcquired = $false
try {
    try {
        $mutexAcquired = $launchMutex.WaitOne(0)
    } catch [Threading.AbandonedMutexException] {
        $mutexAcquired = $true
    }
    if (-not $mutexAcquired) {
        Write-LauncherLog "another wire-v5 watcher is already active; refusing a duplicate watcher"
        exit 7
    }

    Write-LauncherLog "waiting for exact formal OCR-v5 four-gate success terminal status"
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($true) {
        if ((Get-Date) -gt $deadline) {
            Write-LauncherLog "timeout waiting for OCR-v5 all-gate success; Wire-v5 was not launched"
            exit 2
        }
        $upstreamState = Get-OcrQueueState
        if ($upstreamState -eq "passed") {
            break
        }
        if ($upstreamState -eq "rejected") {
            Write-LauncherLog "OCR-v5 reached terminal state without passing every bound gate; Wire-v5 was not launched"
            exit 8
        }
        if ($upstreamState -eq "ambiguous") {
            Write-LauncherLog "OCR-v5 status or lineage is ambiguous; Wire-v5 was not launched"
            exit 5
        }
        Start-Sleep -Seconds $PollSeconds
    }

    $wireStatus = Join-Path $wireOutput "torch_run_status.json"
    $existingStatus = $null
    if (Test-Path -LiteralPath $wireStatus -PathType Leaf) {
        try {
            $existingStatus = Get-Content -LiteralPath $wireStatus -Raw -Encoding UTF8 | ConvertFrom-Json
        } catch {
            Write-LauncherLog "Wire-v5 status is unreadable; refusing an ambiguous launch"
            exit 5
        }
        $hasFormalTerminalStatus = (
            [bool]$existingStatus.finished_at_utc -and
            $existingStatus.status -in @("candidate_evidence_ready", "gate_failed")
        )
        if ($hasFormalTerminalStatus) {
            if (Test-WireStatusLineage $existingStatus $true) {
                Write-LauncherLog "Wire-v5 already has a compatible formal terminal status; no launch needed"
                exit 0
            }
            Write-LauncherLog "Wire-v5 terminal status has incompatible task/config/v2-policy lineage; refusing launch"
            exit 5
        }
    }

    $recordedProcessState = Get-RecordedProcessState
    if ($recordedProcessState -eq "active") {
        Write-LauncherLog "Wire-v5 already has an active recorded process; refusing a duplicate launch"
        exit 4
    }
    if ($recordedProcessState -eq "ambiguous") {
        Write-LauncherLog "Wire-v5 PID metadata is ambiguous; refusing a duplicate launch"
        exit 5
    }
    if ($null -ne $existingStatus -and $existingStatus.status -eq "running") {
        if ($recordedProcessState -ne "inactive") {
            Write-LauncherLog "Wire-v5 reports running without a provably inactive recorded process; refusing launch"
            exit 5
        }
        if (-not (Test-WireStatusLineage $existingStatus $false)) {
            Write-LauncherLog "Wire-v5 running status lacks exact task/config/v2 all-gate lineage; refusing launch"
            exit 5
        }
    }
    if (
        $null -ne $existingStatus -and
        $existingStatus.status -eq "failed" -and
        [string]$existingStatus.error.message -like "*predates the all-gate model-selection policy*"
    ) {
        Write-LauncherLog "Wire-v5 output contains a rejected v1 resume checkpoint; refusing retry"
        exit 6
    }

    $resume = Join-Path $wireOutput "resume_checkpoint.pt"
    if (Test-Path -LiteralPath $wireOutput) {
        $entries = @(Get-ChildItem -LiteralPath $wireOutput -Force)
        if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $resume -PathType Leaf)) {
            Write-LauncherLog "Wire-v5 output is non-empty without its own resume checkpoint; refusing overwrite"
            exit 3
        }
    }

    $arguments = @(
        "-m", "pcbparse.training.torch_train_cli",
        "--config", $wireConfig,
        "--task", "wire",
        "--output", $wireOutput,
        "--workspace-root", $workspaceRoot,
        "--device", "cuda"
    )
    if (Test-Path -LiteralPath $resume -PathType Leaf) {
        if ($null -eq $existingStatus -or -not (Test-WireStatusLineage $existingStatus $false)) {
            Write-LauncherLog "Wire-v5 resume metadata is not the exact fresh-v5/v2/all-gate lineage; refusing resume"
            exit 6
        }
        $arguments += @("--resume-checkpoint", $resume)
        Write-LauncherLog "Wire-v5 will resume only from its verified v2 all-gate epoch checkpoint"
    }

    $process = Start-BoundPythonProcess $arguments $stdoutLog $stderrLog
    $pidState = [ordered]@{
        schema = $pidSchema
        pid = $process.Id
        started_at_utc = $process.StartTime.ToUniversalTime().ToString("o")
        task = "wire"
        output = $wireOutput
        training_config_path = $wireConfig
        training_config_sha256 = $expectedWireTrainingConfigSha256
        training_config_file_sha256 = $expectedWireConfigFileSha256
        upstream_task = "ocr"
        upstream_status_path = $ocrStatus
        upstream_training_config_sha256 = $expectedOcrTrainingConfigSha256
        resume_schema = $expectedResumeSchema
        model_selection_policy = $expectedSelectionPolicy
        launch_environment_mode = $launchEnvironmentMode
        launch_working_directory = $sourceRoot
        pythonpath_mode = "working_directory_src_only"
    }
    $pidTemporary = "$pidFile.$PID.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $pidState | ConvertTo-Json | Set-Content -LiteralPath $pidTemporary -Encoding UTF8
        Move-Item -LiteralPath $pidTemporary -Destination $pidFile -Force
    } catch {
        # Never leave an untracked trainer behind if durable PID publication
        # fails after Start-Process succeeds.
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw
    } finally {
        Remove-Item -LiteralPath $pidTemporary -Force -ErrorAction SilentlyContinue
    }
    Write-LauncherLog "Wire-v5 launched (pid $($process.Id))"
} finally {
    if ($mutexAcquired) {
        try {
            $launchMutex.ReleaseMutex()
        } catch {
            # Process exit still releases the owned local mutex.
        }
    }
    $launchMutex.Dispose()
}
