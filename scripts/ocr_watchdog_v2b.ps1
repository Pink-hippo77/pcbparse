# OCR watchdog: restart auto-resume training when it stalls
$workspace = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $workspace 'work\train-ocr-v2b'
$logPath = Join-Path $outDir 'training_log.jsonl'
$watchLog = Join-Path $workspace 'work\ocr_watchdog_v2.log'
$configPath = Join-Path $workspace 'configs\train\dataset300_v2_torch.json'
$python = Join-Path $workspace '.venv-train\Scripts\python.exe'
$stallMinutes = 90
$checkSeconds = 300

$env:PYTHONPATH = (Resolve-Path (Join-Path $workspace 'src')).Path

function Write-Log([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -LiteralPath $watchLog -Value $line
}

function Start-OcrTraining {
    Start-Process -FilePath $python -ArgumentList @(
        '-m','pcbparse.training.torch_train_cli',
        '--config',$configPath,
        '--task','ocr',
        '--output',$outDir,
        '--workspace-root',$workspace,
        '--device','cuda',
        '--precision','fp16'
    ) -WorkingDirectory $workspace -WindowStyle Hidden
}

Write-Log 'watchdog started'
while ($true) {
    Start-Sleep -Seconds $checkSeconds
    $metrics = Join-Path $outDir 'metrics.json'
    $done = $false
    if (Test-Path $metrics) {
        try {
            $m = Get-Content -Raw -Encoding UTF8 $metrics | ConvertFrom-Json
            if ($m.formal_evidence -eq $true) { $done = $true }
        } catch { $done = $false }
    }
    if ($done) {
        Write-Log 'training complete; watchdog exit'
        break
    }
    $trainers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'torch_train_cli' -and $_.CommandLine -match '--task ocr' })
    if ($trainers.Count -eq 0) {
        Write-Log 'no training process; relaunch'
        Start-OcrTraining
        continue
    }
    $age = 0.0
    if (Test-Path $logPath) {
        $age = (New-TimeSpan -Start (Get-Item $logPath).LastWriteTime -End (Get-Date)).TotalMinutes
    }
    if ($age -gt $stallMinutes) {
        Write-Log ('STALL log age={0}min killing pid={1}' -f [math]::Round($age,1), $trainers[0].ProcessId)
        Stop-Process -Id $trainers[0].ProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 10
        Write-Log 'relaunch after stall'
        Start-OcrTraining
    }
}
Write-Log 'watchdog exit'

