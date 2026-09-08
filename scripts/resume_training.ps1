# 一键断点续训所有训练任务（自动跳过已正式完成的任务）
param(
    [string]$Config = "configs\train\dataset300_v2_torch.json",
    [string]$WorkspaceRoot = "E:\EDA挑战题目六",
    [string]$Device = "cuda",
    [string]$Precision = "fp16"
)

$env:PYTHONPATH = (Resolve-Path (Join-Path $WorkspaceRoot "src")).Path
$python = Join-Path $WorkspaceRoot ".venv-train\Scripts\python.exe"
$configPath = Join-Path $WorkspaceRoot $Config

$taskDirs = [ordered]@{
    "junction"  = "work\train-junction-formal"
    "pin"       = "work\train-pin-formal"
    "wire"      = "work\train-wire-formal4"
    "component" = "work\train-component-formal4"
    "ocr"       = "work\train-ocr-resume"
}

foreach ($task in $taskDirs.Keys) {
    $out = Join-Path $WorkspaceRoot $taskDirs[$task]
    $ckpt = Join-Path $out "resume_checkpoint.pt"
    $metrics = Join-Path $out "metrics.json"
    $done = $false
    if (Test-Path $metrics) {
        try {
            $m = Get-Content -Raw -Encoding UTF8 $metrics | ConvertFrom-Json
            if ($m.formal_evidence -eq $true) { $done = $true }
        } catch { $done = $false }
    }
    if ($done) {
        Write-Host "== $task : already completed (formal evidence), skip =="
        continue
    }
    if (-not (Test-Path $ckpt)) {
        Write-Host "== $task : no resume checkpoint, skip =="
        continue
    }
    Write-Host "== $task : resuming from $out =="
    & $python -m pcbparse.training.torch_train_cli `
        --config $configPath --task $task --output $out `
        --workspace-root $WorkspaceRoot --device $Device --precision $Precision `
        --resume-checkpoint $ckpt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "== $task : resume failed (exit $LASTEXITCODE) =="
    }
}
Write-Host "resume pass finished"
