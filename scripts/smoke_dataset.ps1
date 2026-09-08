param(
    [string]$Python = 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$DatasetRoot = "$PSScriptRoot\..\Dataset300\Dataset300"
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\src").Path
$env:PYTHONUTF8 = '1'

$workspace = (Resolve-Path "$PSScriptRoot\..").Path
$manifest = Join-Path $workspace 'data\manifests\dataset300.jsonl'
$audit = Join-Path $workspace 'artifacts\audits\dataset300_audit.json'
$split = Join-Path $workspace 'data\splits\dataset300_split.jsonl'

& $Python -m data_tools.pcbnet_ingest --dataset-root $DatasetRoot --output $manifest --expected-count 300
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m data_tools.audit --dataset-root $DatasetRoot --output $audit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m data_tools.split_grouped --manifest $manifest --output $split --seed 20260812
exit $LASTEXITCODE
