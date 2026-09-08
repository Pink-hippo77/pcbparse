param(
    [string]$Python = 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path "$PSScriptRoot\..\src").Path
$env:PYTHONUTF8 = '1'

& $Python -m unittest discover -s "$PSScriptRoot\..\tests\unit" -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m unittest discover -s "$PSScriptRoot\..\tests\integration" -p 'test_*.py' -v
exit $LASTEXITCODE

