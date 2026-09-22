#!/usr/bin/env python
"""Quick GPU smoke test."""

import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
from _paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT, TRAIN_CONFIG
sys.path.insert(0, str(REPO_ROOT / 'src'))

os.environ['PCBPARSE_TRAIN_DEVICE'] = 'cuda'
os.environ['PCBPARSE_TRAIN_PRECISION'] = 'fp32'

from pathlib import Path
from pcbparse.training.torch_train_cli import main as train_main

WORKSPACE = str(REPO_ROOT)
CONFIG = str(TRAIN_CONFIG)
OUTPUT = str(RUNS_ROOT / 'training_smoke_gpu')

if os.path.exists(OUTPUT):
    import shutil
    shutil.rmtree(OUTPUT)

print('Running GPU smoke test...')
args = [
    '--config', CONFIG,
    '--task', 'pin',
    '--output', OUTPUT,
    '--workspace-root', WORKSPACE,
    '--device', 'cuda',
    '--precision', 'fp32',
    '--smoke',
    '--smoke-epochs', '3',
    '--smoke-train-batches', '10',
    '--smoke-val-batches', '5',
]

start = time.time()
result = train_main(args)
elapsed = time.time() - start
print(f'Result: {result}, Time: {elapsed:.1f}s')

# Check status
status_file = os.path.join(OUTPUT, 'torch_run_status.json')
if os.path.exists(status_file):
    with open(status_file, 'r') as f:
        status = json.load(f)
    metrics = status.get('metrics', {})
    gates = status.get('gates', [])
    print(f'Metrics: {metrics}')
    for gate in gates:
        passed = gate.get('passed', False)
        metric = gate.get('metric', '')
        value = gate.get('value', 0)
        threshold = gate.get('threshold', 0)
        status_str = 'PASS' if passed else 'FAIL'
        print(f'Gate {metric}: {value:.4f} >= {threshold} -> {status_str}')
