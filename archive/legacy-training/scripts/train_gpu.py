#!/usr/bin/env python
"""Train models on GPU."""

import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
from _paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT, TRAIN_CONFIG
sys.path.insert(0, str(REPO_ROOT / 'src'))

os.environ['PCBPARSE_TRAIN_DEVICE'] = 'cuda'

from pathlib import Path
from pcbparse.training.torch_train_cli import main as train_main

WORKSPACE = str(REPO_ROOT)
CONFIG = str(TRAIN_CONFIG)
OUTPUT_BASE = str(RUNS_ROOT / 'training_gpu')

os.makedirs(OUTPUT_BASE, exist_ok=True)

# Train pin task on GPU with more epochs
print('Training pin detection on GPU...')
args = [
    '--config', CONFIG,
    '--task', 'pin',
    '--output', os.path.join(OUTPUT_BASE, 'pin'),
    '--workspace-root', WORKSPACE,
    '--device', 'cuda',
    '--smoke',
    '--smoke-epochs', '10',
    '--smoke-train-batches', '20',
    '--smoke-val-batches', '10',
]

start = time.time()
result = train_main(args)
elapsed = time.time() - start
print(f'Pin training: result={result}, time={elapsed:.1f}s')

# Check status
status_file = os.path.join(OUTPUT_BASE, 'pin', 'torch_run_status.json')
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
