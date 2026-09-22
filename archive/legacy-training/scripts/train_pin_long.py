#!/usr/bin/env python
"""Long Pin training 200 epochs with weighted BCE."""

import sys, os, json, time, shutil
sys.stdout.reconfigure(encoding='utf-8')
from _paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT, TRAIN_CONFIG
sys.path.insert(0, str(REPO_ROOT / 'src'))

os.environ['PCBPARSE_TRAIN_DEVICE'] = 'cuda'
os.environ['PCBPARSE_TRAIN_PRECISION'] = 'fp32'

from pathlib import Path
from pcbparse.training.torch_train_cli import main as train_main

WORKSPACE = str(REPO_ROOT)
CONFIG = str(TRAIN_CONFIG)
OUTPUT = str(RUNS_ROOT / 'training_pin_long')

if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
os.makedirs(OUTPUT, exist_ok=True)

print('Training Pin 200 epochs with weighted BCE...')
print(f'Device: CUDA (RTX 3050 Ti 4GB)')

args = [
    '--config', CONFIG,
    '--task', 'pin',
    '--output', OUTPUT,
    '--workspace-root', WORKSPACE,
    '--device', 'cuda',
    '--precision', 'fp32',
    '--smoke',
    '--smoke-epochs', '200',
    '--smoke-train-batches', '100',
    '--smoke-val-batches', '50',
]

start = time.time()
result = train_main(args)
elapsed = time.time() - start

print(f'Pin: result={result}, time={elapsed:.1f}s')

status_file = os.path.join(OUTPUT, 'torch_run_status.json')
if os.path.exists(status_file):
    with open(status_file, 'r') as f:
        status = json.load(f)
    metrics = status.get('metrics', {})
    completed = status.get('completed_epochs', 0)
    print(f'Completed epochs: {completed}')
    print(f'Metrics: {metrics}')
