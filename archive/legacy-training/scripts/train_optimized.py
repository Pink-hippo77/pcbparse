#!/usr/bin/env python
"""Long training for all modules - 50 epochs."""

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
OUTPUT_BASE = str(RUNS_ROOT / 'training_optimized')

TASKS = {
    'component': {'epochs': 50, 'train_batches': 100, 'val_batches': 50},
    'pin': {'epochs': 50, 'train_batches': 100, 'val_batches': 50},
    'wire': {'epochs': 50, 'train_batches': 100, 'val_batches': 50},
    'ocr': {'epochs': 50, 'train_batches': 100, 'val_batches': 50},
}

def train_task(task_name, config):
    output_dir = os.path.join(OUTPUT_BASE, task_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Training {task_name} ({config["epochs"]} epochs)...')
    print(f'{"="*60}')

    args = [
        '--config', CONFIG,
        '--task', task_name,
        '--output', output_dir,
        '--workspace-root', WORKSPACE,
        '--device', 'cuda',
        '--precision', 'fp32',
        '--smoke',
        '--smoke-epochs', str(config['epochs']),
        '--smoke-train-batches', str(config['train_batches']),
        '--smoke-val-batches', str(config['val_batches']),
    ]

    start = time.time()
    result = train_main(args)
    elapsed = time.time() - start

    print(f'{task_name}: result={result}, time={elapsed:.1f}s')

    # Check status
    status_file = os.path.join(output_dir, 'torch_run_status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
        metrics = status.get('metrics', {})
        gates = status.get('gates', [])
        print(f'  Metrics: {metrics}')
        for gate in gates:
            passed = gate.get('passed', False)
            metric = gate.get('metric', '')
            value = gate.get('value', 0)
            threshold = gate.get('threshold', 0)
            status_str = 'PASS' if passed else 'FAIL'
            print(f'  Gate {metric}: {value:.4f} >= {threshold} -> {status_str}')

    return result

if __name__ == '__main__':
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print('Starting optimized training (50 epochs)...')
    print(f'Device: CUDA (RTX 3050 Ti 4GB)')
    print(f'Output: {OUTPUT_BASE}')

    results = {}
    total_start = time.time()

    for task_name, config in TASKS.items():
        result = train_task(task_name, config)
        results[task_name] = result

    total_elapsed = time.time() - total_start

    print(f'\n{"="*60}')
    print('Training Summary')
    print(f'{"="*60}')
    print(f'Total time: {total_elapsed:.1f}s')
    for task, result in results.items():
        status = 'OK' if result == 0 else 'FAILED'
        print(f'  {task}: {status}')
