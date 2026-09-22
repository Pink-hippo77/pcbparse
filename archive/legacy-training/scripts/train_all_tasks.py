#!/usr/bin/env python
"""Train all PCB parsing tasks on CPU with reduced epochs for competition readiness."""

import sys
import os
import json
import time
from pathlib import Path

# Add source to path
from _paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT, TRAIN_CONFIG
sys.path.insert(0, str(REPO_ROOT / 'src'))

# Force CPU training
os.environ['PCBPARSE_TRAIN_DEVICE'] = 'cpu'

from pcbparse.training.torch_train_cli import main as train_main

WORKSPACE = str(REPO_ROOT)
CONFIG = str(TRAIN_CONFIG)
OUTPUT_BASE = str(RUNS_ROOT / 'training')

# Training tasks with reduced epochs for CPU training
TASKS = {
    'component': {'epochs': 3, 'train_batches': 10, 'val_batches': 5},
    'pin': {'epochs': 5, 'train_batches': 10, 'val_batches': 5},
    'ocr': {'epochs': 5, 'train_batches': 10, 'val_batches': 5},
    'wire': {'epochs': 3, 'train_batches': 10, 'val_batches': 5},
    'junction': {'epochs': 5, 'train_batches': 10, 'val_batches': 5},
}

def train_task(task_name, epochs, train_batches, val_batches):
    """Train a single task."""
    output_dir = os.path.join(OUTPUT_BASE, task_name)

    # Remove existing output
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print(f"\n{'='*60}")
    print(f"Training {task_name}...")
    print(f"{'='*60}")

    args = [
        '--config', CONFIG,
        '--task', task_name,
        '--output', output_dir,
        '--workspace-root', WORKSPACE,
        '--device', 'cpu',
        '--smoke',
        '--smoke-epochs', str(epochs),
        '--smoke-train-batches', str(train_batches),
        '--smoke-val-batches', str(val_batches),
    ]

    start = time.time()
    result = train_main(args)
    elapsed = time.time() - start

    print(f"Task {task_name}: result={result}, time={elapsed:.1f}s")

    # Read the status file
    status_file = os.path.join(output_dir, 'torch_run_status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
        metrics = status.get('metrics', {})
        gates = status.get('gates', [])
        print(f"  Metrics: {metrics}")
        for gate in gates:
            passed = gate.get('passed', False)
            metric = gate.get('metric', '')
            value = gate.get('value', 0)
            threshold = gate.get('threshold', 0)
            print(f"  Gate {metric}: {value:.4f} >= {threshold} -> {'PASS' if passed else 'FAIL'}")

    return result

def main():
    """Train all tasks."""
    print("Starting PCB parsing model training...")
    print(f"Device: CPU (no GPU available)")
    print(f"Output: {OUTPUT_BASE}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = {}
    total_start = time.time()

    for task_name, config in TASKS.items():
        result = train_task(
            task_name,
            config['epochs'],
            config['train_batches'],
            config['val_batches'],
        )
        results[task_name] = result

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    for task, result in results.items():
        status = "OK" if result == 0 else "FAILED"
        print(f"  {task}: {status}")

    return 0 if all(r == 0 for r in results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
