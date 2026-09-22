#!/usr/bin/env python
"""Evaluate PCB parsing system against golden annotations."""

import sys
import json
import time
from pathlib import Path

from _paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT, TRAIN_CONFIG
sys.path.insert(0, str(REPO_ROOT / 'src'))

from pcbparse.pipeline.orchestrator import infer_batch
from pcbparse.config import load_config
from pcbparse.perception.golden_provider import DatasetGoldenProvider
from pcbparse.evaluation.official_metrics import evaluate_official_json, aggregate_reports

def load_golden_json(image_path: Path) -> dict:
    """Load golden JSON for an image."""
    golden_path = image_path.parent / f'{image_path.stem}_CVJsonStd.json'
    if not golden_path.exists():
        return None
    with open(golden_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Run evaluation."""
    print("PCB Parsing System Evaluation")
    print("=" * 60)

    # Load config and provider
    config = load_config(REPO_ROOT / 'configs' / 'pipeline.yaml')
    provider = DatasetGoldenProvider(policy='dataset_debug')

    # Get all images
    dataset_dir = REPO_ROOT / 'Dataset300' / 'Dataset300'
    images = []
    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            for f in folder.iterdir():
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                    images.append(f)
                    break

    print(f"Found {len(images)} images")

    # Process images and collect results
    output_dir = RESULTS_ROOT / 'inference' / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running inference...")
    start = time.time()
    results, errors = infer_batch(images, output_dir, provider=provider, config=config, continue_on_error=True)
    elapsed = time.time() - start
    print(f"Inference: {len(results)} succeeded, {len(errors)} failed in {elapsed:.2f}s")

    # Evaluate against golden
    print("\nEvaluating against golden annotations...")
    reports = []
    for result in results:
        golden = load_golden_json(result.input_path)
        if golden is None:
            continue

        # Load predicted JSON
        with open(result.output_path, 'r', encoding='utf-8') as f:
            predicted = json.load(f)

        try:
            report = evaluate_official_json(predicted, golden)
            reports.append(report)
        except Exception as e:
            print(f"  Error evaluating {result.input_path.parent.name}: {e}")

    if not reports:
        print("No valid evaluations!")
        return 1

    # Aggregate results
    aggregate = aggregate_reports(reports)

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Evaluated: {len(reports)} images")
    print(f"\nComponent F1 (30% weight): {aggregate.component.f1:.4f}")
    print(f"  Precision: {aggregate.component.precision:.4f}")
    print(f"  Recall: {aggregate.component.recall:.4f}")

    print(f"\nPin F1 (25% weight): {aggregate.pin.f1:.4f}")
    print(f"  Precision: {aggregate.pin.precision:.4f}")
    print(f"  Recall: {aggregate.pin.recall:.4f}")

    print(f"\nNetHypergraph F1 (35% weight): {aggregate.net_hypergraph.f1:.4f}")
    print(f"  Precision: {aggregate.net_hypergraph.precision:.4f}")
    print(f"  Recall: {aggregate.net_hypergraph.recall:.4f}")

    print(f"\nNetLine F1 (10% weight): {aggregate.net_line.f1:.4f}")
    print(f"  Precision: {aggregate.net_line.precision:.4f}")
    print(f"  Recall: {aggregate.net_line.recall:.4f}")

    print(f"\nWeighted Score: {aggregate.weighted_score:.4f}")

    # Save detailed results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate.as_dict(), f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
