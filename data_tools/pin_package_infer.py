"""Run the imported Pin module with locked Component/OCR/Wire adapters.

This entry point is explicitly experimental and never promotes a model lock.
It preserves strict validation and emits replayable Circuit-IR on failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pcbparse.config import load_config
from pcbparse.pipeline.context import stable_file_hash


def _reject_sidecar_reads(event, args):
    if event == "open" and args:
        path = str(args[0]).replace("\\", "/")
        if ("_CVJsonStd" in path or "/oracle_diagnostic/" in path or "/offline_evaluation/" in path
                or path.endswith((".eds", "_spice.net", "_geometry.json"))):
            raise RuntimeError(f"Pin experiment prohibits target/sidecar access: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--input-manifest", type=Path, help="JSON array with image_path and image_sha256; no targets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.pin-package-experimental.yaml"))
    parser.add_argument("--models-lock", type=Path, default=Path("models/models.lock.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cpu":
        # ORT defaults can oversubscribe every ROI on a shared server.
        import onnxruntime as ort
        original_session = ort.InferenceSession
        def bounded_session(path, sess_options=None, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            return original_session(path, sess_options=options, **kwargs)
        ort.InferenceSession = bounded_session
    from pcbparse.cli import _discover_images
    from pcbparse.perception.pin_package_provider import load_pin_package_experiment
    from pcbparse.pipeline.orchestrator import infer_batch
    config = load_config(args.config)
    if args.input_manifest:
        rows = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        images = []
        for row in rows:
            image = Path(row["image_path"])
            if stable_file_hash(image) != row["image_sha256"]:
                raise ValueError(f"input image SHA256 mismatch: {image}")
            images.append(image)
    else:
        images = _discover_images(args.input)
    if not images:
        raise ValueError("no input images")
    provider = load_pin_package_experiment(args.models_lock, config, device=args.device)
    started = time.monotonic()
    deadline = started + float(config["run"]["max_total_seconds"])
    original_build_ir = provider.build_ir
    def bounded_build_ir(path, ctx=None):
        if ctx is not None:
            ctx.time_budget.deadline_at = deadline
        return original_build_ir(path, ctx)
    provider.build_ir = bounded_build_ir
    sys.addaudithook(_reject_sidecar_reads)
    def evidence_sink(stage, ir):
        target = args.output / "evidence" / ir.image.image_id / f"{stage}.circuit-ir.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ir.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    results, errors = infer_batch(images, args.output, provider=provider, config=config, evidence_sink=evidence_sink)
    report = {"execution_mode": "pin_package_experiment", "production_ready": False,
              "training_lineage_verified": False, "strict_validation": True,
              "sidecar_access": False, "sidecar_reads_blocked_by_audit_hook": True,
              "inputs": len(images), "succeeded": len(results), "failed": len(errors),
              "elapsed_seconds": time.monotonic() - started, "torch_threads": args.threads,
              "models_lock_sha256": stable_file_hash(args.models_lock),
              "config_sha256": stable_file_hash(args.config), "pin": provider.experiment_metadata,
              "error_manifest": str(args.output / "error_manifest.json") if errors else None}
    (args.output / "experiment_manifest.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
