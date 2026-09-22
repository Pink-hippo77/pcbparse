"""Reproducible YOLO11n Component training launcher.

Run a short CUDA/data smoke test first:
    .venv-yolo/Scripts/python.exe scripts/train_yolo_component.py --mode smoke

Then start the complete training run:
    .venv-yolo/Scripts/python.exe scripts/train_yolo_component.py --mode train
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "train" / "yolo_component_y11n_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"training config must be a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _completed_training_status(
    *,
    config: dict[str, Any],
    config_path: Path,
    model_path: Path,
    dataset_path: Path,
    save_dir: Path,
    mode: str,
) -> dict[str, Any]:
    results_path = save_dir / "results.csv"
    rows: list[dict[str, str]] = []
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    parsed = [
        {
            "epoch": int(row["epoch"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }
        for row in rows
        if row.get("epoch") not in (None, "")
    ]
    best = (
        max(parsed, key=lambda row: (row["map50_95"], row["map50"], -row["epoch"]))
        if parsed
        else None
    )
    artifacts: dict[str, Any] = {}
    for name in ("results.csv", "weights/best.pt", "weights/last.pt"):
        path = save_dir / name
        if path.is_file() and path.stat().st_size > 0:
            artifacts[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    configured_epochs = 1 if mode == "smoke" else int(config["epochs"])
    # Current Ultralytics results.csv uses one-based epoch numbers.
    completed_epochs = parsed[-1]["epoch"] if parsed else 0
    return {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_training_launcher_status",
        "status": "training_complete",
        "ok": True,
        "mode": mode,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "run_name": str(config["run_name"]),
        "save_dir": str(save_dir),
        "training_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "input_model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
        },
        "dataset": {
            "path": str(dataset_path),
            "conversion_config_sha256": str(config["dataset_config_sha256"]),
        },
        "configured_epochs": configured_epochs,
        "completed_epochs": completed_epochs,
        "patience": int(config["patience"]),
        "stopped_before_epoch_limit": completed_epochs < configured_epochs,
        "epoch_indexing": "results_csv_one_based",
        "best_epoch": best["epoch"] if best is not None else None,
        "best_epoch_zero_based": best["epoch"] - 1 if best is not None else None,
        "best_map50": best["map50"] if best is not None else None,
        "best_map50_95": best["map50_95"] if best is not None else None,
        "artifacts": artifacts,
        "models_lock_modified": False,
    }


def _verify_inputs(config: dict[str, Any]) -> tuple[Path, Path]:
    model_path = REPOSITORY_ROOT / str(config["model"])
    dataset_path = REPOSITORY_ROOT / str(config["dataset"])
    if not model_path.is_file():
        raise FileNotFoundError(f"pretrained model is missing: {model_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset YAML is missing: {dataset_path}")

    actual_model_hash = _sha256(model_path)
    if actual_model_hash != str(config["model_sha256"]).lower():
        raise ValueError(
            f"model SHA-256 mismatch: expected={config['model_sha256']} actual={actual_model_hash}"
        )

    audit_path = dataset_path.parents[1] / "conversion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_dataset_hash = str(audit.get("config_sha256", ""))
    if actual_dataset_hash != str(config["dataset_config_sha256"]).lower():
        raise ValueError(
            "dataset conversion config mismatch: "
            f"expected={config['dataset_config_sha256']} actual={actual_dataset_hash}"
        )
    return model_path, dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    model_path, dataset_path = _verify_inputs(config)

    # Keep Ultralytics settings and downloads inside the ignored workspace.
    os.environ.setdefault(
        "ULTRALYTICS_CONFIG_DIR", str(REPOSITORY_ROOT / "work" / "ultralytics-config")
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != str(config["ultralytics_version"]):
        raise RuntimeError(
            f"Ultralytics version mismatch: expected={config['ultralytics_version']} "
            f"actual={ultralytics.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start the GPU training contract")

    excluded = {
        "schema_version",
        "run_name",
        "model",
        "model_sha256",
        "dataset",
        "dataset_config_sha256",
        "ultralytics_version",
        "task",
    }
    train_args = {key: value for key, value in config.items() if key not in excluded}
    train_args.update(
        data=str(dataset_path),
        project=str(REPOSITORY_ROOT / "work" / "yolo"),
        name=str(config["run_name"]),
        exist_ok=False,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    if args.mode == "smoke":
        train_args.update(
            name=f"{config['run_name']}-smoke",
            epochs=1,
            fraction=0.03,
            val=False,
            plots=False,
            save=False,
            save_period=-1,
            mosaic=0.0,
            close_mosaic=0,
        )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "model": str(model_path),
                "dataset": str(dataset_path),
                "output": str(Path(train_args["project"]) / str(train_args["name"])),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = YOLO(str(model_path))
    model.train(**train_args)
    save_dir = Path(model.trainer.save_dir).resolve(strict=True)
    status = _completed_training_status(
        config=config,
        config_path=config_path,
        model_path=model_path,
        dataset_path=dataset_path,
        save_dir=save_dir,
        mode=args.mode,
    )
    status_path = save_dir / "launcher_status.json"
    _atomic_json(status_path, status)
    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "save_dir": str(save_dir),
                "status": str(status_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
