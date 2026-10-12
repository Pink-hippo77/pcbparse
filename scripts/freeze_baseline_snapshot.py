"""Freeze a reproducible baseline snapshot (A00).

Records dirty source, model, config and data hashes, and separates
historical / candidate / main-lock metrics. Does not modify the model lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCOPES = (
    "src",
    "tests",
    "scripts",
    "data_tools",
    "configs",
    "models",
    "docker",
    "pyproject.toml",
    "README.md",
)

SKIP_DIR_PARTS = {
    "__pycache__",
    ".git",
    ".venv",
    ".venv-test",
    ".venv-train",
    ".venv-yolo",
    ".pytest_cache",
    "node_modules",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path, scopes: list[str]) -> list[Path]:
    out: list[Path] = []
    for scope in scopes:
        p = root / scope
        if p.is_file():
            out.append(p)
            continue
        if not p.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS and not d.startswith(".venv")]
            for name in filenames:
                if name.endswith((".pyc", ".pyo")):
                    continue
                out.append(Path(dirpath) / name)
    return sorted(out)


def git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        info = {
            "available": True,
            "head": head,
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines()[:200],
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main_lock_metrics(lock: dict[str, Any]) -> dict[str, Any]:
    models = lock.get("models") or []
    tasks: dict[str, Any] = {}
    for item in models if isinstance(models, list) else []:
        task = item.get("task") or item.get("name")
        if not task:
            continue
        metrics = item.get("metrics") or item.get("evaluation") or {}
        tasks[task] = {
            "path": item.get("path"),
            "blessed": item.get("blessed"),
            "status": item.get("status"),
            "sha256": item.get("sha256"),
            "metrics_recorded_in_lock": metrics,
            "metrics_kind": "historical_locked_not_rerun",
        }
    return {
        "required_tasks": lock.get("required_tasks"),
        "junction_mode": lock.get("junction_mode"),
        "status": lock.get("status"),
        "blessed": lock.get("blessed"),
        "created_at_utc": lock.get("created_at_utc"),
        "lineage_contract": lock.get("lineage_contract"),
        "tasks": tasks,
    }


def historical_status_bundle(root: Path) -> dict[str, Any]:
    candidates = [
        root / "docs" / "optimization-20260908" / "authoritative_status.json",
        root / "artifacts" / "audits" / "completion_readiness_audit.json",
    ]
    found: dict[str, Any] = {}
    for p in candidates:
        if p.is_file():
            try:
                found[str(p.relative_to(root)).replace("\\", "/")] = {
                    "kind": "historical_snapshot_not_rerun",
                    "sha256": sha256_file(p),
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            except OSError:
                continue
    return found


def readiness_labels() -> dict[str, Any]:
    """Explicit status vocabulary — runtime_ready is never release_ready."""
    return {
        "runtime_loadable": None,
        "strict_parse_pass_rate": None,
        "degraded_output_rate": None,
        "quality_gate_passed": False,
        "release_evidence_complete": False,
        "production_ready": False,
        "note": (
            "Unset numeric fields require a same-run measurement. "
            "quality_gate_passed/release_evidence_complete stay false until A13/A14 evidence exists."
        ),
    }


def build_manifest(root: Path, scopes: list[str]) -> dict[str, Any]:
    files = iter_files(root, scopes)
    entries = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            st = path.stat()
            digest = sha256_file(path)
        except OSError:
            continue
        entries.append(
            {
                "path": rel,
                "bytes": st.st_size,
                "sha256": digest,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return {
        "schema_version": "1.0",
        "kind": "source_manifest",
        "workspace": str(root),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "scopes": scopes,
        "file_count": len(entries),
        "files": entries,
    }


def build_baseline(root: Path, manifest: dict[str, Any], scopes: list[str]) -> dict[str, Any]:
    lock_path = root / "models" / "models.lock.json"
    lock = read_json(lock_path) if lock_path.is_file() else {}
    config_files = sorted((root / "configs").rglob("*")) if (root / "configs").is_dir() else []
    config_hashes = {}
    for p in config_files:
        if p.is_file() and p.suffix in {".yaml", ".yml", ".json"}:
            config_hashes[p.relative_to(root).as_posix()] = sha256_file(p)

    data_manifest = root / "data" / "manifests" / "dataset300.jsonl"
    split_manifest = root / "data" / "splits" / "dataset300_split.jsonl"

    return {
        "schema_version": "1.0",
        "kind": "baseline_snapshot",
        "task_id": "A00",
        "workspace": str(root),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git": git_info(root),
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "source_file_count": manifest["file_count"],
        "model_lock": {
            "path": "models/models.lock.json",
            "sha256": sha256_file(lock_path) if lock_path.is_file() else None,
            "metrics_kind": "main_lock_historical",
            **main_lock_metrics(lock),
        },
        "config_sha256": config_hashes,
        "data": {
            "dataset300_manifest": {
                "path": "data/manifests/dataset300.jsonl",
                "sha256": sha256_file(data_manifest) if data_manifest.is_file() else None,
            },
            "dataset300_split": {
                "path": "data/splits/dataset300_split.jsonl",
                "sha256": sha256_file(split_manifest) if split_manifest.is_file() else None,
            },
            "public_dataset_root": "赛题六公开数据集",
            "note": "Public images are byte-overlapping with Dataset300 and must not be claimed as new holdout.",
        },
        "metrics_taxonomy": {
            "historical": historical_status_bundle(root),
            "candidate": {
                "kind": "candidate_metrics_require_shadow_ab",
                "note": "YOLO/v2 candidate numbers never describe the main lock until promotion.",
            },
            "main_lock": "see model_lock.metrics_recorded_in_lock",
        },
        "readiness": readiness_labels(),
        "acceptance": {
            "snapshot_reproducible": True,
            "runtime_ready_is_not_release_ready": True,
            "main_lock_untouched": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze baseline + source manifest (A00)")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <workspace>/artifacts/audits",
    )
    parser.add_argument(
        "--scopes",
        nargs="*",
        default=list(DEFAULT_SCOPES),
    )
    args = parser.parse_args(argv)

    root = args.workspace_root.resolve()
    out_dir = (args.output_dir or (root / "artifacts" / "audits")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root, args.scopes)
    baseline = build_baseline(root, manifest, args.scopes)

    manifest_path = out_dir / "source_manifest.json"
    baseline_path = out_dir / "baseline.json"

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "baseline": str(baseline_path),
            "source_manifest": str(manifest_path),
            "source_file_count": manifest["file_count"],
            "source_manifest_sha256": baseline["source_manifest_sha256"],
            "model_lock_sha256": baseline["model_lock"]["sha256"],
            "production_ready": baseline["readiness"]["production_ready"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
