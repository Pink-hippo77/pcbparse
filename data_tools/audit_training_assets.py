from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import ensure_source_path

ensure_source_path()

from pcbparse.training.config import load_training_config, training_config_sha256  # noqa: E402


def audit_training_assets(
    config_path: str | Path,
    *,
    workspace_root: str | Path,
    tasks: Sequence[str] | None = None,
) -> dict:
    """Audit physical initialization assets; this does not certify data/model quality."""
    root = Path(workspace_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = root / path
    report: dict = {
        "schema_version": 1,
        "scope": "training_initialization_assets_only",
        "workspace_root": str(root),
        "config_path": str(path),
        "ok": False,
        "assets": [],
        "issues": [],
    }
    try:
        path.resolve().relative_to(root)
        config = load_training_config(path)
    except (ValueError, OSError) as exc:
        report["issues"].append({"code": "config_invalid", "message": str(exc)})
        return report
    report["training_config_sha256"] = training_config_sha256(config)
    selected = sorted(set(tasks if tasks is not None else config.tasks))
    for task in selected:
        if task not in config.tasks:
            report["issues"].append({"code": "task_not_configured", "task": task})
            continue
        artifact = config.tasks[task].pretrained_artifact
        if artifact is None:
            report["assets"].append({"task": task, "status": "random_initialization"})
            continue
        asset_path = (root / artifact.path).resolve()
        row = {
            "task": task,
            "path": artifact.path,
            "expected_sha256": artifact.sha256,
            "source": artifact.source,
            "license": artifact.license,
        }
        try:
            asset_path.relative_to(root)
        except ValueError:
            row["status"] = "outside_workspace"
        else:
            if not asset_path.is_file():
                row["status"] = "missing"
            else:
                try:
                    digest = hashlib.sha256()
                    with asset_path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    row["actual_sha256"] = digest.hexdigest()
                    row["bytes"] = asset_path.stat().st_size
                    row["status"] = "verified" if row["actual_sha256"] == artifact.sha256 else "hash_mismatch"
                except OSError as exc:
                    row["status"] = "unreadable"
                    row["message"] = str(exc)
        report["assets"].append(row)
        if row["status"] != "verified":
            report["issues"].append({
                "code": "pretrained_asset_" + row["status"],
                "task": task,
                "path": artifact.path,
            })
    report["ok"] = not report["issues"]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strictly audit hash-bound training initialization assets.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args(argv)
    report = audit_training_assets(args.config, workspace_root=args.workspace_root, tasks=args.tasks)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
