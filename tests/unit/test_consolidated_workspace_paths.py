from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATHS = (
    PROJECT_ROOT / "archive" / "legacy-training" / "scripts" / "_paths.py"
)


def test_archived_scripts_resolve_from_the_consolidated_project_root() -> None:
    spec = importlib.util.spec_from_file_location("legacy_training_paths", ARCHIVE_PATHS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.REPO_ROOT == PROJECT_ROOT
    assert module.ARCHIVE_ROOT == PROJECT_ROOT / "archive" / "legacy-training"
    assert module.RUNS_ROOT.is_dir()
    assert module.RESULTS_ROOT.is_dir()
    assert module.TRAIN_CONFIG.is_file()


def test_configured_json_input_paths_exist() -> None:
    missing: list[str] = []

    def visit(value: object, *, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, key=child_key)
        elif isinstance(value, list):
            for child_value in value:
                visit(child_value, key=key)
        elif (
            isinstance(value, str)
            and value
            and key in {"model", "path", "split_manifest"}
            and ("/" in value or "\\" in value)
        ):
            candidate = Path(value)
            resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
            if not resolved.exists():
                missing.append(value)

    for config_path in sorted((PROJECT_ROOT / "configs").rglob("*.json")):
        visit(json.loads(config_path.read_text(encoding="utf-8")))

    assert missing == []


def test_active_code_has_no_retired_workspace_roots() -> None:
    retired_roots = ("E:\\EDA比赛测试", "E:\\EDA官方dockers")
    active_roots = (
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "data_tools",
        PROJECT_ROOT / "docker",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "tools",
    )
    text_suffixes = {".json", ".md", ".ps1", ".py", ".toml", ".yaml", ".yml"}
    stale: list[str] = []

    for root in active_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(retired_root in text for retired_root in retired_roots):
                stale.append(str(path.relative_to(PROJECT_ROOT)))

    assert stale == []
