from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def ensure_source_path() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def default_dataset_root() -> Path:
    return REPOSITORY_ROOT / "Dataset300" / "Dataset300"

