"""Shared paths for the archived training scripts.

These scripts used to live in ``E:\\EDA比赛测试`` and contained machine-specific
absolute paths.  Resolve everything from this file so the archive remains usable
after the workspace consolidation.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPO_ROOT / "archive" / "legacy-training"
RUNS_ROOT = ARCHIVE_ROOT / "runs"
RESULTS_ROOT = ARCHIVE_ROOT / "results"
TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "dataset300_v2_torch.json"
