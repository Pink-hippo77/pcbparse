"""Dataset discovery, validation, manifests and leakage-safe splits."""

from .dataset import EXPECTED_ROOT_KEYS, audit_dataset
from .hashing import apply_duplicate_groups, sha256_file
from .manifest import read_jsonl, write_json, write_jsonl
from .models import AuditIssue, DatasetAuditResult
from .splitting import (
    DEFAULT_SPLIT_RATIOS,
    assign_grouped_splits,
    validate_grouped_splits,
)

__all__ = [
    "AuditIssue",
    "DatasetAuditResult",
    "DEFAULT_SPLIT_RATIOS",
    "EXPECTED_ROOT_KEYS",
    "apply_duplicate_groups",
    "assign_grouped_splits",
    "audit_dataset",
    "read_jsonl",
    "sha256_file",
    "validate_grouped_splits",
    "write_json",
    "write_jsonl",
]

