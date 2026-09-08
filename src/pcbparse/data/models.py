from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """A deterministic, machine-readable dataset audit finding."""

    code: str
    severity: str
    message: str
    action: str
    sample_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "action": self.action,
        }
        if self.sample_id is not None:
            result["sample_id"] = self.sample_id
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(slots=True)
class DatasetAuditResult:
    """Validated manifest records and the report produced alongside them."""

    dataset_root: Path
    records: list[dict[str, Any]]
    report: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.report.get("ok"))

    @property
    def strict_ok(self) -> bool:
        return bool(self.report.get("strict_ok"))

