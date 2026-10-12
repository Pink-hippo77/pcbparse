from __future__ import annotations

from collections.abc import Iterable

from pcbparse.ir.models import Diagnostic


_SEVERITY_ORDER = {"ERROR": 0, "WARNING": 1, "INFO": 2}


def diagnostic_sort_key(issue: Diagnostic) -> tuple[object, ...]:
    """Return a deterministic ordering key for validation diagnostics."""

    return (
        _SEVERITY_ORDER[issue.severity],
        issue.code,
        tuple(issue.object_ids),
        issue.stage,
        issue.message,
        issue.repair_id or "",
    )


def stable_diagnostics(issues: Iterable[Diagnostic]) -> list[Diagnostic]:
    return sorted(issues, key=diagnostic_sort_key)


def merge_diagnostics(*groups: Iterable[Diagnostic]) -> list[Diagnostic]:
    """Merge diagnostic streams without losing upstream provenance.

    Providers and association/graph stages may already have attached useful
    diagnostics before validation starts.  Revalidation can also regenerate a
    byte-for-byte equivalent validator diagnostic.  Keep the first occurrence
    of each complete diagnostic identity and return the usual stable order.
    """

    unique: dict[tuple[object, ...], Diagnostic] = {}
    for group in groups:
        for issue in group:
            unique.setdefault(diagnostic_sort_key(issue), issue)
    return sorted(unique.values(), key=diagnostic_sort_key)


def error(
    code: str,
    message: str,
    *object_ids: str,
    stage: str = "validation",
) -> Diagnostic:
    return Diagnostic(
        severity="ERROR",
        code=code,
        message=message,
        object_ids=list(object_ids),
        stage=stage,
    )


def warning(
    code: str,
    message: str,
    *object_ids: str,
    stage: str = "validation",
) -> Diagnostic:
    return Diagnostic(
        severity="WARNING",
        code=code,
        message=message,
        object_ids=list(object_ids),
        stage=stage,
    )


def info(
    code: str,
    message: str,
    *object_ids: str,
    stage: str = "validation",
) -> Diagnostic:
    return Diagnostic(
        severity="INFO",
        code=code,
        message=message,
        object_ids=list(object_ids),
        stage=stage,
    )
