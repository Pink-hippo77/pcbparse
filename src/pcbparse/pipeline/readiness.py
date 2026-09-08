"""Deprecated compatibility import for the single production readiness gate."""

from pcbparse.service.readiness import (  # noqa: F401
    LOCK_SCHEMA_VERSION,
    REQUIRED_LEARNED_TASKS,
    REQUIRED_MODEL_TASKS,
    SUPPORTED_PRECISIONS,
    ReadinessGateError,
    readiness_report,
)

__all__ = [
    "LOCK_SCHEMA_VERSION",
    "REQUIRED_LEARNED_TASKS",
    "REQUIRED_MODEL_TASKS",
    "SUPPORTED_PRECISIONS",
    "ReadinessGateError",
    "readiness_report",
]
