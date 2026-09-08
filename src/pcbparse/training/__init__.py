"""Fail-closed training lineage, preflight, and candidate-bundle utilities.

The core package deliberately has no deep-learning framework dependency.  A
training backend can consume :class:`TrainingConfig`, but no artifact is
eligible for registration until the independent bundle gates pass.
"""

from .artifacts import (
    AdapterImportSmoke,
    CandidateBundleManifest,
    EvidenceAttachment,
    EvaluationEvidence,
    ExportConsistency,
    GatePolicy,
    MetricGateResult,
    ModelCard,
    build_candidate_lock_entry,
    finalize_candidate_bundle,
    load_candidate_bundle,
)
from .config import (
    DATASET_TASK_BY_MODEL_TASK,
    MODEL_TASKS,
    TrainingConfig,
    load_training_config,
    training_config_sha256,
)
from .preflight import preflight_training_data
from .rebind import CandidateRebindError, rebind_legacy_candidate_bundle

__all__ = [
    "AdapterImportSmoke",
    "CandidateBundleManifest",
    "CandidateRebindError",
    "DATASET_TASK_BY_MODEL_TASK",
    "EvaluationEvidence",
    "EvidenceAttachment",
    "ExportConsistency",
    "GatePolicy",
    "MODEL_TASKS",
    "MetricGateResult",
    "ModelCard",
    "TrainingConfig",
    "build_candidate_lock_entry",
    "finalize_candidate_bundle",
    "load_candidate_bundle",
    "load_training_config",
    "preflight_training_data",
    "rebind_legacy_candidate_bundle",
    "training_config_sha256",
]
