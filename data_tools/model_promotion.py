from __future__ import annotations

"""Assemble or explicitly promote a configured cryptographically bound model set.

This module is deliberately separate from model training.  A staging lock is
always unblessed.  A production lock can only be produced with ``--promote``
after every adapter has loaded its bound artifact and passed the same small,
deterministic protocol fixture.  Existing output files are never replaced.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._bootstrap import ensure_source_path

ensure_source_path()

from pcbparse.io import ComponentView, ImageView  # noqa: E402
from pcbparse.ir.models import (  # noqa: E402
    BBox,
    Component,
    Junction,
    Pin,
    Point,
    TextToken,
    WireSegment,
)
from pcbparse.ir.transforms import AffineTransform, TransformChain  # noqa: E402
from pcbparse.perception.protocols import WireMask  # noqa: E402
from pcbparse.config import load_config  # noqa: E402
from pcbparse.perception.composite_provider import CompositePerceptionProvider  # noqa: E402
from pcbparse.perception.production_provider import (  # noqa: E402
    composite_provider_config_from_pipeline,
    required_model_tasks_for_config,
)
from pcbparse.perception.wire_vectorizer import vectorize_wire_mask  # noqa: E402
from pcbparse.training.artifacts import (  # noqa: E402
    CandidateBundleError,
    CandidateBundleManifest,
    ModelCard,
    build_candidate_lock_entry,
    load_candidate_bundle,
    validate_source_repository_commit,
)
from pcbparse.training.config import ModelTask  # noqa: E402


PROMOTION_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = "1.0"
REQUIRED_TASKS = ("component", "junction", "ocr", "pin", "wire")
DETERMINISTIC_JUNCTION_TASKS = ("component", "ocr", "pin", "wire")
JunctionMode = Literal["learned_model", "deterministic_vectorizer"]
JUNCTION_CLASSES = (
    "endpoint",
    "t_junction",
    "connected_cross",
    "non_connected_cross",
    "bend",
    "noise",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_PLACEHOLDERS = frozenset(
    {"", "dev", "none", "null", "tbd", "todo", "unknown", "unlicensed", "unversioned"}
)
_FIXTURE_SPEC = {
    "schema_version": 1,
    "name": "pcbparse-promotion-fixture-v1",
    "image": {"mode": "RGB", "width": 64, "height": 64, "fill": [255, 255, 255]},
    "component_bbox": [8.0, 8.0, 56.0, 56.0],
    "junction_segments": [
        [[8.0, 32.0], [56.0, 32.0]],
        [[32.0, 8.0], [32.0, 56.0]],
    ],
}
FIXTURE_SHA256 = hashlib.sha256(
    json.dumps(
        _FIXTURE_SPEC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


class ModelPromotionError(RuntimeError):
    """Candidate bundles cannot safely form the requested lock."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JunctionClassCoverage(_StrictModel):
    class_name: Literal[
        "endpoint",
        "t_junction",
        "connected_cross",
        "non_connected_cross",
        "bend",
        "noise",
    ]
    official_real_val_sample_ids: tuple[str, ...] = Field(min_length=1)
    sample_count: int = Field(gt=0)
    reviewer_confirmed: Literal[True]

    @model_validator(mode="after")
    def validate_samples(self) -> "JunctionClassCoverage":
        if self.sample_count != len(self.official_real_val_sample_ids):
            raise ValueError("sample_count differs from official_real_val_sample_ids")
        if len(set(self.official_real_val_sample_ids)) != self.sample_count:
            raise ValueError("official real validation sample IDs must be unique per class")
        if any(not _declared_text(value) for value in self.official_real_val_sample_ids):
            raise ValueError("official real validation sample IDs must be non-placeholder text")
        return self


class JunctionHumanReviewEvidence(_StrictModel):
    """Human attestation required before a junction candidate may be blessed."""

    schema_version: Literal[1]
    kind: Literal["junction_real_validation_human_review"]
    task: Literal["junction"]
    decision: Literal["approve_for_production_promotion"]
    candidate_manifest_sha256: str
    source_manifest_sha256: str
    split_manifest_sha256: str
    split: Literal["val"]
    data_origin: Literal["official_real_validation"]
    synthetic_sample_count: Literal[0]
    reviewer: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    reviewed_at_utc: str = Field(min_length=1)
    class_coverage: tuple[JunctionClassCoverage, ...] = Field(min_length=6, max_length=6)
    reviewed_samples_sha256: str

    @model_validator(mode="after")
    def validate_review(self) -> "JunctionHumanReviewEvidence":
        for field_name in (
            "candidate_manifest_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "reviewed_samples_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not _declared_text(self.reviewer) or not _declared_text(self.review_id):
            raise ValueError("reviewer and review_id must be non-placeholder text")
        try:
            reviewed_at = datetime.fromisoformat(self.reviewed_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at_utc must be an ISO-8601 timestamp") from exc
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at_utc must include a timezone")
        names = tuple(row.class_name for row in self.class_coverage)
        if len(set(names)) != len(JUNCTION_CLASSES) or set(names) != set(JUNCTION_CLASSES):
            raise ValueError("class_coverage must contain each canonical junction class exactly once")
        sample_keys = [
            (row.class_name, sample_id)
            for row in self.class_coverage
            for sample_id in row.official_real_val_sample_ids
        ]
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("reviewed junction class/sample pairs must be unique")
        actual = hashlib.sha256(_canonical_bytes(_coverage_payload(self.class_coverage))).hexdigest()
        if actual != self.reviewed_samples_sha256:
            raise ValueError("reviewed_samples_sha256 does not bind class_coverage")
        return self


class PretrainedExternalReviewEvidence(_StrictModel):
    """Human attestation required before a pretrained-external candidate is blessed.

    An externally pretrained asset carries no dataset300 training lineage, so
    promotion additionally requires an explicit human review of the pinned
    source commit, the upstream asset hashes, and the license.  The evidence
    file must bind the exact candidate manifest bytes it approves.
    """

    schema_version: Literal[1]
    kind: Literal["pretrained_external_human_review"]
    decision: Literal["approve_for_production_promotion"]
    task: ModelTask
    lineage_mode: Literal["pretrained_external"]
    candidate_manifest_sha256: str
    descriptor_sha256: str
    source_repository_commit: str
    source_asset_sha256: tuple[str, ...] = Field(min_length=1)
    source_license: str
    dataset_config_hash: str
    split_manifest_sha256: str
    provenance_verified: Literal[True]
    reviewer_confirmed: Literal[True]
    reviewer: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    reviewed_at_utc: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review(self) -> "PretrainedExternalReviewEvidence":
        for field_name in (
            "candidate_manifest_sha256",
            "descriptor_sha256",
            "dataset_config_hash",
            "split_manifest_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        validate_source_repository_commit(self.source_repository_commit)
        if not _declared_text(self.source_license):
            raise ValueError("source_license must be non-placeholder text")
        seen: set[str] = set()
        for digest in self.source_asset_sha256:
            if not _SHA256.fullmatch(digest):
                raise ValueError(
                    "source_asset_sha256 must contain lowercase SHA-256 digests"
                )
            if digest in seen:
                raise ValueError("source_asset_sha256 must not repeat a digest")
            seen.add(digest)
        if not _declared_text(self.reviewer) or not _declared_text(self.review_id):
            raise ValueError("reviewer and review_id must be non-placeholder text")
        try:
            reviewed_at = datetime.fromisoformat(
                self.reviewed_at_utc.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("reviewed_at_utc must be an ISO-8601 timestamp") from exc
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at_utc must include a timezone")
        return self


@dataclass(frozen=True, slots=True)
class _VerifiedBundle:
    manifest_path: Path
    manifest_sha256: str
    manifest: CandidateBundleManifest
    artifact_path: Path
    card_path: Path
    card: ModelCard
    entry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _BoundPretrainedExternalReview:
    path: Path
    sha256: str
    review: PretrainedExternalReviewEvidence


def reviewed_samples_sha256(class_coverage: Sequence[Mapping[str, Any]]) -> str:
    """Return the digest a human-review evidence file must use.

    The input is validated and canonicalized, so callers cannot accidentally
    hash a different key order or an incomplete subset of the reviewed rows.
    """

    rows = tuple(JunctionClassCoverage.model_validate(value) for value in class_coverage)
    return hashlib.sha256(_canonical_bytes(_coverage_payload(rows))).hexdigest()


def _tasks_for_junction_mode(junction_mode: str) -> tuple[str, ...]:
    if junction_mode == "learned_model":
        return REQUIRED_TASKS
    if junction_mode == "deterministic_vectorizer":
        return DETERMINISTIC_JUNCTION_TASKS
    raise ModelPromotionError(
        "junction_mode must be 'learned_model' or 'deterministic_vectorizer'"
    )


def _verify_deterministic_junction_contract(
    production_config: str | Path | None,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    config_path = (
        repository_root / "configs" / "pipeline.yaml"
        if production_config is None
        else Path(production_config).expanduser()
    ).resolve(strict=False)
    if not config_path.is_file():
        raise ModelPromotionError(
            f"deterministic Junction production config is missing: {config_path}"
        )
    try:
        raw_config = load_config(config_path)
        resolved_config = composite_provider_config_from_pipeline(raw_config)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelPromotionError(
            f"deterministic Junction production config is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    if resolved_config.learned_junctions_enabled:
        raise ModelPromotionError(
            "deterministic_vectorizer promotion requires graph.learned_junctions_enabled=false"
        )
    required_tasks = tuple(sorted(required_model_tasks_for_config(resolved_config)))
    if required_tasks != DETERMINISTIC_JUNCTION_TASKS:
        raise ModelPromotionError(
            "resolved deterministic Junction task plan is not the exact four-model plan"
        )

    class _NoopProvider:
        version = "deterministic-junction-promotion-contract-v1"

    noop = _NoopProvider()
    composite = CompositePerceptionProvider(
        component_provider=noop,
        ocr_provider=noop,
        pin_provider=noop,
        wire_provider=noop,
        junction_provider=None,
        config=resolved_config,
    )
    expected_plan = {
        "contract": "production_model_task_plan_v1",
        "junction_mode": "deterministic_vectorizer",
        "required_model_tasks": list(DETERMINISTIC_JUNCTION_TASKS),
        "loaded_model_tasks": list(DETERMINISTIC_JUNCTION_TASKS),
        "learned_junction_model_required": False,
        "learned_junction_model_loaded": False,
        "learned_junction_model_invoked": False,
    }
    if composite.model_task_plan != expected_plan:
        raise ModelPromotionError(
            "CompositePerceptionProvider does not expose the exact deterministic Junction plan"
        )

    mask = np.zeros((17, 17), dtype=np.uint8)
    mask[3:9, 8] = 1
    mask[8, 3:14] = 1
    vectorized = vectorize_wire_mask(
        mask,
        id_namespace="promotion-deterministic-junction-v1",
    )
    junction_kinds = sorted(item.kind for item in vectorized.junctions)
    if "t_junction" not in junction_kinds or junction_kinds.count("endpoint") < 3:
        raise ModelPromotionError(
            "deterministic wire vectorizer fixture did not produce a T-junction and endpoints"
        )

    source_paths = (
        repository_root / "src" / "pcbparse" / "perception" / "production_provider.py",
        repository_root / "src" / "pcbparse" / "perception" / "composite_provider.py",
        repository_root / "src" / "pcbparse" / "perception" / "wire_vectorizer.py",
        repository_root / "src" / "pcbparse" / "service" / "readiness.py",
    )
    source_manifest = []
    for path in source_paths:
        source_manifest.append(
            {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": _sha256_file(path, "deterministic Junction runtime source"),
            }
        )
    contract: dict[str, Any] = {
        "schema_version": 1,
        "kind": "deterministic_junction_promotion_contract",
        "production_config_path": str(config_path),
        "production_config_sha256": _sha256_file(
            config_path, "deterministic Junction production config"
        ),
        "model_task_plan": expected_plan,
        "vectorizer_fixture": {
            "shape": [17, 17],
            "foreground_pixels": int(mask.sum()),
            "junction_kinds": junction_kinds,
            "polylines": len(vectorized.polylines),
        },
        "runtime_sources": source_manifest,
    }
    contract["contract_payload_sha256"] = hashlib.sha256(
        _canonical_bytes(contract)
    ).hexdigest()
    return contract


def assemble_model_lock(
    bundle_manifests: Sequence[str | Path],
    *,
    output: str | Path,
    promote: bool = False,
    junction_mode: JunctionMode = "learned_model",
    production_config: str | Path | None = None,
    human_review_evidence: str | Path | None = None,
    pretrained_external_review_evidence: Sequence[str | Path] | None = None,
    report_output: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the configured candidates and atomically create a lock plus report.

    ``output`` and the report path must not already exist.  In staging mode no
    adapter factory is invoked and every entry remains ``candidate`` /
    ``blessed=false``.  Promotion additionally loads every model and runs the
    fixed protocol fixture. Learned-Junction promotion requires explicit
    six-class official-real-validation review. Deterministic-Junction mode uses
    an exact four-model plan and a config/source/vectorizer runtime contract.
    ``pretrained_external`` candidates keep the same gates and additionally
    require one binding ``PretrainedExternalReviewEvidence`` per candidate
    before their entry may be blessed.
    """

    output_path = Path(output).expanduser().resolve(strict=False)
    if report_output is None:
        report_path = output_path.with_name(f"{output_path.name}.promotion-report.json")
    else:
        report_path = Path(report_output).expanduser().resolve(strict=False)
    _ensure_distinct_new_outputs(output_path, report_path)
    required_tasks = _tasks_for_junction_mode(junction_mode)
    if len(bundle_manifests) != len(required_tasks):
        raise ModelPromotionError(
            f"exactly {len(required_tasks)} candidate manifests are required for "
            f"junction_mode={junction_mode!r}, got {len(bundle_manifests)}"
        )

    verified = [_verify_bundle(path, output_path.parent) for path in bundle_manifests]
    by_task: dict[str, list[_VerifiedBundle]] = {}
    for item in verified:
        by_task.setdefault(str(item.manifest.task), []).append(item)
    missing = sorted(set(required_tasks) - set(by_task))
    duplicate = sorted(task for task, rows in by_task.items() if len(rows) != 1)
    unexpected = sorted(set(by_task) - set(required_tasks))
    if missing or duplicate or unexpected:
        raise ModelPromotionError(
            "configured candidate set is invalid: "
            f"missing={missing}, duplicate={duplicate}, unexpected={unexpected}"
        )
    ordered = [by_task[task][0] for task in required_tasks]
    _verify_shared_dataset_lineage(ordered)

    deterministic_contract: dict[str, Any] | None = None
    if junction_mode == "deterministic_vectorizer":
        if human_review_evidence is not None:
            raise ModelPromotionError(
                "human_review_evidence cannot be supplied in deterministic_vectorizer mode"
            )
        deterministic_contract = _verify_deterministic_junction_contract(
            production_config
        )

    review: JunctionHumanReviewEvidence | None = None
    review_path: Path | None = None
    review_sha256: str | None = None
    external_reviews: dict[str, _BoundPretrainedExternalReview] = {}
    smoke_results: list[dict[str, Any]] = []
    if not promote and pretrained_external_review_evidence:
        raise ModelPromotionError(
            "pretrained_external_review_evidence is only accepted with promote=True"
        )
    if promote:
        _verify_promotion_declarations(ordered)
        if junction_mode == "learned_model":
            junction = by_task["junction"][0]
            if human_review_evidence is None:
                raise ModelPromotionError(
                    "learned Junction promotion requires human_review_evidence proving all six "
                    "classes on official real validation data"
                )
            review_path = Path(human_review_evidence).expanduser().resolve(strict=False)
            review_sha256 = _sha256_file(review_path, "junction human-review evidence")
            review = _load_human_review_evidence(review_path)
            _bind_human_review(review, junction)
        external_reviews = _verify_pretrained_external_reviews(
            ordered, pretrained_external_review_evidence
        )
        for item in ordered:
            smoke_results.append(_load_and_smoke(item))
    elif human_review_evidence is not None:
        raise ModelPromotionError("human_review_evidence is only accepted with promote=True")

    entries: list[dict[str, Any]] = []
    for item in ordered:
        entry = dict(item.entry)
        entry["path"] = _relative_path(item.artifact_path, output_path.parent, "model artifact")
        entry["model_card_path"] = _relative_path(item.card_path, output_path.parent, "model card")
        entry["candidate_manifest_path"] = _relative_path(
            item.manifest_path, output_path.parent, "candidate manifest"
        )
        entry["model_card_sha256"] = item.manifest.model_card_sha256
        entry["export_consistency"] = item.manifest.export_consistency.model_dump(mode="json")
        if promote:
            entry["status"] = "blessed"
            entry["blessed"] = True
            if item.manifest.task == "junction":
                assert review_path is not None and review_sha256 is not None and review is not None
                entry["human_review_evidence_path"] = _relative_path(
                    review_path, output_path.parent, "junction human-review evidence"
                )
                entry["human_review_evidence_sha256"] = review_sha256
                entry["real_val_class_coverage"] = {
                    row.class_name: row.sample_count for row in review.class_coverage
                }
            elif item.manifest.lineage_mode == "pretrained_external":
                external = external_reviews[str(item.manifest.task)]
                entry["pretrained_external_review_evidence_path"] = _relative_path(
                    external.path,
                    output_path.parent,
                    "pretrained-external human-review evidence",
                )
                entry["pretrained_external_review_evidence_sha256"] = external.sha256
        else:
            # Keep this assignment explicit: staging metadata can never be
            # interpreted as authority, even if an upstream format changes.
            entry["status"] = "candidate"
            entry["blessed"] = False
        entries.append(entry)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lineage_contract": "candidate_bundle_v1",
        "status": "blessed" if promote else "candidate",
        "blessed": bool(promote),
        "created_at_utc": now,
        "junction_mode": junction_mode,
        "required_tasks": list(required_tasks),
        "models": entries,
        "note": (
            f"Explicit {len(required_tasks)}-task production promotion in {junction_mode} "
            "mode; see bound promotion report."
            if promote
            else "Unblessed staging assembly. This lock is intentionally not production-ready."
        ),
    }
    lock_payload = _pretty_bytes(lock)
    lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
    input_rows = []
    for item in ordered:
        input_rows.append(
            {
                "task": item.manifest.task,
                "lineage_mode": item.manifest.lineage_mode,
                "candidate_manifest_path": str(item.manifest_path),
                "candidate_manifest_sha256": item.manifest_sha256,
                "artifact_sha256": item.manifest.artifact_sha256,
                "artifact_bytes": item.manifest.artifact_bytes,
                "model_card_sha256": item.manifest.model_card_sha256,
                "training_config_sha256": item.manifest.training_config_sha256,
                "source_repository_commit": item.manifest.source_repository_commit,
                "source_asset_sha256": list(item.manifest.source_asset_sha256),
                "source_license": item.manifest.source_license,
                "evaluation_predictions_sha256": item.manifest.evaluation.predictions_sha256,
                "export_input_manifest_sha256": (
                    item.manifest.export_consistency.input_manifest_sha256
                ),
            }
        )
    has_pretrained_external = any(
        item.manifest.lineage_mode == "pretrained_external" for item in ordered
    )
    report: dict[str, Any] = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "kind": "model_promotion_report",
        "ok": True,
        "mode": "promote" if promote else "staging",
        "junction_mode": junction_mode,
        "created_at_utc": now,
        "output_lock": str(output_path),
        "output_lock_sha256": lock_sha256,
        "required_tasks": list(required_tasks),
        "inputs": input_rows,
        "checks": {
            "exactly_one_candidate_per_task": True,
            "artifact_and_model_card_bytes_verified": True,
            "evaluation_and_export_gates_revalidated": True,
            "adapter_imports_revalidated": True,
            "shared_dataset_lineage": True,
            "production_adapter_load_and_fixture_smoke": bool(promote),
            "junction_official_real_val_six_class_human_review": bool(
                promote and junction_mode == "learned_model"
            ),
            "deterministic_junction_config_and_vectorizer_contract": bool(
                deterministic_contract
            ),
            "pretrained_external_source_pin_and_descriptor_binding": (
                has_pretrained_external
            ),
            "pretrained_external_human_review": bool(
                promote and external_reviews
            ),
        },
        "fixture": {
            "name": _FIXTURE_SPEC["name"],
            "sha256": FIXTURE_SHA256,
            "smoke_results": smoke_results,
        },
        "junction_human_review": None,
        "pretrained_external_reviews": [
            {
                "task": task,
                "path": str(bound.path),
                "sha256": bound.sha256,
                "review_id": bound.review.review_id,
                "reviewer": bound.review.reviewer,
                "source_repository_commit": bound.review.source_repository_commit,
                "source_asset_sha256": list(bound.review.source_asset_sha256),
                "source_license": bound.review.source_license,
                "provenance_verified": True,
                "reviewer_confirmed": True,
            }
            for task, bound in sorted(external_reviews.items())
        ],
        "deterministic_junction_contract": deterministic_contract,
    }
    if review is not None and review_path is not None and review_sha256 is not None:
        report["junction_human_review"] = {
            "path": str(review_path),
            "sha256": review_sha256,
            "review_id": review.review_id,
            "reviewer": review.reviewer,
            "reviewed_samples_sha256": review.reviewed_samples_sha256,
            "class_coverage": {
                row.class_name: row.sample_count for row in review.class_coverage
            },
        }

    _write_new_pair(
        output_path,
        lock_payload,
        report_path,
        _pretty_bytes(report),
    )
    return {
        "ok": True,
        "mode": report["mode"],
        "output": str(output_path),
        "output_sha256": lock_sha256,
        "report": str(report_path),
        "tasks": list(required_tasks),
        "ready_claimed": bool(promote),
    }


def _verify_bundle(path: str | Path, lock_directory: Path) -> _VerifiedBundle:
    manifest_path = Path(path).expanduser().resolve(strict=False)
    manifest_sha256 = _sha256_file(manifest_path, "candidate manifest")
    try:
        manifest = load_candidate_bundle(manifest_path)
        entry = build_candidate_lock_entry(manifest_path, lock_directory=lock_directory)
    except (CandidateBundleError, ValueError, OSError) as exc:
        raise ModelPromotionError(
            f"candidate verification failed for {manifest_path}: {type(exc).__name__}: {exc}"
        ) from exc
    bundle_root = manifest_path.parent
    artifact_path = _bound_relative_file(
        bundle_root, manifest.artifact_path, "model artifact"
    )
    card_path = _bound_relative_file(bundle_root, manifest.model_card_path, "model card")
    _verify_file(
        artifact_path,
        expected_sha256=manifest.artifact_sha256,
        expected_bytes=manifest.artifact_bytes,
        label="model artifact",
    )
    _verify_file(
        card_path,
        expected_sha256=manifest.model_card_sha256,
        expected_bytes=None,
        label="model card",
    )
    try:
        card = ModelCard.model_validate(_load_strict_json(card_path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelPromotionError(f"model card is invalid: {card_path}: {exc}") from exc
    comparisons = {
        "task": (card.task, manifest.task),
        "lineage_mode": (card.lineage_mode, manifest.lineage_mode),
        "version": (card.version, manifest.version),
        "license": (card.license, manifest.license),
        "training_config_sha256": (
            card.training_config_sha256,
            manifest.training_config_sha256,
        ),
        "dataset_config_hash": (card.dataset_config_hash, manifest.dataset_config_hash),
        "evaluation_predictions_path": (
            card.evaluation_predictions_path,
            manifest.evaluation_predictions_path,
        ),
        "inference": (card.inference, manifest.inference),
        "source_manifest_sha256": (
            card.source_manifest_sha256,
            manifest.source_manifest_sha256,
        ),
        "split_manifest_sha256": (
            card.split_manifest_sha256,
            manifest.split_manifest_sha256,
        ),
        "source_repository_commit": (
            card.source_repository_commit,
            manifest.source_repository_commit,
        ),
        "source_asset_sha256": (
            card.source_asset_sha256,
            manifest.source_asset_sha256,
        ),
        "source_license": (card.source_license, manifest.source_license),
        "artifact_sha256": (card.artifact_sha256, manifest.artifact_sha256),
        "artifact_bytes": (card.artifact_bytes, manifest.artifact_bytes),
        "gate_policy": (card.gate_policy, manifest.gate_policy),
        "evaluation": (card.evaluation, manifest.evaluation),
        "export_consistency": (
            card.export_consistency,
            manifest.export_consistency,
        ),
    }
    mismatches = [name for name, (left, right) in comparisons.items() if left != right]
    if mismatches:
        raise ModelPromotionError(
            f"model card differs from candidate manifest for fields: {', '.join(mismatches)}"
        )
    if manifest.lineage_mode == "pretrained_external":
        # Descriptor bundles are accepted only with their multi-asset evidence:
        # a lone descriptor can never prove its own bytes.
        attached = {attachment.sha256 for attachment in manifest.evidence_attachments}
        if not set(manifest.source_asset_sha256) <= attached:
            raise ModelPromotionError(
                "pretrained-external candidate does not carry its source assets "
                "as hash-bound evidence attachments"
            )
    elif (
        manifest.source_repository_commit is not None
        or manifest.source_license is not None
        or manifest.source_asset_sha256
    ):
        raise ModelPromotionError(
            "dataset300-trained candidate declares pretrained-external source pinning"
        )
    return _VerifiedBundle(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        artifact_path=artifact_path,
        card_path=card_path,
        card=card,
        entry=entry,
    )


def _verify_shared_dataset_lineage(items: Sequence[_VerifiedBundle]) -> None:
    for field_name in (
        "dataset_config_hash",
        "source_manifest_sha256",
        "split_manifest_sha256",
    ):
        values = {getattr(item.manifest, field_name) for item in items}
        if len(values) != 1:
            raise ModelPromotionError(
                f"candidate tasks do not share one {field_name}: {sorted(values)}"
            )


def _verify_promotion_declarations(items: Sequence[_VerifiedBundle]) -> None:
    for item in items:
        manifest = item.manifest
        if not _declared_text(manifest.version) or not _VERSION.fullmatch(manifest.version.strip()):
            raise ModelPromotionError(
                f"task {manifest.task!r} version is missing, placeholder, or incompatible"
            )
        if not _declared_text(manifest.license):
            raise ModelPromotionError(f"task {manifest.task!r} license is a placeholder")
        if not _declared_text(manifest.precision):
            raise ModelPromotionError(f"task {manifest.task!r} precision is a placeholder")


def _load_human_review_evidence(path: Path) -> JunctionHumanReviewEvidence:
    try:
        return JunctionHumanReviewEvidence.model_validate(_load_strict_json(path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelPromotionError(
            f"junction human-review evidence is invalid: {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _bind_human_review(
    review: JunctionHumanReviewEvidence, junction: _VerifiedBundle
) -> None:
    expected = {
        "candidate_manifest_sha256": junction.manifest_sha256,
        "source_manifest_sha256": junction.manifest.source_manifest_sha256,
        "split_manifest_sha256": junction.manifest.split_manifest_sha256,
    }
    for field_name, expected_value in expected.items():
        if getattr(review, field_name) != expected_value:
            raise ModelPromotionError(
                f"junction human-review {field_name} does not bind the candidate lineage"
            )
    if junction.manifest.evaluation.split != "val":
        raise ModelPromotionError(
            "junction candidate evaluation must use the val split before production promotion"
        )


def _load_pretrained_external_review(path: Path) -> PretrainedExternalReviewEvidence:
    try:
        return PretrainedExternalReviewEvidence.model_validate(_load_strict_json(path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ModelPromotionError(
            f"pretrained-external human-review evidence is invalid: {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _bind_pretrained_external_review(
    review: PretrainedExternalReviewEvidence, item: _VerifiedBundle
) -> None:
    manifest = item.manifest
    expected = {
        "candidate_manifest_sha256": item.manifest_sha256,
        "descriptor_sha256": manifest.artifact_sha256,
        "source_repository_commit": manifest.source_repository_commit,
        "source_license": manifest.source_license,
        "dataset_config_hash": manifest.dataset_config_hash,
        "split_manifest_sha256": manifest.split_manifest_sha256,
    }
    for field_name, expected_value in expected.items():
        if getattr(review, field_name) != expected_value:
            raise ModelPromotionError(
                f"pretrained-external human review {field_name} does not bind "
                "the candidate lineage"
            )
    if review.task != manifest.task:
        raise ModelPromotionError(
            "pretrained-external human review task does not bind the candidate"
        )
    reviewed = tuple(review.source_asset_sha256)
    pinned = tuple(manifest.source_asset_sha256)
    if len(reviewed) != len(pinned) or set(reviewed) != set(pinned):
        raise ModelPromotionError(
            "pretrained-external human review source_asset_sha256 does not bind "
            "the candidate assets"
        )
    if manifest.evaluation.split != "val":
        raise ModelPromotionError(
            "pretrained-external candidate evaluation must use the val split "
            "before production promotion"
        )


def _verify_pretrained_external_reviews(
    items: Sequence[_VerifiedBundle],
    provided: Sequence[str | Path] | None,
) -> dict[str, _BoundPretrainedExternalReview]:
    """Bind exactly one human review to every pretrained-external candidate."""

    external = [
        item for item in items if item.manifest.lineage_mode == "pretrained_external"
    ]
    paths = [Path(path).expanduser().resolve(strict=False) for path in (provided or ())]
    if not external and not paths:
        return {}
    if not external:
        raise ModelPromotionError(
            "pretrained_external_review_evidence was supplied but no candidate "
            "declares lineage_mode='pretrained_external'"
        )
    if not paths:
        raise ModelPromotionError(
            "pretrained_external candidates require human review evidence of kind "
            "pretrained_external_human_review before promotion"
        )
    if len(paths) != len(external):
        raise ModelPromotionError(
            f"exactly {len(external)} pretrained-external review evidence file(s) "
            f"are required, got {len(paths)}"
        )
    loaded: list[tuple[Path, str, PretrainedExternalReviewEvidence]] = []
    for path in paths:
        sha256 = _sha256_file(path, "pretrained-external human-review evidence")
        loaded.append((path, sha256, _load_pretrained_external_review(path)))
    if len({row[2].candidate_manifest_sha256 for row in loaded}) != len(loaded):
        raise ModelPromotionError(
            "pretrained-external human review evidence files must bind distinct "
            "candidate manifests"
        )
    bound: dict[str, _BoundPretrainedExternalReview] = {}
    for item in external:
        matches = [
            row
            for row in loaded
            if row[2].candidate_manifest_sha256 == item.manifest_sha256
        ]
        if len(matches) != 1:
            raise ModelPromotionError(
                f"pretrained-external candidate {item.manifest.task!r} needs exactly "
                "one human review evidence file binding its manifest"
            )
        path, sha256, review = matches[0]
        _bind_pretrained_external_review(review, item)
        bound[str(item.manifest.task)] = _BoundPretrainedExternalReview(
            path=path, sha256=sha256, review=review
        )
    return bound


def _load_and_smoke(item: _VerifiedBundle) -> dict[str, Any]:
    entrypoint = item.manifest.adapter_smoke.entrypoint
    module_name, attribute_name = entrypoint.rsplit(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise ModelPromotionError(
            f"adapter import failed during promotion for task {item.manifest.task!r}: {exc}"
        ) from exc
    if not callable(factory):
        raise ModelPromotionError(
            f"adapter factory for task {item.manifest.task!r} is no longer callable"
        )
    live_entry = dict(item.entry)
    live_entry["path"] = str(item.artifact_path)
    try:
        provider = factory(item.artifact_path, live_entry)
    except Exception as exc:
        raise ModelPromotionError(
            f"adapter factory failed to load task {item.manifest.task!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    version = getattr(provider, "version", None)
    if not _declared_text(version):
        raise ModelPromotionError(
            f"loaded provider for task {item.manifest.task!r} has no non-placeholder version"
        )
    try:
        output = _run_fixture(str(item.manifest.task), provider)
    except Exception as exc:
        raise ModelPromotionError(
            f"fixed fixture smoke failed for task {item.manifest.task!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "task": item.manifest.task,
        "adapter": entrypoint,
        "provider_version": version,
        "fixture_sha256": FIXTURE_SHA256,
        "output_count": len(output),
        "passed": True,
    }


def _run_fixture(task: str, provider: Any) -> list[Any]:
    view = ImageView(
        view_id="promotion-fixture-v1",
        image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
        transform=TransformChain([AffineTransform.identity("promotion-fixture-v1")]),
        metadata={"fixture_sha256": FIXTURE_SHA256},
    )
    component_view = ComponentView(
        component_id="promotion-fixture-component",
        component_bbox=BBox(x1=8.0, y1=8.0, x2=56.0, y2=56.0),
        full_roi=view,
    )
    segments = [
        WireSegment(
            internal_id="promotion-fixture-horizontal",
            start=Point(x=8.0, y=32.0),
            end=Point(x=56.0, y=32.0),
        ),
        WireSegment(
            internal_id="promotion-fixture-vertical",
            start=Point(x=32.0, y=8.0),
            end=Point(x=32.0, y=56.0),
        ),
    ]
    expected_type: type[Any]
    method_name: str
    if task == "component":
        method_name, expected_type = "predict", Component
        method = getattr(provider, method_name, None)
        result = method(view, None) if callable(method) else None
    elif task == "pin":
        method_name, expected_type = "predict", Pin
        method = getattr(provider, method_name, None)
        result = method([component_view], None) if callable(method) else None
    elif task == "ocr":
        method_name, expected_type = "recognize", TextToken
        method = getattr(provider, method_name, None)
        result = method([view], None) if callable(method) else None
    elif task == "wire":
        method_name, expected_type = "segment", WireMask
        method = getattr(provider, method_name, None)
        result = method([view], None) if callable(method) else None
    elif task == "junction":
        method_name, expected_type = "classify", Junction
        method = getattr(provider, method_name, None)
        result = method([view], segments, None) if callable(method) else None
    else:  # pragma: no cover - caller has already enforced the task enum.
        raise ValueError(f"unsupported task {task!r}")
    if not isinstance(result, list):
        raise TypeError(f"{task} provider {method_name}() must return a list")
    if any(not isinstance(value, expected_type) for value in result):
        raise TypeError(
            f"{task} provider returned an item outside the {expected_type.__name__} contract"
        )
    if task == "wire":
        for mask in result:
            if mask.view_id != view.view_id or mask.mask.shape != (view.height, view.width):
                raise ValueError("wire fixture output must preserve view ID and raster dimensions")
            if not bool(getattr(mask.mask, "dtype", None)) or not math.isfinite(float(mask.score)):
                raise ValueError("wire fixture output has invalid dtype or score")
    return result


def _coverage_payload(
    rows: Sequence[JunctionClassCoverage],
) -> list[dict[str, Any]]:
    return [
        {
            "class_name": row.class_name,
            "official_real_val_sample_ids": list(row.official_real_val_sample_ids),
            "sample_count": row.sample_count,
            "reviewer_confirmed": True,
        }
        for row in sorted(rows, key=lambda value: value.class_name)
    ]


def _bound_relative_file(root: Path, raw_path: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelPromotionError(f"{label} path must be relative and cannot contain '..'")
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelPromotionError(f"{label} path escapes its candidate bundle") from exc
    if not path.is_file():
        raise ModelPromotionError(f"{label} file is missing: {path}")
    return path


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    label: str,
) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelPromotionError(f"cannot stat {label}: {path}: {exc}") from exc
    if size <= 0:
        raise ModelPromotionError(f"{label} is empty: {path}")
    if expected_bytes is not None and size != expected_bytes:
        raise ModelPromotionError(f"{label} byte size changed: {path}")
    if _sha256_file(path, label) != expected_sha256:
        raise ModelPromotionError(f"{label} SHA-256 changed: {path}")


def _relative_path(path: Path, root: Path, label: str) -> str:
    try:
        value = os.path.relpath(path, root)
    except ValueError as exc:
        raise ModelPromotionError(f"{label} and output lock are on different volumes") from exc
    relative = Path(value)
    if relative.is_absolute():
        raise ModelPromotionError(f"{label} lock path must be relative")
    return relative.as_posix()


def _ensure_distinct_new_outputs(output: Path, report: Path) -> None:
    if output == report:
        raise ModelPromotionError("output lock and promotion report paths must differ")
    existing = [str(path) for path in (output, report) if path.exists()]
    if existing:
        raise ModelPromotionError(
            "refusing to overwrite existing output file(s): " + ", ".join(existing)
        )


def _write_new_pair(
    output: Path,
    output_payload: bytes,
    report: Path,
    report_payload: bytes,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    _ensure_distinct_new_outputs(output, report)
    output_temp = _prepare_temp(output, output_payload)
    report_temp = _prepare_temp(report, report_payload)
    report_created = False
    try:
        # Publish the report first.  A process interruption can therefore
        # never leave a blessed lock without its evidence report.
        os.replace(report_temp, report)
        report_created = True
        if output.exists():
            raise FileExistsError(f"output appeared during promotion: {output}")
        os.replace(output_temp, output)
    except BaseException:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        if report_created and not output.exists():
            report.unlink(missing_ok=True)
        raise


def _prepare_temp(target: Path, payload: bytes) -> Path:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ModelPromotionError(f"{label} file is missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ModelPromotionError(f"cannot hash {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def _declared_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return bool(normalized) and normalized.casefold() not in _PLACEHOLDERS


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the configured candidate bundle set and create either an unblessed "
            "staging lock or an explicitly promoted production lock."
        )
    )
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        type=Path,
        help="Candidate bundle.manifest.json (repeat once per required learned task).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New lock path. Existing files are rejected and never overwritten.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional new report path (default: <output>.promotion-report.json).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Explicitly request blessed entries after live load and fixture smoke.",
    )
    parser.add_argument(
        "--junction-mode",
        choices=("learned_model", "deterministic_vectorizer"),
        default="learned_model",
        help=(
            "Use five learned tasks, or the exact four-model plan with deterministic "
            "wire-vectorizer Junction topology."
        ),
    )
    parser.add_argument(
        "--production-config",
        type=Path,
        help=(
            "Pipeline config bound by deterministic_vectorizer mode "
            "(default: repository configs/pipeline.yaml)."
        ),
    )
    parser.add_argument(
        "--human-review-evidence",
        type=Path,
        help="Strict Junction six-class official-real-val human-review JSON (promotion only).",
    )
    parser.add_argument(
        "--pretrained-external-review-evidence",
        action="append",
        type=Path,
        default=None,
        help=(
            "Strict pretrained-external provenance human-review JSON, one per "
            "pretrained_external candidate bundle (promotion only)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = assemble_model_lock(
            args.bundle,
            output=args.output,
            promote=args.promote,
            junction_mode=args.junction_mode,
            production_config=args.production_config,
            human_review_evidence=args.human_review_evidence,
            pretrained_external_review_evidence=(
                args.pretrained_external_review_evidence
            ),
            report_output=args.report_output,
        )
    except Exception as exc:  # CLI boundary: concise, machine-readable failure.
        failure = {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "output": str(args.output.expanduser().resolve(strict=False)),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXTURE_SHA256",
    "JUNCTION_CLASSES",
    "DETERMINISTIC_JUNCTION_TASKS",
    "JunctionClassCoverage",
    "JunctionHumanReviewEvidence",
    "ModelPromotionError",
    "PretrainedExternalReviewEvidence",
    "REQUIRED_TASKS",
    "assemble_model_lock",
    "build_parser",
    "main",
    "reviewed_samples_sha256",
]
