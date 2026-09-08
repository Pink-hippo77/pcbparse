from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    CandidateBundleError,
    GatePolicy,
    ModelCard,
    build_candidate_lock_entry,
    build_inference_contract,
    finalize_candidate_bundle,
    load_candidate_bundle,
)
from .config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)


class CandidateRebindError(RuntimeError):
    """Legacy candidate evidence cannot be safely rebound to runtime options."""


def rebind_legacy_candidate_bundle(
    source_manifest: str | Path,
    training_config: str | Path,
    destination_directory: str | Path,
) -> dict[str, Any]:
    """Create a new, non-blessed candidate without changing model predictions.

    Historical schema-v1 bundles did not retain the complete runtime adapter
    contract or a path to the prediction bytes.  This operation verifies the
    old artifact/card/predictions/config lineage, copies the immutable artifact
    and predictions into a *new* directory, and finalizes current metadata.  It
    never changes the source bundle or ``models.lock.json``.
    """

    source_path = Path(source_manifest).expanduser().resolve(strict=False)
    config_path = Path(training_config).expanduser().resolve(strict=False)
    destination = Path(destination_directory).expanduser().resolve(strict=False)
    if not source_path.is_file():
        raise CandidateRebindError(f"source candidate manifest is missing: {source_path}")
    if not config_path.is_file():
        raise CandidateRebindError(f"training config is missing: {config_path}")
    if destination.exists():
        raise CandidateRebindError(
            f"destination already exists; refusing to overwrite it: {destination}"
        )

    source_root = source_path.parent.resolve(strict=True)
    if _contains(source_root, destination) or _contains(destination, source_root):
        raise CandidateRebindError(
            "source and destination candidate directories must not contain one another"
        )

    try:
        manifest = load_candidate_bundle(source_path)
    except CandidateBundleError as exc:
        raise CandidateRebindError(str(exc)) from exc
    artifact = _bound_file(source_root, manifest.artifact_path, "source artifact")
    card_path = _bound_file(source_root, manifest.model_card_path, "source model card")
    _verify_file(
        artifact,
        expected_sha256=manifest.artifact_sha256,
        expected_bytes=manifest.artifact_bytes,
        label="source artifact",
    )
    _verify_file(
        card_path,
        expected_sha256=manifest.model_card_sha256,
        expected_bytes=None,
        label="source model card",
    )
    try:
        card = ModelCard.model_validate(_read_strict_json(card_path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CandidateRebindError(f"source model card is invalid: {exc}") from exc
    _verify_card_manifest_lineage(card, manifest)

    if manifest.evaluation_predictions_path is None:
        predictions = _bound_file(
            source_root, "predictions.jsonl", "legacy evaluation predictions"
        )
    else:
        predictions = _bound_file(
            source_root,
            manifest.evaluation_predictions_path,
            "source evaluation predictions",
        )
    _verify_file(
        predictions,
        expected_sha256=manifest.evaluation.predictions_sha256,
        expected_bytes=None,
        label="source evaluation predictions",
    )
    source_attachments: list[tuple[Any, Path]] = []
    for attachment in manifest.evidence_attachments:
        attachment_path = _bound_file(
            source_root,
            attachment.path,
            f"source evidence attachment {attachment.kind!r}",
        )
        _verify_file(
            attachment_path,
            expected_sha256=attachment.sha256,
            expected_bytes=None,
            label=f"source evidence attachment {attachment.kind!r}",
        )
        source_attachments.append((attachment, attachment_path))

    try:
        config = load_training_config(config_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CandidateRebindError(f"training config is invalid: {exc}") from exc
    semantic_config_sha256 = training_config_sha256(config)
    compatible_config_sha256s = training_config_resume_sha256s(config)
    if manifest.training_config_sha256 not in compatible_config_sha256s:
        raise CandidateRebindError(
            "training config is not resume-compatible with the source candidate"
        )
    if (
        config.dataset.config_hash != manifest.dataset_config_hash
        or config.dataset.source_manifest_sha256 != manifest.source_manifest_sha256
        or config.dataset.split_manifest_sha256 != manifest.split_manifest_sha256
    ):
        raise CandidateRebindError(
            "training config dataset lineage differs from the source candidate"
        )
    task_config = config.tasks[manifest.task]
    expected_policy = GatePolicy(
        task_manifest_sha256=config.dataset.task_manifest_sha256[
            task_config.dataset_task
        ],
        export_min_samples=task_config.export_min_samples,
        export_max_abs_error=task_config.export_max_abs_error,
        evaluation_gates=task_config.evaluation_gates,
    )
    if expected_policy != manifest.gate_policy:
        raise CandidateRebindError(
            "training config gate policy differs from the source candidate"
        )
    inference = build_inference_contract(manifest.task, task_config)
    for label, declared in (
        ("source manifest", manifest.inference),
        ("source model card", card.inference),
    ):
        if declared is not None and declared != inference:
            raise CandidateRebindError(
                f"{label} runtime inference differs from the hashed training config"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.rebind-", dir=str(destination.parent)
        )
    ).resolve(strict=True)
    moved = False
    try:
        if artifact.name == predictions.name:
            raise CandidateRebindError(
                "source artifact and evaluation predictions have the same file name"
            )
        copied_artifact = staging / artifact.name
        copied_predictions = staging / predictions.name
        shutil.copyfile(artifact, copied_artifact)
        shutil.copyfile(predictions, copied_predictions)
        copied_attachment_rows: list[dict[str, Any]] = []
        for attachment, attachment_path in source_attachments:
            copied_attachment = staging / Path(attachment.path)
            copied_attachment.parent.mkdir(parents=True, exist_ok=True)
            if copied_attachment in {copied_artifact, copied_predictions}:
                raise CandidateRebindError(
                    f"evidence attachment {attachment.kind!r} collides with copied bundle bytes"
                )
            shutil.copyfile(attachment_path, copied_attachment)
            copied_attachment_rows.append(attachment.model_dump(mode="json"))
        rebound_manifest = finalize_candidate_bundle(
            staging,
            artifact_path=copied_artifact,
            evaluation_predictions_path=copied_predictions,
            task=manifest.task,
            version=manifest.version,
            license_name=manifest.license,
            precision=manifest.precision,
            adapter_entrypoint=manifest.adapter_smoke.entrypoint,
            inference=inference,
            training_config_sha256=manifest.training_config_sha256,
            dataset_config_hash=manifest.dataset_config_hash,
            source_manifest_sha256=manifest.source_manifest_sha256,
            split_manifest_sha256=manifest.split_manifest_sha256,
            gate_policy=manifest.gate_policy,
            export_consistency=manifest.export_consistency,
            evaluation=manifest.evaluation,
            intended_use=card.intended_use,
            limitations=card.limitations,
            evidence_attachments=copied_attachment_rows,
            out_of_scope_use=card.out_of_scope_use,
        )
        lock_entry = build_candidate_lock_entry(
            rebound_manifest, lock_directory=staging
        )
        rebound_card = staging / "model_card.json"
        audit = {
            "schema_version": 1,
            "kind": "legacy_candidate_runtime_rebind",
            "status": "candidate_evidence_ready",
            "created_at_utc": _utc_now(),
            "task": manifest.task,
            "version": manifest.version,
            "source": {
                "bundle_manifest": str(source_path),
                "bundle_manifest_sha256": _sha256_file(source_path),
                "model_card_sha256": manifest.model_card_sha256,
                "artifact_sha256": manifest.artifact_sha256,
                "evaluation_predictions_sha256": (
                    manifest.evaluation.predictions_sha256
                ),
                "legacy_inference_bound": manifest.inference is not None,
                "legacy_prediction_path_bound": (
                    manifest.evaluation_predictions_path is not None
                ),
            },
            "training_config": {
                "path": str(config_path),
                "file_sha256": _sha256_file(config_path),
                "semantic_sha256": semantic_config_sha256,
                "source_candidate_sha256": manifest.training_config_sha256,
                "exact_semantic_match": (
                    semantic_config_sha256 == manifest.training_config_sha256
                ),
                "resume_compatible": True,
                "resume_compatible_sha256s": list(compatible_config_sha256s),
            },
            "runtime_inference": inference,
            "destination": {
                "directory": str(destination),
                "bundle_manifest_sha256": _sha256_file(rebound_manifest),
                "model_card_sha256": _sha256_file(rebound_card),
                "artifact_sha256": _sha256_file(copied_artifact),
                "evaluation_predictions_sha256": _sha256_file(
                    copied_predictions
                ),
                "candidate_lock_entry_sha256": _canonical_sha256(lock_entry),
            },
            "checks": {
                "source_bytes_verified": True,
                "source_card_manifest_lineage_verified": True,
                "training_config_lineage_verified": True,
                "gate_policy_verified": True,
                "runtime_inference_bound": True,
                "evaluation_prediction_bytes_bound": True,
                "evidence_attachment_bytes_bound": True,
                "candidate_lock_entry_buildable": True,
            },
            "boundaries": {
                "model_retrained": False,
                "predictions_recomputed": False,
                "metrics_recomputed": False,
                "models_lock_modified": False,
                "blessed": False,
            },
        }
        _write_json(staging / "rebind.audit.json", audit)
        os.replace(staging, destination)
        moved = True
    except (CandidateBundleError, OSError, ValueError) as exc:
        if isinstance(exc, CandidateRebindError):
            raise
        raise CandidateRebindError(
            f"candidate rebind failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if not moved and staging.exists():
            shutil.rmtree(staging)

    final_manifest = destination / "bundle.manifest.json"
    final_audit = destination / "rebind.audit.json"
    return {
        "ok": True,
        "status": "candidate_evidence_ready",
        "task": manifest.task,
        "output": str(destination),
        "bundle_manifest": str(final_manifest),
        "bundle_manifest_sha256": _sha256_file(final_manifest),
        "audit": str(final_audit),
        "audit_sha256": _sha256_file(final_audit),
        "blessed": False,
        "models_lock_modified": False,
    }


def _verify_card_manifest_lineage(card: ModelCard, manifest: Any) -> None:
    comparisons = {
        "task": (card.task, manifest.task),
        "version": (card.version, manifest.version),
        "license": (card.license, manifest.license),
        "created_at_utc": (card.created_at_utc, manifest.created_at_utc),
        "training_config_sha256": (
            card.training_config_sha256,
            manifest.training_config_sha256,
        ),
        "dataset_config_hash": (
            card.dataset_config_hash,
            manifest.dataset_config_hash,
        ),
        "source_manifest_sha256": (
            card.source_manifest_sha256,
            manifest.source_manifest_sha256,
        ),
        "split_manifest_sha256": (
            card.split_manifest_sha256,
            manifest.split_manifest_sha256,
        ),
        "artifact_sha256": (card.artifact_sha256, manifest.artifact_sha256),
        "artifact_bytes": (card.artifact_bytes, manifest.artifact_bytes),
        "evaluation_predictions_path": (
            card.evaluation_predictions_path,
            manifest.evaluation_predictions_path,
        ),
        "inference": (card.inference, manifest.inference),
        "evidence_attachments": (
            card.evidence_attachments,
            manifest.evidence_attachments,
        ),
        "gate_policy": (card.gate_policy, manifest.gate_policy),
        "evaluation": (card.evaluation, manifest.evaluation),
        "export_consistency": (
            card.export_consistency,
            manifest.export_consistency,
        ),
    }
    mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
    if mismatches:
        raise CandidateRebindError(
            "source model card differs from manifest for fields: "
            + ", ".join(mismatches)
        )


def _bound_file(root: Path, raw_path: str, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CandidateRebindError(f"{label} path must be bundle-relative")
    resolved = (root / candidate).resolve(strict=False)
    if not _contains(root, resolved):
        raise CandidateRebindError(f"{label} escapes the source bundle")
    if not resolved.is_file():
        raise CandidateRebindError(f"{label} is missing: {resolved}")
    return resolved


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    label: str,
) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise CandidateRebindError(f"{label} is empty")
    if expected_bytes is not None and size != expected_bytes:
        raise CandidateRebindError(f"{label} byte size differs from source metadata")
    if _sha256_file(path) != expected_sha256:
        raise CandidateRebindError(f"{label} SHA-256 differs from source metadata")


def _read_strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CandidateRebindError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise CandidateRebindError(f"non-standard JSON constant {value!r}")


def _write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CandidateRebindError",
    "rebind_legacy_candidate_bundle",
]
