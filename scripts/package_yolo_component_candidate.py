from __future__ import annotations

"""Package the confirmed YOLO Component checkpoint as an unblessed candidate."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as base  # noqa: E402
from pcbparse.training.artifacts import (  # noqa: E402
    build_candidate_lock_entry,
    finalize_candidate_bundle,
    load_candidate_bundle,
)


CONFIRMATION_KIND = "pcbparse_yolo_component_fixed_frozen_test_confirmation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        default=Path("configs/candidates/yolo_component_y11n_v1_fixed020_test.json"),
    )
    parser.add_argument(
        "--confirmation-status",
        type=Path,
        default=Path(
            "work/confirm-component-yolo11n-v1-fixed020-test-cap128-20260914/"
            "fixed_test_status.json"
        ),
    )
    parser.add_argument(
        "--validation-status",
        type=Path,
        default=Path(
            "work/eval-component-yolo11n-v1-tiled512-val-cap128-20260914/"
            "diagnostic_status.json"
        ),
    )
    parser.add_argument(
        "--calibration-audit",
        type=Path,
        default=Path(
            "artifacts/audits/component_yolo11n_v1_score_calibration_cap128.json"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("work/candidates-runtime-bound/component-yolo11n-v1-fixed020"),
    )
    return parser


def _source_path(workspace: Path, raw: Any, label: str) -> Path:
    path = base._resolve_within(
        workspace,
        base._nonempty_string(raw, label),
        label,
    )
    return path


def _verify_payload(document: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(document)
    claimed = payload.pop(field, None)
    if claimed != base._canonical_sha256(payload):
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")


def _copy_attachment(
    source: Path,
    *,
    root: Path,
    relative: str,
    kind: str,
) -> dict[str, str]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"kind": kind, "path": relative, "sha256": base._sha256(destination)}


def package_candidate(
    *,
    workspace: Path,
    candidate_path: Path,
    confirmation_path: Path,
    validation_path: Path,
    calibration_path: Path,
    destination: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing overwrite: {destination}")
    candidate = base._read_object(candidate_path, "YOLO candidate contract")
    confirmation = base._read_object(confirmation_path, "YOLO test confirmation")
    validation = base._read_object(validation_path, "YOLO validation status")
    calibration = base._read_object(calibration_path, "YOLO calibration audit")
    if (
        confirmation.get("kind") != CONFIRMATION_KIND
        or confirmation.get("status") != "frozen_test_confirmation_complete"
        or confirmation.get("formal_evidence") is not True
        or confirmation.get("gates_passed") is not True
        or confirmation.get("promotion_eligible") is not False
        or confirmation.get("blessing_authorized") is not False
    ):
        raise ValueError("YOLO test confirmation is not final passing evidence")
    _verify_payload(confirmation, "confirmation_payload_sha256", "YOLO test confirmation")
    _verify_payload(validation, "diagnostic_payload_sha256", "YOLO validation status")
    _verify_payload(calibration, "calibration_payload_sha256", "YOLO calibration audit")
    confirmation_contract = base._mapping(
        confirmation.get("candidate_contract"), "confirmation candidate contract"
    )
    if (
        _source_path(workspace, confirmation_contract.get("path"), "candidate contract")
        != candidate_path
        or confirmation_contract.get("sha256") != base._sha256(candidate_path)
    ):
        raise ValueError("test confirmation is not bound to the supplied candidate contract")
    checkpoint_entry = base._mapping(candidate.get("checkpoint"), "candidate checkpoint")
    checkpoint = _source_path(workspace, checkpoint_entry.get("path"), "checkpoint")
    if base._sha256(checkpoint) != checkpoint_entry.get("sha256"):
        raise ValueError("candidate checkpoint digest mismatch")
    category_entry = base._mapping(candidate.get("category_table"), "category table")
    category_path = _source_path(
        workspace, category_entry.get("path"), "category table"
    )
    if base._sha256(category_path) != category_entry.get("sha256"):
        raise ValueError("candidate category table digest mismatch")
    category_payload = base._read_object(category_path, "category table")
    classes = base._sequence(category_payload.get("classes"), "category classes")
    predictions_entry = base._mapping(
        base._mapping(confirmation.get("artifacts"), "confirmation artifacts").get(
            "predictions"
        ),
        "confirmation predictions",
    )
    predictions = _source_path(
        workspace, predictions_entry.get("path"), "confirmation predictions"
    )
    if base._sha256(predictions) != predictions_entry.get("sha256"):
        raise ValueError("confirmation predictions digest mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        model_path = temporary / "model.pt"
        shutil.copy2(checkpoint, model_path)
        predictions_copy = temporary / "predictions.jsonl"
        shutil.copy2(predictions, predictions_copy)
        runtime_contract = {
            "schema_version": 1,
            "kind": "pcbparse_yolo_component_runtime_contract",
            "checkpoint_sha256": base._sha256(model_path),
            "nms_iou": float(
                base._mapping(candidate.get("protocol"), "candidate protocol")[
                    "component_nms_iou"
                ]
            ),
            "imgsz": int(
                base._mapping(candidate.get("protocol"), "candidate protocol")[
                    "model_imgsz"
                ]
            ),
            "source_category_table_sha256": base._sha256(category_path),
            "classes": [dict(base._mapping(item, "category class")) for item in classes],
        }
        runtime_path = temporary / "evidence" / "yolo_runtime_contract.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        base._atomic_write_json(runtime_path, runtime_contract)
        attachments = [
            {
                "kind": "yolo_component_runtime_contract_v1",
                "path": "evidence/yolo_runtime_contract.json",
                "sha256": base._sha256(runtime_path),
            },
            _copy_attachment(
                confirmation_path,
                root=temporary,
                relative="evidence/frozen_test_confirmation.json",
                kind="frozen_test_confirmation",
            ),
            _copy_attachment(
                validation_path,
                root=temporary,
                relative="evidence/frozen_validation_diagnostic.json",
                kind="frozen_validation_diagnostic",
            ),
            _copy_attachment(
                calibration_path,
                root=temporary,
                relative="evidence/validation_calibration.json",
                kind="validation_calibration_diagnostic_only",
            ),
            _copy_attachment(
                candidate_path,
                root=temporary,
                relative="evidence/predeclared_candidate_contract.json",
                kind="predeclared_candidate_contract",
            ),
        ]
        metrics = base._mapping(confirmation.get("metrics"), "confirmation metrics")
        metric_names = ("bbox_map50", "precision50", "recall50", "f1_50")
        evaluation_metrics = {
            name: base._finite(metrics.get(name), f"confirmation {name}")
            for name in metric_names
        }
        gates = []
        policy_gates = []
        for raw in base._sequence(
            confirmation.get("decision_gates"), "confirmation decision gates"
        ):
            gate = base._mapping(raw, "confirmation decision gate")
            policy_gates.append(
                {
                    "metric": str(gate["metric"]),
                    "operator": str(gate["operator"]),
                    "threshold": float(gate["threshold"]),
                }
            )
            gates.append(
                {
                    "metric": str(gate["metric"]),
                    "operator": str(gate["operator"]),
                    "threshold": float(gate["threshold"]),
                    "value": float(gate["value"]),
                    "passed": bool(gate["passed"]),
                }
            )
        selection = base._mapping(
            confirmation.get("selection_boundary"), "confirmation selection boundary"
        )
        bound_inputs = base._mapping(candidate.get("bound_inputs"), "candidate inputs")
        training_config_sha = str(
            base._mapping(bound_inputs.get("training_config"), "training config")[
                "sha256"
            ]
        )
        manifest_path = finalize_candidate_bundle(
            temporary,
            artifact_path=model_path,
            evaluation_predictions_path=predictions_copy,
            task="component",
            version=f"component-yolo-{base._sha256(model_path)[:16]}",
            license_name="Official Dataset300 competition terms; redistribution not granted",
            precision="fp32",
            adapter_entrypoint="pcbparse.perception.yolo_component_adapter:factory",
            inference={
                "device": "auto",
                "score_threshold": float(
                    base._mapping(candidate.get("protocol"), "candidate protocol")[
                        "score_threshold"
                    ]
                ),
                "max_detections": int(
                    base._mapping(candidate.get("protocol"), "candidate protocol")[
                        "max_detections_per_tile"
                    ]
                ),
            },
            training_config_sha256=training_config_sha,
            dataset_config_hash="3381e705c06642cacf8d3e9d82afbf2f725d3e925b67b049a57bd9b858632009",
            source_manifest_sha256="fe379b30be21bb63c90709841700689bcd793ba907f1b40d81a3e8b38cad52fe",
            split_manifest_sha256="c6f95c3f4240f039672483877d2f9fe405e7883a2bfdac60afd8343bec25c7c1",
            gate_policy={
                "task_manifest_sha256": "fe379b30be21bb63c90709841700689bcd793ba907f1b40d81a3e8b38cad52fe",
                "export_min_samples": int(selection["selected_test_samples"]),
                "export_max_abs_error": 0.0,
                "evaluation_gates": policy_gates,
            },
            export_consistency={
                "reference_backend": "ultralytics-8.4.149-native-pytorch",
                "export_backend": "ultralytics-8.4.149-native-pytorch",
                "sample_count": int(selection["selected_test_samples"]),
                "minimum_samples": int(selection["selected_test_samples"]),
                "max_abs_error": 0.0,
                "tolerance": 0.0,
                "input_manifest_sha256": "fe379b30be21bb63c90709841700689bcd793ba907f1b40d81a3e8b38cad52fe",
                "passed": True,
            },
            evaluation={
                "evaluator": "pcbparse.yolo.component.tiled512-cap128",
                "evaluator_version": "1",
                "split": "test",
                "dataset_config_hash": "3381e705c06642cacf8d3e9d82afbf2f725d3e925b67b049a57bd9b858632009",
                "split_manifest_sha256": "c6f95c3f4240f039672483877d2f9fe405e7883a2bfdac60afd8343bec25c7c1",
                "predictions_sha256": base._sha256(predictions_copy),
                "metrics": evaluation_metrics,
                "gates": gates,
            },
            intended_use=(
                "Unblessed Component shadow candidate for runtime and five-module "
                "end-to-end validation under the fixed 512/64 protocol."
            ),
            limitations=[
                "Not production-blessed and not eligible to replace models.lock.json automatically.",
                "The frozen test set contains 30 Dataset300 images and is not the organizer hidden set.",
                "Box, NET_input, Other, PIN, and rare-class behavior require targeted monitoring.",
                "No separate export is used; the Ultralytics native PyTorch checkpoint is loaded directly.",
            ],
            evidence_attachments=attachments,
        )
        load_candidate_bundle(manifest_path)
        build_candidate_lock_entry(manifest_path, lock_directory=temporary)
        os.replace(temporary, destination)
        return destination / "bundle.manifest.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    destination = base._resolve_within(
        workspace,
        arguments.destination,
        "candidate destination",
        kind="directory",
        must_exist=False,
    )
    manifest = package_candidate(
        workspace=workspace,
        candidate_path=base._resolve_within(
            workspace, arguments.candidate_contract, "candidate contract"
        ),
        confirmation_path=base._resolve_within(
            workspace, arguments.confirmation_status, "test confirmation"
        ),
        validation_path=base._resolve_within(
            workspace, arguments.validation_status, "validation status"
        ),
        calibration_path=base._resolve_within(
            workspace, arguments.calibration_audit, "calibration audit"
        ),
        destination=destination,
    )
    print(manifest)
    print(f"bundle_manifest_sha256={base._sha256(manifest)}")
    print(json.dumps({"status": "candidate", "blessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
