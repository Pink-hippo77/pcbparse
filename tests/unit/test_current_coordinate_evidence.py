from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pcbparse.config import config_hash, load_config
from pcbparse.evaluation.infer_manifest import (
    _apply_accuracy_profile,
    production_runtime_source_manifest,
)
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    Evidence,
    ImageInfo,
    IRState,
    Junction,
    PhysicalNet,
    Pin,
    Point,
    TextToken,
    WireSegment,
)
from pcbparse.io import load_image, make_component_view
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.output.official_schema import load_official_schema
from pcbparse.output.official_serializer import official_json_bytes, serialize_official
from pcbparse.perception.coordinate_binding import (
    bind_bbox_evidence,
    bind_point_evidence,
    bind_segment_evidence,
)
from pcbparse.perception.production_provider import (
    composite_provider_config_from_pipeline,
    required_model_tasks_for_config,
)


WORKSPACE = Path(__file__).resolve().parents[2]
GENERATOR_PATH = WORKSPACE / "scripts" / "generate_current_coordinate_evidence.py"
AUDIT_PATH = WORKSPACE / "scripts" / "audit_release_hard_gates.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_module("current_coordinate_evidence", GENERATOR_PATH)
AUDIT = _load_module("coordinate_release_gate_audit", AUDIT_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class CurrentCoordinateEvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path | str]:
        runtime_relatives = {
            str(entry["path"])
            for entry in production_runtime_source_manifest(WORKSPACE)["files"]
        }
        for relative in sorted(
            runtime_relatives | set(GENERATOR.RUNTIME_COORDINATE_SOURCES)
        ):
            source = WORKSPACE / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        official_profile = root / "src" / "pcbparse" / "resources" / "official_schema.json"
        official_profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            WORKSPACE / "src" / "pcbparse" / "resources" / "official_schema.json",
            official_profile,
        )
        pipeline = root / "configs" / "pipeline.yaml"
        profile = root / "configs" / "profiles" / "competition.yaml"
        pipeline.parent.mkdir(parents=True, exist_ok=True)
        profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / "configs" / "pipeline.yaml", pipeline)
        shutil.copy2(WORKSPACE / "configs" / "profiles" / "competition.yaml", profile)
        effective = _apply_accuracy_profile(load_config(pipeline), load_config(profile))
        effective_hash = config_hash(effective)
        resolved_config = composite_provider_config_from_pipeline(effective)
        resolved_perception = resolved_config.to_dict()
        required_model_tasks = sorted(required_model_tasks_for_config(resolved_config))
        runtime_source_manifest = production_runtime_source_manifest(root)
        model_task_plan = {
            "contract": "production_model_task_plan_v1",
            "junction_mode": "deterministic_vectorizer",
            "required_model_tasks": required_model_tasks,
            "loaded_model_tasks": required_model_tasks,
            "learned_junction_model_required": False,
            "learned_junction_model_loaded": False,
            "learned_junction_model_invoked": False,
        }

        lock = root / "models" / "models.lock.json"
        models = []
        model_paths: list[Path] = []
        for task in required_model_tasks:
            artifact = root / "models" / "candidates" / task / "model.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"coordinate-fixture-{task}".encode("ascii"))
            model_paths.append(artifact)
            models.append(
                {
                    "task": task,
                    "path": artifact.relative_to(lock.parent).as_posix(),
                    "sha256": _sha(artifact),
                    "status": "blessed",
                    "blessed": True,
                }
            )
        _write_json(lock, {"schema_version": "1.0", "models": models})

        dataset = root / "dataset"
        release_manifest = root / GENERATOR.CANONICAL_RELEASE_MANIFEST
        release_manifest.parent.mkdir(parents=True, exist_ok=True)
        run = root / "work" / "production-e2e-current"
        (run / "circuit_ir").mkdir(parents=True, exist_ok=True)
        (run / "official").mkdir(parents=True, exist_ok=True)
        schema = load_official_schema(official_profile)
        run_id = "a" * 64
        shared_inputs = [pipeline, profile, lock, *model_paths, official_profile]
        release_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        official_paths: list[Path] = []
        membership: dict[str, list[str]] = {}
        for index in range(1, GENERATOR.FORMAL_RELEASE_CASES + 1):
            case_id = f"case-{index:02d}"
            duplicate_group = "group-shared" if index <= 2 else f"group-{index:02d}"
            group_size = 2 if index <= 2 else 1
            membership.setdefault(duplicate_group, []).append(case_id)
            image_path = dataset / case_id / "board.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (100, 80), (index % 256, (index * 3) % 256, 255)
            ).save(image_path)
            image_sha = _sha(image_path)
            original_view = load_image(image_path)
            component = Component(
                internal_id="component:1",
                key="R1",
                category="resistor",
                name="R",
                value="10k",
                bbox=BBox(x1=10, y1=20, x2=30, y2=40),
                evidence=[
                    Evidence(
                        evidence_id="component-local:1",
                        source="detector",
                        view_id=original_view.view_id,
                        metadata={"coordinate_system": "image_view_local_xyxy"},
                    )
                ],
            )
            bind_bbox_evidence(
                component.evidence,
                view=original_view,
                local_bbox=component.bbox,
                original_bbox=component.bbox,
                object_kind="component",
            )
            text = TextToken(
                internal_id="text:1",
                raw_text="R1",
                bbox=BBox(x1=40, y1=10, x2=50, y2=18),
                evidence=[
                    Evidence(
                        evidence_id="ocr-local:1",
                        source="ocr",
                        view_id=original_view.view_id,
                        metadata={"coordinate_system": "image_view_local_xyxy"},
                    )
                ],
            )
            bind_bbox_evidence(
                text.evidence,
                view=original_view,
                local_bbox=text.bbox,
                original_bbox=text.bbox,
                object_kind="text",
            )
            pin_view = make_component_view(
                original_view,
                component,
                padding=max(
                    max(component.bbox.width, component.bbox.height)
                    * resolved_config.component_roi_padding_factor,
                    resolved_config.resolved_pin_component_roi_padding_px,
                ),
                view_id_prefix="component-pin",
            ).full_roi
            pin_global_point = Point(x=10, y=30)
            pin_local_point = pin_view.to_view(pin_global_point)
            pin = Pin(
                internal_id="pin:1",
                component_id="component:1",
                number="1",
                name="A",
                point=pin_global_point,
                edge_ids=["segment:1"],
                evidence=[
                    Evidence(
                        evidence_id="pin-local:1",
                        source="detector",
                        view_id=pin_view.view_id,
                        metadata={"heatmap_point": [1.0, 1.0]},
                    )
                ],
            )
            bind_point_evidence(
                pin.evidence,
                view=pin_view,
                local_point=pin_local_point,
                original_point=pin.point,
                object_kind="pin",
            )
            segment = WireSegment(
                internal_id="segment:1",
                start=Point(x=10, y=30),
                end=Point(x=55, y=30),
                evidence=[
                    Evidence(
                        evidence_id="wire-local:1",
                        source="geometry",
                        view_id=original_view.view_id,
                        metadata={"vectorizer": "fixture"},
                    )
                ],
            )
            bind_segment_evidence(
                segment.evidence,
                view=original_view,
                local_start=segment.start,
                local_end=segment.end,
                original_start=segment.start,
                original_end=segment.end,
            )
            junction = Junction(
                internal_id="junction:1",
                point=Point(x=55, y=30),
                kind="endpoint",
                segment_ids=["segment:1"],
                evidence=[
                    Evidence(
                        evidence_id="junction-local:1",
                        source="geometry",
                        view_id=original_view.view_id,
                        metadata={"vectorizer": "fixture"},
                    )
                ],
            )
            bind_point_evidence(
                junction.evidence,
                view=original_view,
                local_point=junction.point,
                original_point=junction.point,
                object_kind="junction",
            )
            release_rows.append(
                {
                    "sample_id": case_id,
                    "split": GENERATOR.CANONICAL_RELEASE_SPLIT,
                    "duplicate_group": duplicate_group,
                    "duplicate_group_size": group_size,
                    "files": {
                        "image": {
                            "path": f"{case_id}/board.png",
                            "sha256": image_sha,
                        }
                    },
                }
            )

            ir_path = run / "circuit_ir" / f"{case_id}.circuit-ir.json"
            official_path = run / "official" / f"{case_id}.json"
            base_ir = CircuitIR(
                state=IRState.VALIDATED,
                image=ImageInfo(
                    image_id=case_id,
                    path=str(image_path.resolve()),
                    width=100,
                    height=80,
                    sha256=image_sha,
                    actual_format="PNG",
                ),
                transforms=[
                    TransformChain(
                        [AffineTransform.identity("original")]
                    ).to_records()[0]
                ],
                components=[component],
                pins=[pin],
                texts=[text],
                segments=[segment],
                junctions=[junction],
                physical_nets=[
                    PhysicalNet(
                        internal_id="physical:1",
                        pins=["pin:1"],
                        segments=["segment:1"],
                    )
                ],
                electrical_nets=[
                    ElectricalNet(
                        internal_id="electrical:1",
                        pins=["pin:1"],
                        segments=["segment:1"],
                        physical_net_ids=["physical:1"],
                    )
                ],
                metadata={
                    "provider": {
                        "name": "ProductionPerceptionProvider",
                        "provider_kind": "production",
                    },
                    "model_task_plan": model_task_plan,
                    "model_versions": {
                        task: "coordinate-fixture-v1"
                        for task in required_model_tasks
                    },
                    "quarantined_model_versions": {},
                },
            )
            official = serialize_official(base_ir, schema)
            official_path.write_bytes(official_json_bytes(official, schema))
            official_paths.append(official_path)
            persisted = base_ir.model_copy(deep=True)
            persisted.state = IRState.SERIALIZED
            ir_path.write_bytes(
                _canonical_line(persisted.model_dump(mode="json"))
            )
            prediction_rows.append(
                {
                    "case_id": case_id,
                    "duplicate_group": duplicate_group,
                    "image_sha256": image_sha,
                    "ir_path": f"circuit_ir/{case_id}.circuit-ir.json",
                    "ir_sha256": _sha(ir_path),
                    "official_path": f"official/{case_id}.json",
                    "official_sha256": _sha(official_path),
                    "official_output_sha256": _sha(official_path),
                    "provider_kind": "production",
                    "model_lock_sha256": _sha(lock),
                    "config_hash": effective_hash,
                    "inference_run_id": run_id,
                    "sidecar_access": False,
                    "input_paths_read": [
                        str(image_path.resolve()),
                        *(str(path.resolve()) for path in shared_inputs),
                    ],
                    "elapsed_seconds": 0.001,
                    "status": "ok",
                    "error": None,
                }
            )
        release_manifest.write_bytes(
            b"".join(_canonical_line(item) for item in release_rows)
        )
        prediction_manifest = run / "predictions.jsonl"
        prediction_manifest.write_bytes(
            b"".join(_canonical_line(item) for item in prediction_rows)
        )
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        stable_fields = (
            "case_id",
            "duplicate_group",
            "image_sha256",
            "ir_sha256",
            "official_sha256",
            "official_output_sha256",
            "provider_kind",
            "model_lock_sha256",
            "config_hash",
            "sidecar_access",
            "status",
            "error",
        )
        summary = run / "summary.json"
        _write_json(
            summary,
            {
                "schema_version": 2,
                "kind": "production-image-only-inference",
                "inference_run_id": run_id,
                "started_at_utc": now,
                "completed_at_utc": now,
                "generated_at_utc": now,
                "accuracy_mode": "image-only",
                "execution_mode": "production",
                "provider_kind": "production",
                "sidecar_access": False,
                "status": "ok",
                "all_successful": True,
                "split": GENERATOR.CANONICAL_RELEASE_SPLIT,
                "selected_cases": GENERATOR.FORMAL_RELEASE_CASES,
                "selected_duplicate_groups": len(membership),
                "attempted": GENERATOR.FORMAL_RELEASE_CASES,
                "succeeded": GENERATOR.FORMAL_RELEASE_CASES,
                "failed": 0,
                "not_run": 0,
                "elapsed_seconds": 0.1,
                "timing": {
                    "contract": "production_wall_clock_scope_v1",
                    "scope": "function_entry_through_prediction_manifest_fsync",
                    "cold_start_included": True,
                    "config_and_profile_load_included": True,
                    "readiness_included": True,
                    "provider_and_model_load_included": True,
                    "case_loop_included": True,
                    "prediction_manifest_write_included": True,
                    "summary_write_and_cli_teardown_included": False,
                    "initialization_seconds": 0.02,
                    "case_loop_seconds": 0.07,
                    "evidence_finalize_seconds": 0.01,
                    "total_elapsed_seconds": 0.1,
                },
                "config_hash": effective_hash,
                "resolved_perception_config": resolved_perception,
                "resolved_perception_config_sha256": _compact_hash(
                    resolved_perception
                ),
                "model_task_plan": model_task_plan,
                "model_lock_sha256": _sha(lock),
                "runtime_source_manifest": runtime_source_manifest,
                "split_manifest_sha256": _sha(release_manifest),
                "prediction_manifest": prediction_manifest.name,
                "prediction_manifest_sha256": _sha(prediction_manifest),
                "prediction_set_sha256": _compact_hash(
                    [
                        {field: row[field] for field in stable_fields}
                        for row in prediction_rows
                    ]
                ),
                "coverage": {
                    "selected_duplicate_groups": len(membership),
                    "expected_case_ids_sha256": _compact_hash(
                        [row["case_id"] for row in prediction_rows]
                    ),
                    "manifest_case_ids_sha256": _compact_hash(
                        [row["case_id"] for row in prediction_rows]
                    ),
                    "expected_duplicate_group_membership_sha256": _compact_hash(
                        dict(sorted(membership.items()))
                    ),
                    "exact_case_coverage": True,
                    "exact_duplicate_group_coverage": True,
                    "duplicate_case_ids": [],
                },
                "control_inputs": {
                    "dataset_root": str(dataset.resolve()),
                    "split_manifest": str(release_manifest.resolve()),
                    "pipeline_config": str(pipeline.resolve()),
                    "profile": str(profile.resolve()),
                    "official_profile": str(official_profile.resolve()),
                    "models_lock": str(lock.resolve()),
                },
                "selection_control_paths_read": [str(release_manifest.resolve())],
                "readiness": {
                    "status": "ready",
                    "ready": True,
                    "required_learned_tasks": required_model_tasks,
                    "registered_learned_tasks": required_model_tasks,
                    "verified_learned_tasks": required_model_tasks,
                },
            },
        )
        return {
            "summary": summary,
            "release_manifest": release_manifest,
            "lock": lock,
            "pipeline": pipeline,
            "profile": profile,
            "official": official_paths[0],
            "config_hash": effective_hash,
            "resolved_perception": resolved_perception,
        }

    def test_generator_produces_formal_current_report_accepted_by_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            case_manifest = root / "artifacts" / "release" / "coordinate-cases.jsonl"
            report_path = root / "artifacts" / "release" / "coordinate-current.json"

            report = GENERATOR.build_and_write_coordinate_evidence(
                workspace_root=root,
                source_summary=fixture["summary"],
                release_manifest=fixture["release_manifest"],
                current_models_lock=fixture["lock"],
                pipeline_config=fixture["pipeline"],
                profile=fixture["profile"],
                coordinate_case_manifest=case_manifest,
                output=report_path,
            )
            audited = AUDIT._audit_coordinate_report(
                root,
                report_path,
                _sha(fixture["lock"]),
                official_guide_verified=True,
                production_config_sha256=str(fixture["config_hash"]),
                require_formal_binding=True,
            )

        self.assertEqual("ok", report["status"])
        self.assertEqual(50, report["cases_checked"])
        self.assertEqual(250, report["points_checked"])
        self.assertEqual(
            "canonical_dataset300_release50_run_binding_v1",
            report["release_suite_contract"],
        )
        self.assertEqual("release50", report["release_suite_split"])
        self.assertEqual(50, report["release_suite_cases"])
        self.assertEqual(49, report["release_suite_duplicate_groups"])
        self.assertEqual(
            report["source_summary_sha256"],
            report["source_run_binding"]["source_summary_sha256"],
        )
        self.assertEqual(
            report["source_inference_run_id"],
            report["source_run_binding"]["source_inference_run_id"],
        )
        self.assertEqual(
            _compact_hash(report["source_run_binding"]),
            report["source_run_binding_sha256"],
        )
        self.assertEqual(0.0, report["max_source_to_serialized_error_px"])
        self.assertEqual("pass", audited["status"], audited)
        self.assertTrue(audited["current_lock_inference_proven"])

    def test_formal_generation_rejects_noncanonical_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            arbitrary_manifest = root / "data" / "splits" / "copied-release50.jsonl"
            shutil.copy2(fixture["release_manifest"], arbitrary_manifest)

            with self.assertRaisesRegex(
                GENERATOR.CoordinateEvidenceError,
                "requires canonical data/splits/dataset300_release50_v1.jsonl",
            ):
                GENERATOR.recompute_coordinate_evidence(
                    workspace_root=root,
                    source_summary=fixture["summary"],
                    release_manifest=arbitrary_manifest,
                    current_models_lock=fixture["lock"],
                    pipeline_config=fixture["pipeline"],
                    profile=fixture["profile"],
                )

    def test_formal_generation_rejects_partial_canonical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            release_manifest = Path(fixture["release_manifest"])
            first_row = release_manifest.read_bytes().splitlines(keepends=True)[0]
            release_manifest.write_bytes(first_row)
            summary = json.loads(Path(fixture["summary"]).read_text(encoding="utf-8"))
            summary["split_manifest_sha256"] = _sha(release_manifest)
            _write_json(Path(fixture["summary"]), summary)

            with self.assertRaisesRegex(
                GENERATOR.CoordinateEvidenceError,
                "does not contain exactly 50 rows",
            ):
                GENERATOR.recompute_coordinate_evidence(
                    workspace_root=root,
                    source_summary=fixture["summary"],
                    release_manifest=release_manifest,
                    current_models_lock=fixture["lock"],
                    pipeline_config=fixture["pipeline"],
                    profile=fixture["profile"],
                )

    def test_formal_generation_rejects_non_release50_summary_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            summary_path = Path(fixture["summary"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["split"] = "test"
            _write_json(summary_path, summary)

            with self.assertRaisesRegex(
                GENERATOR.CoordinateEvidenceError,
                "source summary split is not 'release50'",
            ):
                GENERATOR.recompute_coordinate_evidence(
                    workspace_root=root,
                    source_summary=summary_path,
                    release_manifest=fixture["release_manifest"],
                    current_models_lock=fixture["lock"],
                    pipeline_config=fixture["pipeline"],
                    profile=fixture["profile"],
                )

    def test_tampered_output_atomically_replaces_old_pass_with_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            case_manifest = root / "artifacts" / "release" / "coordinate-cases.jsonl"
            report_path = root / "artifacts" / "release" / "coordinate-current.json"
            GENERATOR.build_and_write_coordinate_evidence(
                workspace_root=root,
                source_summary=fixture["summary"],
                release_manifest=fixture["release_manifest"],
                current_models_lock=fixture["lock"],
                pipeline_config=fixture["pipeline"],
                profile=fixture["profile"],
                coordinate_case_manifest=case_manifest,
                output=report_path,
            )
            _write_json(fixture["official"], {"tampered": True})

            with self.assertRaises(GENERATOR.CoordinateEvidenceError):
                GENERATOR.build_and_write_coordinate_evidence(
                    workspace_root=root,
                    source_summary=fixture["summary"],
                    release_manifest=fixture["release_manifest"],
                    current_models_lock=fixture["lock"],
                    pipeline_config=fixture["pipeline"],
                    profile=fixture["profile"],
                    coordinate_case_manifest=case_manifest,
                    output=report_path,
                )
            failure = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual("failed", failure["status"])
        self.assertFalse(failure["formal_release_evidence"])
        self.assertFalse(failure["current_lock_inference_executed"])

    def test_gate_rejects_negative_or_partial_claim_even_with_bound_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            case_manifest = root / "artifacts" / "release" / "coordinate-cases.jsonl"
            report_path = root / "artifacts" / "release" / "coordinate-current.json"
            GENERATOR.build_and_write_coordinate_evidence(
                workspace_root=root,
                source_summary=fixture["summary"],
                release_manifest=fixture["release_manifest"],
                current_models_lock=fixture["lock"],
                pipeline_config=fixture["pipeline"],
                profile=fixture["profile"],
                coordinate_case_manifest=case_manifest,
                output=report_path,
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["cases_checked"] = 0
            payload["max_roundtrip_error_px"] = -1.0
            payload["max_source_to_serialized_error_px"] = 1.0
            _write_json(report_path, payload)
            audited = AUDIT._audit_coordinate_report(
                root,
                report_path,
                _sha(fixture["lock"]),
                official_guide_verified=True,
                production_config_sha256=str(fixture["config_hash"]),
                require_formal_binding=True,
            )

        self.assertEqual("fail", audited["status"])
        self.assertIn(
            "coordinate report cases_checked must be positive", audited["problems"]
        )
        self.assertIn(
            "coordinate report max error cannot be negative", audited["problems"]
        )
        self.assertIn(
            "coordinate forward serialization error exceeds 0.5 px",
            audited["problems"],
        )

    def test_historical_summary_cannot_be_rewrapped_after_runtime_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            changed_source = root / "src" / "pcbparse" / "association" / "geometry.py"
            changed_source.write_text(
                changed_source.read_text(encoding="utf-8") + "\n# post-inference drift\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                GENERATOR.CoordinateEvidenceError,
                "inference-time runtime source manifest differs",
            ):
                GENERATOR.recompute_coordinate_evidence(
                    workspace_root=root,
                    source_summary=fixture["summary"],
                    release_manifest=fixture["release_manifest"],
                    current_models_lock=fixture["lock"],
                    pipeline_config=fixture["pipeline"],
                    profile=fixture["profile"],
                )

    def test_local_binding_tamper_is_rejected_against_reconstructed_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            ir_path = (
                Path(fixture["summary"]).parent
                / "circuit_ir"
                / "case-01.circuit-ir.json"
            )
            payload = json.loads(ir_path.read_text(encoding="utf-8"))
            binding = payload["pins"][0]["evidence"][0]["metadata"][
                "local_to_global_binding"
            ]
            binding["local_to_original_matrix"][0][2] += 7.0
            ir = CircuitIR.model_validate(payload)

            with self.assertRaisesRegex(
                GENERATOR.CoordinateEvidenceError,
                "binding matrix differs from reconstructed view",
            ):
                GENERATOR._local_view_roundtrip(
                    ir,
                    fixture["resolved_perception"],
                )

    def test_net_edge_oracle_rejects_serializer_common_mode_direct_y(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            ir_path = (
                Path(fixture["summary"]).parent
                / "circuit_ir"
                / "case-01.circuit-ir.json"
            )
            ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
            records = GENERATOR._independent_net_geometry_records(
                ir,
                ir.electrical_nets[0],
                float(ir.image.height),
            )
            expected_h_minus_y = ((10.0, 50.0), (55.0, 50.0))
            direct_y = ((10.0, 30.0), (55.0, 30.0))

        self.assertIn(expected_h_minus_y, records)
        self.assertNotIn(direct_y, records)

    def test_gate_recomputes_and_rejects_tampered_case_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            case_manifest = root / "artifacts" / "release" / "coordinate-cases.jsonl"
            report_path = root / "artifacts" / "release" / "coordinate-current.json"
            GENERATOR.build_and_write_coordinate_evidence(
                workspace_root=root,
                source_summary=fixture["summary"],
                release_manifest=fixture["release_manifest"],
                current_models_lock=fixture["lock"],
                pipeline_config=fixture["pipeline"],
                profile=fixture["profile"],
                coordinate_case_manifest=case_manifest,
                output=report_path,
            )
            rows = [
                json.loads(line)
                for line in case_manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows[0]["points_checked"] = 1
            case_manifest.write_bytes(
                b"".join(_canonical_line(row) for row in rows)
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["coordinate_case_manifest_sha256"] = _sha(case_manifest)
            _write_json(report_path, payload)
            audited = AUDIT._audit_coordinate_report(
                root,
                report_path,
                _sha(fixture["lock"]),
                official_guide_verified=True,
                production_config_sha256=str(fixture["config_hash"]),
                require_formal_binding=True,
            )

        self.assertEqual("fail", audited["status"])
        self.assertIn(
            "coordinate case manifest differs from recomputation",
            audited["problems"],
        )


if __name__ == "__main__":
    unittest.main()
