from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = WORKSPACE / "scripts" / "audit_release_hard_gates.py"
SPEC = importlib.util.spec_from_file_location("release_hard_gate_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


SCHEMA = {
    "version": "guide-v1",
    "top_level_keys": ["components", "pins", "nets"],
    "component": {
        "fields": ["type", "Name", "value", "bbox"],
        "bbox_encoding": "xyxy_list",
    },
    "pin": {
        "key_template": "pin_{number}",
        "fields": ["pinname", "point"],
        "point_encoding": "object_xy",
    },
    "net": {
        "key_template": "net_{index}",
        "fields": ["hyperGraph", "edges"],
        "hypergraph_pin_template": "{component_key}.{pin_number}",
        "hypergraph_separator": ",",
        "edge_key_template": "edge_{index}",
        "edge_encoding": "two_point_array",
    },
    "numbers": {
        "coordinate_type": "number",
        "coordinate_system": "original_bottom_left_xy",
        "source_coordinate_system": "image_top_left_xy",
    },
    "ordering": {"deterministic": True, "natural_sort_keys": True},
    "extra_fields": "forbid",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_output() -> dict:
    return {
        "components": {
            "R1": {"type": "resistor", "Name": "R", "value": "10k", "bbox": [1, 2, 3, 4]}
        },
        "pins": {"R1": {"pin_1": {"pinname": "A", "point": {"x": 1, "y": 2}}}},
        "nets": {"net_1": {"hyperGraph": "(R1.1)", "edges": {}}},
    }


def _make_determinism(root: Path, payload: dict | None = None) -> Path:
    destination = root / "artifacts" / "determinism"
    for number in range(1, 4):
        _write_json(destination / f"run{number}" / "case.json", payload or _clean_output())
    return destination


def _make_lock(
    root: Path,
    *,
    blessed: bool,
    include_junction: bool = False,
) -> Path:
    lock = root / "models" / "models.lock.json"
    entries = []
    tasks = (
        AUDIT.REQUIRED_MODEL_TASKS
        if include_junction
        else AUDIT.DETERMINISTIC_JUNCTION_MODEL_TASKS
    )
    for task in tasks:
        artifact = root / "models" / "candidates" / task / "model.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"fixture-{task}".encode("ascii"))
        entries.append(
            {
                "task": task,
                "path": f"candidates/{task}/model.bin",
                "sha256": _sha(artifact),
                "status": "blessed" if blessed else "candidate",
                "blessed": blessed,
            }
        )
    _write_json(lock, {"schema_version": "1.0", "models": entries})
    return lock


def _make_production_config(root: Path, *, learned_junctions_enabled: bool = False) -> tuple[Path, Path, str]:
    pipeline = root / "configs" / "pipeline.yaml"
    profile = root / "configs" / "profiles" / "competition.yaml"
    _write_json(
        pipeline,
        {
            "run": {"profile": "baseline"},
            "graph": {"learned_junctions_enabled": learned_junctions_enabled},
            "qwen": {"enabled": False},
            "validation": {"require_zero_errors": True},
        },
    )
    _write_json(
        profile,
        {
            "name": "competition",
            "provider": "production",
            "qwen_enabled": False,
            "artifact_policy": "error",
            "strict_validation": True,
        },
    )
    binding = AUDIT._production_config_binding(root, pipeline, profile)
    if binding["status"] != "valid":  # pragma: no cover - fixture assertion
        raise AssertionError(binding)
    return pipeline, profile, binding["effective_config_sha256"]


def _make_runtime_source_population(root: Path) -> None:
    """Create the smallest complete source tree accepted by the formal fixture."""

    for relative in AUDIT.PRODUCTION_RUNTIME_SOURCE_DIRECTORIES:
        source = root / relative / "__init__.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            source.write_text(f'"""fixture for {relative}"""\n', encoding="utf-8")
    for relative in AUDIT.PRODUCTION_RUNTIME_SOURCE_FILES:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            continue
        if source.suffix == ".json":
            _write_json(source, {"fixture": relative})
        else:
            source.write_text(f'"""fixture for {relative}"""\n', encoding="utf-8")


def _make_formal_production_run(
    root: Path,
    *,
    name: str,
    lock_hash: str,
    config_hash: str,
    generated_at_utc: str = "2026-01-01T00:00:00Z",
) -> Path:
    run = root / "work" / name
    image = root / "inputs" / "case-1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture-image")
    pipeline = root / "configs" / "pipeline.yaml"
    profile = root / "configs" / "profiles" / "competition.yaml"
    lock = root / "models" / "models.lock.json"
    official_profile = root / "src" / "pcbparse" / "resources" / "official_schema.json"
    _write_json(official_profile, {"fixture": "official-schema"})
    _make_runtime_source_population(root)
    split_manifest = root / "data" / "split.jsonl"
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.write_text('{"case_id":"case-1"}\n', encoding="utf-8")
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    pipeline_payload = json.loads(pipeline.read_text(encoding="utf-8"))
    learned_junctions_enabled = pipeline_payload["graph"][
        "learned_junctions_enabled"
    ]
    model_tasks = sorted(entry["task"] for entry in lock_payload["models"])
    model_task_plan = {
        "contract": "production_model_task_plan_v1",
        "junction_mode": (
            "learned_model_plus_deterministic_vectorizer"
            if learned_junctions_enabled
            else "deterministic_vectorizer"
        ),
        "required_model_tasks": model_tasks,
        "loaded_model_tasks": model_tasks,
        "learned_junction_model_required": learned_junctions_enabled,
        "learned_junction_model_loaded": learned_junctions_enabled,
        "learned_junction_model_invoked": learned_junctions_enabled,
    }
    resolved_perception_config = {
        "learned_junctions_enabled": learned_junctions_enabled
    }
    model_paths = [
        (lock.parent / entry["path"]).resolve() for entry in lock_payload["models"]
    ]
    shared_inputs = [
        str(path.resolve())
        for path in [pipeline, profile, lock, *model_paths, official_profile]
    ]
    official = run / "official" / "case-1.json"
    circuit_ir = run / "circuit_ir" / "case-1.circuit-ir.json"
    _write_json(official, _clean_output())
    _write_json(
        circuit_ir,
        {
            "state": "SERIALIZED",
            "case_id": "case-1",
            "metadata": {
                "model_task_plan": model_task_plan,
                "model_versions": {task: "fixture-v1" for task in model_tasks},
                "quarantined_model_versions": {},
            },
        },
    )
    row = {
        "case_id": "case-1",
        "duplicate_group": "group-1",
        "image_sha256": _sha(image),
        "ir_sha256": _sha(circuit_ir),
        "official_sha256": _sha(official),
        "official_output_sha256": _sha(official),
        "provider_kind": "production",
        "model_lock_sha256": lock_hash,
        "config_hash": config_hash,
        "inference_run_id": "a" * 64,
        "sidecar_access": False,
        "status": "ok",
        "error": None,
        "input_paths_read": [str(image.resolve()), *shared_inputs],
        "ir_path": "circuit_ir/case-1.circuit-ir.json",
        "official_path": "official/case-1.json",
    }
    manifest = run / "predictions.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(AUDIT._canonical_bytes(row))
    summary = run / "summary.json"
    _write_json(
        summary,
        {
            "schema_version": 2,
            "kind": "production-image-only-inference",
            "accuracy_mode": "image-only",
            "execution_mode": "production",
            "provider_kind": "production",
            "sidecar_access": False,
            "status": "ok",
            "all_successful": True,
            "selected_cases": 1,
            "succeeded": 1,
            "failed": 0,
            "not_run": 0,
            "elapsed_seconds": 1.0,
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
                "initialization_seconds": 0.1,
                "case_loop_seconds": 0.8,
                "evidence_finalize_seconds": 0.1,
                "total_elapsed_seconds": 1.0,
            },
            "model_lock_sha256": lock_hash,
            "config_hash": config_hash,
            "resolved_perception_config": resolved_perception_config,
            "resolved_perception_config_sha256": AUDIT._compact_json_sha256(
                resolved_perception_config
            ),
            "model_task_plan": model_task_plan,
            "runtime_source_manifest": (
                AUDIT._independent_production_runtime_source_manifest(root)
            ),
            "generated_at_utc": generated_at_utc,
            "inference_run_id": "a" * 64,
            "prediction_manifest": "predictions.jsonl",
            "prediction_manifest_sha256": _sha(manifest),
            "prediction_set_sha256": AUDIT._stable_prediction_set_sha256([row]),
            "split_manifest_sha256": _sha(split_manifest),
            "control_inputs": {
                "dataset_root": str(image.parent.resolve()),
                "split_manifest": str(split_manifest.resolve()),
                "pipeline_config": str(pipeline.resolve()),
                "profile": str(profile.resolve()),
                "official_profile": str(official_profile.resolve()),
                "models_lock": str(lock.resolve()),
            },
            "selection_control_paths_read": [str(split_manifest.resolve())],
            "readiness": {
                "status": "ready",
                "ready": True,
                "required_learned_tasks": model_tasks,
                "registered_learned_tasks": model_tasks,
                "verified_learned_tasks": model_tasks,
            },
            "coverage": {
                "exact_case_coverage": True,
                "exact_duplicate_group_coverage": True,
                "manifest_case_ids_sha256": AUDIT._compact_json_sha256(["case-1"]),
                "expected_case_ids_sha256": AUDIT._compact_json_sha256(["case-1"]),
                "expected_duplicate_group_membership_sha256": AUDIT._compact_json_sha256(
                    {"group-1": ["case-1"]}
                ),
            },
        },
    )
    return summary


def _make_formal_determinism(
    root: Path,
    *,
    name: str,
    summary: Path,
    lock_hash: str,
    config_hash: str,
    generated_at_utc: str = "2026-01-02T00:00:00Z",
) -> Path:
    destination = root / "artifacts" / "determinism" / name
    source_output = summary.parent / "official" / "case-1.json"
    runs = []
    for number in range(1, 4):
        source_run = root / "work" / f"{name}-source-run{number}"
        shutil.copytree(summary.parent, source_run)
        source_summary = source_run / "summary.json"
        source_summary_payload = json.loads(source_summary.read_text(encoding="utf-8"))
        source_summary_payload["inference_run_id"] = str(number) * 64
        source_manifest = source_run / "predictions.jsonl"
        source_rows = [
            json.loads(line)
            for line in source_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in source_rows:
            row["inference_run_id"] = str(number) * 64
        source_manifest.write_bytes(
            b"".join(AUDIT._canonical_bytes(row) for row in source_rows)
        )
        source_summary_payload["prediction_manifest_sha256"] = _sha(source_manifest)
        _write_json(source_summary, source_summary_payload)
        output = destination / f"run{number}" / "case-1.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source_output.read_bytes())
        files, tree_hash = AUDIT._tree_manifest(output.parent)
        runs.append(
            {
                "run": f"run{number}",
                "environment_fingerprint_sha256": "e" * 64,
                "config_sha256": config_hash,
                "model_lock_sha256": lock_hash,
                "output_tree_sha256": tree_hash,
                "files": files,
                "inference_run_id": str(number) * 64,
                "completed_at_utc": "2026-01-01T00:00:00Z",
                "source_summary_path": source_summary.relative_to(root).as_posix(),
                "source_summary_sha256": _sha(source_summary),
            }
        )
    _write_json(
        destination / "environment.json",
        {
            "schema_version": 3,
            "kind": "same-environment-determinism",
            "status": "ok",
            "formal_release_evidence": True,
            "generated_at_utc": generated_at_utc,
            "evidence_scope": "current_lock_production_image_only_inference",
            "current_lock_inference_executed": True,
            "models_loaded": True,
            "model_lock_sha256": lock_hash,
            "config_sha256": config_hash,
            "runs": runs,
        },
    )
    return destination


class ReleaseHardGateAuditTests(unittest.TestCase):
    def test_coordinate_performance_and_determinism_require_one_shared_run(self) -> None:
        source = {
            "path": "work/run1/summary.json",
            "sha256": "a" * 64,
            "inference_run_id": "b" * 64,
            "model_lock_sha256": "c" * 64,
            "config_hash": "d" * 64,
            "selected_cases": 50,
            "prediction_manifest_sha256": "e" * 64,
            "prediction_set_sha256": "f" * 64,
        }
        coordinate = {
            "status": "pass",
            "current_lock_inference_proven": True,
            "source_summary_binding": source,
            "problems": [],
            "release_binding_problems": [],
        }
        performance = {"status": "pass", "selected_newest_run": dict(source)}
        determinism = {
            "status": "pass",
            "formal_summary_bindings": [
                dict(source),
                {**source, "path": "work/run2/summary.json", "sha256": "1" * 64},
                {**source, "path": "work/run3/summary.json", "sha256": "2" * 64},
            ],
        }

        AUDIT._bind_coordinate_to_release_runs(
            coordinate,
            performance,
            determinism,
        )
        self.assertEqual("pass", coordinate["release_run_cross_binding"]["status"])

        mismatched = {
            **coordinate,
            "status": "pass",
            "current_lock_inference_proven": True,
            "problems": [],
            "release_binding_problems": [],
        }
        AUDIT._bind_coordinate_to_release_runs(
            mismatched,
            {
                "status": "pass",
                "selected_newest_run": {
                    **source,
                    "path": "work/unrelated/summary.json",
                    "sha256": "9" * 64,
                },
            },
            determinism,
        )
        self.assertEqual("fail", mismatched["status"])
        self.assertIn(
            "coordinate evidence is not bound to the selected 50-case performance summary",
            mismatched["problems"],
        )

    def test_guide_shape_counter_finds_extra_fields_and_dangling_pin(self) -> None:
        output = _clean_output()
        output["debug"] = True
        output["components"]["R1"]["confidence"] = 0.9
        output["nets"]["net_1"]["hyperGraph"] = "(R1.9)"

        result = AUDIT.analyze_guide_output(output, SCHEMA)

        self.assertEqual(2, result["extra_fields"])
        self.assertEqual(1, result["dangling_references"])
        self.assertFalse(result["clean"])

    def test_three_equal_hashes_without_environment_provenance_stay_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            determinism = _make_determinism(root)

            result = AUDIT._audit_determinism(root, determinism, None, "a" * 64)

        self.assertEqual("unverified", result["status"])
        self.assertTrue(result["identical_json_hashes"])
        self.assertFalse(result["same_environment_proven"])

    def test_hash_bound_same_environment_manifest_closes_determinism_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            determinism = _make_determinism(root)
            lock_hash = "a" * 64
            first = AUDIT._audit_determinism(root, determinism, None, lock_hash)
            environment = determinism / "environment.json"
            _write_json(
                environment,
                {
                    "kind": "same-environment-determinism",
                    "evidence_scope": "current_lock_production_image_only_inference",
                    "current_lock_inference_executed": True,
                    "models_loaded": True,
                    "runs": [
                        {
                            "run": item["run"],
                            "environment_fingerprint_sha256": "b" * 64,
                            "config_sha256": "c" * 64,
                            "model_lock_sha256": lock_hash,
                            "output_tree_sha256": item["tree_sha256"],
                        }
                        for item in first["runs"]
                    ],
                },
            )

            result = AUDIT._audit_determinism(root, determinism, environment, lock_hash)

        self.assertEqual("pass", result["status"])
        self.assertTrue(result["same_environment_proven"])
        self.assertTrue(result["current_lock_inference_proven"])

    def test_frozen_ir_serializer_replay_does_not_close_inference_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            determinism = _make_determinism(root)
            lock_hash = "a" * 64
            first = AUDIT._audit_determinism(root, determinism, None, lock_hash)
            environment = determinism / "environment.json"
            _write_json(
                environment,
                {
                    "kind": "same-environment-determinism",
                    "evidence_scope": "frozen_ir_current_serializer_replay",
                    "current_lock_inference_executed": False,
                    "models_loaded": False,
                    "source_inference_model_lock_sha256": "d" * 64,
                    "runs": [
                        {
                            "run": item["run"],
                            "environment_fingerprint_sha256": "b" * 64,
                            "config_sha256": "c" * 64,
                            "model_lock_sha256": lock_hash,
                            "output_tree_sha256": item["tree_sha256"],
                        }
                        for item in first["runs"]
                    ],
                },
            )

            result = AUDIT._audit_determinism(root, determinism, environment, lock_hash)

        self.assertEqual("unverified", result["status"])
        self.assertTrue(result["same_environment_proven"])
        self.assertFalse(result["current_lock_inference_proven"])
        self.assertEqual("pass", result["serializer_replay_subcheck"])

    def test_locked_bytes_do_not_make_candidate_models_release_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=False)

            result = AUDIT._audit_model_lock(root, lock)

        self.assertEqual("fail", result["status"])
        self.assertTrue(result["artifact_hashes_all_match"])
        self.assertFalse(result["all_entries_blessed"])

    def test_performance_requires_exact_50_and_ten_percent_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "summary.json"
            lock_hash = "a" * 64
            base = {
                "kind": "production-image-only-inference",
                "status": "ok",
                "all_successful": True,
                "selected_cases": 50,
                "succeeded": 50,
                "failed": 0,
                "elapsed_seconds": 6480.0,
                "model_lock_sha256": lock_hash,
                "prediction_set_sha256": "b" * 64,
            }
            _write_json(path, base)
            passing, _ = AUDIT._audit_performance(root, [path], lock_hash)

            _write_json(path, {**base, "elapsed_seconds": 6480.001})
            no_margin, _ = AUDIT._audit_performance(root, [path], lock_hash)

            _write_json(
                path,
                {**base, "selected_cases": 1, "succeeded": 1, "elapsed_seconds": 1.0},
            )
            one_case, _ = AUDIT._audit_performance(root, [path], lock_hash)

        self.assertEqual("pass", passing["status"])
        self.assertEqual("fail", no_margin["status"])
        self.assertEqual("unverified", one_case["status"])

    def test_official_guide_name_case_conflict_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            guide = root / "guide.pdf"
            guide.write_bytes(b"reviewed-guide-fixture")
            old_hash = AUDIT.OFFICIAL_GUIDE_SHA256
            AUDIT.OFFICIAL_GUIDE_SHA256 = _sha(guide)
            try:
                result = AUDIT._audit_official_contract(
                    root,
                    guide,
                    None,
                    None,
                    None,
                    None,
                    "a" * 64,
                    1,
                    {
                        "status": "pass",
                        "profile_component_fields": ["type", "name", "value", "bbox"],
                    },
                )
            finally:
                AUDIT.OFFICIAL_GUIDE_SHA256 = old_hash

        self.assertTrue(result["official_guide_authority_available"])
        self.assertEqual("fail", result["official_json_zero_extra_fields"]["status"])
        self.assertTrue(
            result["official_json_zero_extra_fields"]["guide_name_case_conflict"]
        )
        self.assertEqual(
            "unverified", result["official_json_zero_dangling_references"]["status"]
        )

    def test_official_evaluator_case_count_is_exact_and_rejects_bool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "official-sample.json"
            evaluator = root / "official-evaluator.exe"
            report = root / "official-report.json"
            sample.write_bytes(b"sample")
            evaluator.write_bytes(b"evaluator")
            base = {
                "kind": "official-evaluator-release-report",
                "official_sample_sha256": _sha(sample),
                "official_evaluator_sha256": _sha(evaluator),
                "evaluated_output_tree_sha256": "a" * 64,
                "exit_code": 0,
                "extra_fields": 0,
                "dangling_references": 0,
            }
            for declared in (1, 51, True):
                with self.subTest(cases_evaluated=declared):
                    _write_json(report, {**base, "cases_evaluated": declared})
                    result = AUDIT._audit_official_contract(
                        root,
                        None,
                        None,
                        sample,
                        evaluator,
                        report,
                        "a" * 64,
                        50,
                        {"status": "pass", "profile_component_fields": ["type", "Name", "value", "bbox"]},
                    )
                    self.assertFalse(result["official_evaluator_report_valid"])
                    self.assertIn("cases_evaluated", result["report_problems"])

            _write_json(report, {**base, "cases_evaluated": 50})
            exact = AUDIT._audit_official_contract(
                root,
                None,
                None,
                sample,
                evaluator,
                report,
                "a" * 64,
                50,
                {"status": "pass", "profile_component_fields": ["type", "Name", "value", "bbox"]},
            )

        self.assertTrue(exact["official_evaluator_report_valid"])

    def test_bottom_left_contract_rejects_direct_y_down_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serializer = root / "src" / "pcbparse" / "output" / "official_serializer.py"
            derive = root / "src" / "pcbparse" / "data" / "derive.py"
            serializer.parent.mkdir(parents=True, exist_ok=True)
            derive.parent.mkdir(parents=True, exist_ok=True)
            serializer.write_text(
                'return {"y": round_half_away_from_zero(point.y)}\n', encoding="utf-8"
            )
            derive.write_text('{"image_y_axis": "down"}\n', encoding="utf-8")

            result = AUDIT._audit_coordinate_report(
                root,
                None,
                "a" * 64,
                official_guide_verified=True,
            )

        self.assertEqual("fail", result["status"])
        self.assertTrue(result["static_origin_mismatch"])

    def test_frozen_ir_coordinate_replay_is_valid_but_not_current_lock_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serializer = root / "src" / "pcbparse" / "output" / "official_serializer.py"
            derive = root / "src" / "pcbparse" / "data" / "derive.py"
            serializer.parent.mkdir(parents=True, exist_ok=True)
            derive.parent.mkdir(parents=True, exist_ok=True)
            serializer.write_text('return {"y": image_height - point.y}\n', encoding="utf-8")
            derive.write_text('{"image_y_axis": "down"}\n', encoding="utf-8")
            report = root / "coordinate.json"
            _write_json(
                report,
                {
                    "kind": "coordinate-roundtrip-release-report",
                    "evidence_scope": "frozen_ir_current_serializer_replay",
                    "model_lock_sha256": "a" * 64,
                    "source_inference_model_lock_sha256": "b" * 64,
                    "current_lock_inference_executed": False,
                    "models_loaded": False,
                    "strict_image_only": True,
                    "official_coordinate_origin": "bottom_left",
                    "official_y_axis": "up",
                    "origin_conversion_verified": True,
                    "config_sha256": "c" * 64,
                    "source_prediction_set_sha256": "d" * 64,
                    "cases_checked": 1,
                    "points_checked": 10,
                    "max_source_to_serialized_error_px": 0.0,
                    "max_roundtrip_error_px": 0.0,
                },
            )

            result = AUDIT._audit_coordinate_report(
                root,
                report,
                "a" * 64,
                official_guide_verified=True,
            )

        self.assertEqual("unverified", result["status"])
        self.assertEqual("pass", result["serializer_replay_subcheck"])
        self.assertFalse(result["current_lock_inference_proven"])

    def test_current_lock_coordinate_e2e_closes_coordinate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serializer = root / "src" / "pcbparse" / "output" / "official_serializer.py"
            derive = root / "src" / "pcbparse" / "data" / "derive.py"
            serializer.parent.mkdir(parents=True, exist_ok=True)
            derive.parent.mkdir(parents=True, exist_ok=True)
            serializer.write_text('return {"y": image_height - point.y}\n', encoding="utf-8")
            derive.write_text('{"image_y_axis": "down"}\n', encoding="utf-8")
            report = root / "coordinate.json"
            _write_json(
                report,
                {
                    "kind": "coordinate-roundtrip-release-report",
                    "evidence_scope": "current_lock_production_image_only_inference",
                    "model_lock_sha256": "a" * 64,
                    "current_lock_inference_executed": True,
                    "models_loaded": True,
                    "strict_image_only": True,
                    "official_coordinate_origin": "bottom_left",
                    "official_y_axis": "up",
                    "origin_conversion_verified": True,
                    "config_sha256": "c" * 64,
                    "source_prediction_set_sha256": "d" * 64,
                    "cases_checked": 1,
                    "points_checked": 10,
                    "max_source_to_serialized_error_px": 0.5,
                    "max_roundtrip_error_px": 0.5,
                },
            )

            result = AUDIT._audit_coordinate_report(
                root,
                report,
                "a" * 64,
                official_guide_verified=True,
            )

        self.assertEqual("pass", result["status"])
        self.assertTrue(result["current_lock_inference_proven"])

    def test_traceability_requires_error_id_stage_and_diagnostics(self) -> None:
        row = {
            "case_id": "1",
            "image_sha256": "a" * 64,
            "config_hash": "b" * 64,
            "model_lock_sha256": "c" * 64,
            "error": {"type": "PipelineError", "message": "failed"},
        }
        missing, problems = AUDIT._is_traceable_error(row)
        complete_row = {
            **row,
            "error": {
                "type": "PipelineError",
                "message": "failed",
                "stage": "validation",
                "diagnostic_count": 1,
                "diagnostic_codes": {"PIN.MISSING_NUMBER": 1},
                "diagnostics": [
                    {
                        "severity": "ERROR",
                        "code": "PIN.MISSING_NUMBER",
                        "message": "missing",
                        "object_ids": ["p1"],
                        "stage": "schema",
                    }
                ],
            },
        }
        complete_row["error"]["error_id"] = AUDIT._compact_json_sha256(
            {
                "case_id": complete_row["case_id"],
                "image_sha256": complete_row["image_sha256"],
                "config_hash": complete_row["config_hash"],
                "model_lock_sha256": complete_row["model_lock_sha256"],
                "error": complete_row["error"],
            }
        )
        complete, complete_problems = AUDIT._is_traceable_error(complete_row)

        self.assertFalse(missing)
        self.assertIn("error.error_id", problems)
        self.assertTrue(complete)
        self.assertEqual([], complete_problems)

    def test_automatic_selection_rejects_legacy_root_and_frozen_ir_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            _make_determinism(root)
            frozen = root / "artifacts" / "determinism" / "frozen-replay"
            for number in range(1, 4):
                _write_json(frozen / f"run{number}" / "case.json", _clean_output())
            first = AUDIT._audit_determinism(root, frozen, None, _sha(lock))
            _write_json(
                frozen / "environment.json",
                {
                    "kind": "same-environment-determinism",
                    "evidence_scope": "frozen_ir_current_serializer_replay",
                    "current_lock_inference_executed": False,
                    "models_loaded": False,
                    "model_lock_sha256": _sha(lock),
                    "config_sha256": config_hash,
                    "runs": [
                        {
                            "run": item["run"],
                            "environment_fingerprint_sha256": "e" * 64,
                            "config_sha256": config_hash,
                            "model_lock_sha256": _sha(lock),
                            "output_tree_sha256": item["tree_sha256"],
                        }
                        for item in first["runs"]
                    ],
                },
            )

            report = AUDIT.build_audit(workspace_root=root)

        selection = report["evidence_selection"]["determinism"]
        self.assertIsNone(selection["selected_path"])
        self.assertEqual("unverified", report["gates"]["same_environment_three_run_json_hash"]["status"])
        self.assertEqual(0, report["internal_guide_v1_subcheck"]["outputs_checked"])
        by_path = {item["path"]: item for item in selection["candidates"]}
        self.assertIn("artifacts/determinism", by_path)
        self.assertTrue(
            any("manifest is missing" in reason for reason in by_path["artifacts/determinism"]["rejection_reasons"])
        )
        frozen_record = by_path["artifacts/determinism/frozen-replay"]
        self.assertEqual("frozen_ir_current_serializer_replay", frozen_record["evidence_scope"])
        self.assertFalse(frozen_record["eligible"])

    def test_automatic_selection_uses_only_formally_bound_current_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-formal",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            valid = _make_formal_determinism(
                root,
                name="formal-current",
                summary=summary,
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            stale = _make_formal_determinism(
                root,
                name="newer-wrong-lock",
                summary=summary,
                lock_hash="f" * 64,
                config_hash=config_hash,
                generated_at_utc="2026-01-03T00:00:00Z",
            )
            # Bind the stale candidate's copied manifest consistently to its
            # own wrong-lock declaration so rejection is specifically against
            # the current lock, not accidental file corruption.
            self.assertTrue((stale / "environment.json").is_file())

            report = AUDIT.build_audit(workspace_root=root)

        selection = report["evidence_selection"]["determinism"]
        self.assertEqual(valid.relative_to(root).as_posix(), selection["selected_path"])
        self.assertEqual("pass", report["gates"]["same_environment_three_run_json_hash"]["status"])
        records = {item["path"]: item for item in selection["candidates"]}
        self.assertFalse(records[stale.relative_to(root).as_posix()]["eligible"])
        self.assertTrue(
            any(
                "current models.lock" in reason
                for reason in records[stale.relative_to(root).as_posix()]["rejection_reasons"]
            )
        )

    def test_formal_summary_binding_detects_materialized_output_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-formal",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            _write_json(summary.parent / "official" / "case-1.json", {"tampered": True})

            result = AUDIT._audit_production_summary_binding(
                root,
                summary,
                _sha(lock),
                config_hash,
                require_success=True,
                require_embedded_time=True,
            )

        self.assertFalse(result["valid"])
        self.assertIn("prediction row 0 official_sha256 does not match bytes", result["problems"])

    def test_formal_summary_binding_rejects_timing_without_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-warm-only-timing",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["timing"]["cold_start_included"] = False
            _write_json(summary, payload)

            result = AUDIT._audit_production_summary_binding(
                root,
                summary,
                _sha(lock),
                config_hash,
                require_success=True,
                require_embedded_time=True,
            )

        self.assertFalse(result["valid"])
        self.assertIn("timing_cold_start", result["problems"])

    def test_formal_summary_binding_rejects_runtime_source_changed_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-stale-source",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            changed = root / "src" / "pcbparse" / "pipeline" / "__init__.py"
            changed.write_text('"""changed after inference"""\n', encoding="utf-8")

            result = AUDIT._audit_production_summary_binding(
                root,
                summary,
                _sha(lock),
                config_hash,
                require_success=True,
                require_embedded_time=True,
            )

        self.assertFalse(result["valid"])
        self.assertIn("runtime_source_current_bytes", result["problems"])

    def test_formal_summary_rejects_resolved_output_alias_and_reused_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-alias",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            manifest = summary.parent / "predictions.jsonl"
            first = json.loads(manifest.read_text(encoding="utf-8"))
            second_ir = summary.parent / "circuit_ir" / "case-2.circuit-ir.json"
            _write_json(second_ir, {"state": "SERIALIZED", "case_id": "case-2"})
            second = {
                **first,
                "case_id": "case-2",
                "duplicate_group": "group-2",
                "ir_path": "circuit_ir/case-2.circuit-ir.json",
                "ir_sha256": _sha(second_ir),
                "official_path": "official/../official/case-1.json",
            }
            rows = [first, second]
            manifest.write_bytes(b"".join(AUDIT._canonical_bytes(row) for row in rows))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload.update(
                {
                    "selected_cases": 2,
                    "succeeded": 2,
                    "prediction_manifest_sha256": _sha(manifest),
                    "prediction_set_sha256": AUDIT._stable_prediction_set_sha256(rows),
                }
            )
            payload["coverage"].update(
                {
                    "manifest_case_ids_sha256": AUDIT._compact_json_sha256(
                        ["case-1", "case-2"]
                    ),
                    "expected_case_ids_sha256": AUDIT._compact_json_sha256(
                        ["case-1", "case-2"]
                    ),
                    "expected_duplicate_group_membership_sha256": AUDIT._compact_json_sha256(
                        {"group-1": ["case-1"], "group-2": ["case-2"]}
                    ),
                }
            )
            _write_json(summary, payload)

            result = AUDIT._audit_production_summary_binding(
                root,
                summary,
                _sha(lock),
                config_hash,
                require_success=True,
                require_embedded_time=True,
            )

        self.assertFalse(result["valid"])
        self.assertIn("prediction row 1 reuses an image input", result["problems"])
        self.assertIn("prediction row 1 reuses official_path", result["problems"])

    def test_determinism_rejects_three_runs_reusing_one_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-formal",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            evidence = _make_formal_determinism(
                root,
                name="aliased-summaries",
                summary=summary,
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            environment = evidence / "environment.json"
            payload = json.loads(environment.read_text(encoding="utf-8"))
            first_path = payload["runs"][0]["source_summary_path"]
            first_hash = payload["runs"][0]["source_summary_sha256"]
            for run in payload["runs"]:
                run["source_summary_path"] = first_path
                run["source_summary_sha256"] = first_hash
            _write_json(environment, payload)

            result = AUDIT._audit_determinism(
                root,
                evidence,
                environment,
                _sha(lock),
                config_hash,
                require_formal_binding=True,
            )

        self.assertEqual("fail", result["status"])
        self.assertIn(
            "formal evidence does not bind three distinct production summaries",
            result["environment_problems"],
        )

    def test_disabled_junction_is_not_exempt_while_runtime_still_loads_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True, include_junction=True)
            pipeline, profile, _ = _make_production_config(
                root,
                learned_junctions_enabled=False,
            )
            payload = json.loads(lock.read_text(encoding="utf-8"))
            for entry in payload["models"]:
                if entry["task"] == "junction":
                    entry["status"] = "candidate"
                    entry["blessed"] = False
            _write_json(lock, payload)
            binding = AUDIT._production_config_binding(root, pipeline, profile)

            result = AUDIT._audit_model_lock(root, lock, binding)

        self.assertEqual("fail", result["status"])
        self.assertIn("junction", result["required_tasks"])
        self.assertEqual([], result["inactive_task_exemptions"])
        self.assertFalse(
            result["junction_dependency_policy"]["inactive_dependency_exemption_proven"]
        )
        self.assertIn("task 'junction' is not blessed", result["problems"])

    def test_workspace_junction_policy_is_bound_to_loader_and_readiness_code(self) -> None:
        binding = AUDIT._production_config_binding(
            WORKSPACE,
            WORKSPACE / "configs" / "pipeline.yaml",
            WORKSPACE / "configs" / "profiles" / "competition.yaml",
        )

        result = AUDIT._audit_model_lock(
            WORKSPACE,
            WORKSPACE / "models" / "models.lock.json",
            binding,
        )

        policy = result["junction_dependency_policy"]
        self.assertFalse(policy["learned_junctions_enabled"])
        self.assertTrue(policy["junction_output_path_disabled_by_config"])
        self.assertTrue(policy["composite_call_guard_and_quarantine_metadata_present"])
        self.assertFalse(policy["production_loader_loads_junction_unconditionally"])
        self.assertTrue(policy["production_loader_uses_configured_task_plan"])
        self.assertTrue(policy["readiness_requires_junction_by_default"])
        self.assertTrue(policy["readiness_supports_exact_configured_task_plan"])
        self.assertTrue(policy["runtime_task_plan_metadata_present"])
        self.assertTrue(policy["inactive_dependency_exemption_static_eligible"])
        self.assertTrue(policy["exemption_applied"])
        self.assertNotIn("junction", result["required_tasks"])
        self.assertEqual(["junction"], result["inactive_task_exemptions"])
        self.assertFalse(policy["inactive_dependency_exemption_proven"])

    def test_current_coordinate_report_without_formal_source_binding_is_not_auto_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            report_path = root / "artifacts" / "release" / "coordinate-current.json"
            _write_json(
                report_path,
                {
                    "kind": "coordinate-roundtrip-release-report",
                    "evidence_scope": "current_lock_production_image_only_inference",
                    "model_lock_sha256": _sha(lock),
                    "current_lock_inference_executed": True,
                    "models_loaded": True,
                    "strict_image_only": True,
                    "official_coordinate_origin": "bottom_left",
                    "official_y_axis": "up",
                    "origin_conversion_verified": True,
                    "config_sha256": config_hash,
                    "source_prediction_set_sha256": "d" * 64,
                    "cases_checked": 1,
                    "points_checked": 10,
                    "max_roundtrip_error_px": 0.0,
                },
            )

            report = AUDIT.build_audit(workspace_root=root)

        selection = report["evidence_selection"]["coordinate"]
        self.assertIsNone(selection["selected_path"])
        self.assertEqual(1, len(selection["candidates"]))
        self.assertFalse(selection["candidates"][0]["eligible"])
        self.assertTrue(
            any(
                "formal" in reason or "source_summary_path" in reason
                for reason in selection["candidates"][0]["rejection_reasons"]
            )
        )

    def test_traceable_errors_cannot_use_a_manifest_other_than_summary_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = _make_lock(root, blessed=True)
            _, _, config_hash = _make_production_config(root)
            summary = _make_formal_production_run(
                root,
                name="production-e2e-trace",
                lock_hash=_sha(lock),
                config_hash=config_hash,
            )
            predictions = summary.parent / "predictions.jsonl"
            good = summary.parent / "good.jsonl"
            good.write_bytes(predictions.read_bytes())
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            summary_payload["prediction_manifest"] = "good.jsonl"
            summary_payload["prediction_manifest_sha256"] = _sha(good)
            _write_json(summary, summary_payload)
            row = json.loads(good.read_text(encoding="utf-8"))
            row.update(
                {
                    "ir_path": None,
                    "ir_sha256": None,
                    "official_path": None,
                    "official_sha256": None,
                    "official_output_sha256": None,
                    "status": "error",
                    "error": {
                        "type": "PipelineError",
                        "message": "failed",
                        "stage": "validation",
                        "error_id": "9" * 64,
                        "diagnostic_count": 0,
                        "diagnostic_codes": {},
                        "diagnostics": [],
                    },
                }
            )
            predictions.write_bytes(AUDIT._canonical_bytes(row))
            harness = root / "harness.py"
            harness.write_text(
                'def _traceable_error():\n    pass\n"error_id"\n"diagnostic_codes"\n"diagnostics"\n',
                encoding="utf-8",
            )

            result = AUDIT._audit_traceable_errors(
                root,
                _sha(lock),
                harness,
                config_hash,
                require_formal_binding=True,
            )

        self.assertEqual("unverified", result["status"])
        self.assertEqual(0, result["current_lock_error_rows"])
        self.assertFalse(result["rows"][0]["summary_manifest_bound"])

    def test_build_never_authorizes_bless_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_lock(root, blessed=False)
            _make_determinism(root)
            _write_json(
                root / "src" / "pcbparse" / "resources" / "official_schema.json",
                {"official_schema": SCHEMA},
            )
            (root / "docker").mkdir(parents=True)
            (root / "docker" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

            result = AUDIT.build_audit(workspace_root=root)

        self.assertFalse(result["decision"]["release_ready"])
        self.assertFalse(result["decision"]["promotion_authorized"])
        self.assertFalse(result["decision"]["bless_authorized"])
        self.assertEqual(64, len(result["audit_payload_sha256"]))


if __name__ == "__main__":
    unittest.main()
