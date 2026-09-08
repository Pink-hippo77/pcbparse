from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = WORKSPACE / "scripts" / "release_trace_probe.py"
AUDIT_SCRIPT = WORKSPACE / "scripts" / "audit_release_hard_gates.py"
HARNESS = WORKSPACE / "src" / "pcbparse" / "evaluation" / "infer_manifest.py"
SPEC = importlib.util.spec_from_file_location("release_trace_probe", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _stable_prediction_set(rows: list[dict[str, object]]) -> str:
    fields = (
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
    return _json_hash([{field: row[field] for field in fields} for row in rows])


def _materialize_runtime_source_population(root: Path, contract) -> dict[str, object]:
    for relative in contract.PRODUCTION_RUNTIME_SOURCE_DIRECTORIES:
        source = root / relative / "__init__.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            source.write_text(f'"""fixture for {relative}"""\n', encoding="utf-8")
    for relative in contract.PRODUCTION_RUNTIME_SOURCE_FILES:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            continue
        if source.suffix == ".json":
            _write_json(source, {"fixture": relative})
        else:
            source.write_text(f'"""fixture for {relative}"""\n', encoding="utf-8")
    return contract._independent_production_runtime_source_manifest(root)


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "scripts").mkdir(parents=True)
        shutil.copyfile(AUDIT_SCRIPT, root / "scripts" / AUDIT_SCRIPT.name)
        self.harness = root / "src" / "pcbparse" / "evaluation" / "infer_manifest.py"
        self.harness.parent.mkdir(parents=True)
        shutil.copyfile(HARNESS, self.harness)

        self.pipeline = root / "configs" / "pipeline.yaml"
        self.profile = root / "configs" / "profiles" / "competition.yaml"
        self.lock = root / "models" / "models.lock.json"
        self.official_profile = (
            root / "src" / "pcbparse" / "resources" / "official_schema.json"
        )
        self.model_tasks = ["component", "ocr", "pin", "wire"]
        self.model_task_plan = {
            "contract": "production_model_task_plan_v1",
            "junction_mode": "deterministic_vectorizer",
            "required_model_tasks": self.model_tasks,
            "loaded_model_tasks": self.model_tasks,
            "learned_junction_model_required": False,
            "learned_junction_model_loaded": False,
            "learned_junction_model_invoked": False,
        }
        self.models: list[Path] = []
        model_entries = []
        for task in self.model_tasks:
            model = root / "models" / f"{task}.bin"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(f"trace-probe-fixture-{task}".encode("ascii"))
            self.models.append(model)
            model_entries.append(
                {
                    "task": task,
                    "path": model.name,
                    "sha256": _sha(model),
                    "status": "blessed",
                    "blessed": True,
                }
            )
        _write_json(
            self.pipeline,
            {
                "run": {},
                "qwen": {},
                "validation": {},
                "graph": {"learned_junctions_enabled": False},
            },
        )
        _write_json(
            self.profile,
            {
                "name": "competition",
                "provider": "production",
                "qwen_enabled": False,
                "artifact_policy": "error",
                "strict_validation": True,
            },
        )
        _write_json(
            self.lock,
            {
                "schema_version": "1.0",
                "models": model_entries,
            },
        )
        _write_json(self.official_profile, {"version": "trace-fixture-v1"})

        self.probe_root = (
            root
            / "artifacts"
            / "release"
            / "trace-probes"
            / "current-lock-invalid-png-v1"
        )
        self.verification_report = (
            root
            / "artifacts"
            / "release"
            / "trace-verifications"
            / "current-lock-invalid-png-v1.json"
        )
        contract = PROBE._load_hard_gate_contract(root)
        self.runtime_source_manifest = _materialize_runtime_source_population(
            root, contract
        )
        bindings = PROBE._current_control_bindings(
            root=root,
            contract=contract,
            models_lock=self.lock,
            pipeline_config=self.pipeline,
            production_profile=self.profile,
            official_profile=self.official_profile,
        )
        self.config_hash = bindings.effective_config_sha256
        self.lock_hash = bindings.model_lock_sha256

    def build(self) -> dict[str, object]:
        return PROBE.build_trace_probe(
            workspace_root=self.root,
            output_root=self.probe_root,
        )

    def make_failed_run(self) -> Path:
        image = self.probe_root / "dataset" / PROBE.PROBE_IMAGE_NAME
        split = self.probe_root / "split.jsonl"
        run = self.root / "work" / "production-e2e-trace-probe-fixture"
        (run / "official").mkdir(parents=True)
        (run / "circuit_ir").mkdir()
        inference_run_id = "a" * 64
        error: dict[str, object] = {
            "type": "UnidentifiedImageError",
            "message": "cannot identify fixed trace probe image",
            "stage": "inference",
            "diagnostic_count": 0,
            "diagnostic_codes": {},
            "diagnostics": [],
        }
        error["error_id"] = _json_hash(
            {
                "case_id": PROBE.PROBE_CASE_ID,
                "image_sha256": PROBE.PROBE_IMAGE_SHA256,
                "config_hash": self.config_hash,
                "model_lock_sha256": self.lock_hash,
                "error": error,
            }
        )
        row: dict[str, object] = {
            "case_id": PROBE.PROBE_CASE_ID,
            "duplicate_group": PROBE.PROBE_DUPLICATE_GROUP,
            "image_sha256": PROBE.PROBE_IMAGE_SHA256,
            "ir_path": None,
            "ir_sha256": None,
            "official_path": None,
            "official_sha256": None,
            "official_output_sha256": None,
            "provider_kind": "production",
            "model_lock_sha256": self.lock_hash,
            "config_hash": self.config_hash,
            "inference_run_id": inference_run_id,
            "sidecar_access": False,
            "input_paths_read": [
                str(image),
                str(self.pipeline),
                str(self.profile),
                str(self.lock),
                *(str(model) for model in self.models),
                str(self.official_profile),
            ],
            "elapsed_seconds": 0.5,
            "status": "error",
            "error": error,
        }
        predictions = run / "predictions.jsonl"
        predictions.write_bytes(_canonical_line(row))
        completed = datetime.now(timezone.utc) - timedelta(seconds=1)
        completed_text = completed.isoformat().replace("+00:00", "Z")
        summary = {
            "schema_version": 2,
            "kind": "production-image-only-inference",
            "inference_run_id": inference_run_id,
            "started_at_utc": (completed - timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "completed_at_utc": completed_text,
            "generated_at_utc": completed_text,
            "accuracy_mode": "image-only",
            "execution_mode": "production",
            "provider_kind": "production",
            "sidecar_access": False,
            "status": "failed",
            "all_successful": False,
            "split": PROBE.PROBE_SPLIT,
            "selected_cases": 1,
            "selected_duplicate_groups": 1,
            "attempted": 1,
            "succeeded": 0,
            "failed": 1,
            "not_run": 0,
            "fail_fast": False,
            "elapsed_seconds": 0.75,
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
                "initialization_seconds": 0.2,
                "case_loop_seconds": 0.5,
                "evidence_finalize_seconds": 0.05,
                "total_elapsed_seconds": 0.75,
            },
            "config_hash": self.config_hash,
            "resolved_perception_config": {
                "learned_junctions_enabled": False
            },
            "resolved_perception_config_sha256": _json_hash(
                {"learned_junctions_enabled": False}
            ),
            "model_task_plan": self.model_task_plan,
            "runtime_source_manifest": self.runtime_source_manifest,
            "model_lock_sha256": self.lock_hash,
            "prediction_manifest": "predictions.jsonl",
            "prediction_manifest_sha256": _sha(predictions),
            "prediction_set_sha256": _stable_prediction_set([row]),
            "split_manifest_sha256": _sha(split),
            "coverage": {
                "expected_case_ids_sha256": _json_hash([PROBE.PROBE_CASE_ID]),
                "expected_duplicate_group_membership_sha256": _json_hash(
                    {PROBE.PROBE_DUPLICATE_GROUP: [PROBE.PROBE_CASE_ID]}
                ),
                "selected_duplicate_groups": 1,
                "manifest_case_ids_sha256": _json_hash([PROBE.PROBE_CASE_ID]),
                "exact_case_coverage": True,
                "exact_duplicate_group_coverage": True,
                "duplicate_case_ids": [],
            },
            "control_inputs": {
                "dataset_root": str(self.probe_root / "dataset"),
                "split_manifest": str(split),
                "pipeline_config": str(self.pipeline),
                "profile": str(self.profile),
                "official_profile": str(self.official_profile),
                "models_lock": str(self.lock),
            },
            "selection_control_paths_read": [str(split)],
            "inference_input_policy": "fixture exact image/control allowlist",
            "readiness": {
                "status": "ready",
                "ready": True,
                "required_learned_tasks": self.model_tasks,
                "registered_learned_tasks": self.model_tasks,
                "verified_learned_tasks": self.model_tasks,
            },
        }
        summary_path = run / "summary.json"
        _write_json(summary_path, summary)
        return summary_path

    def verify(self, summary: Path) -> dict[str, object]:
        return PROBE.verify_trace_probe_result(
            workspace_root=self.root,
            probe_root=self.probe_root,
            inference_summary=summary,
            verification_report=self.verification_report,
        )


class ReleaseTraceProbeTests(unittest.TestCase):
    def test_builds_fixed_image_only_probe_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))

            result = fixture.build()

            self.assertEqual("ready", result["status"])
            image = fixture.probe_root / "dataset" / PROBE.PROBE_IMAGE_NAME
            self.assertEqual(PROBE.PROBE_IMAGE_BYTES, image.read_bytes())
            self.assertEqual(PROBE.PROBE_IMAGE_SHA256, _sha(image))
            self.assertEqual([image], list((fixture.probe_root / "dataset").iterdir()))
            self.assertEqual(
                {"dataset", "source_report.json", "split.jsonl"},
                {path.name for path in fixture.probe_root.iterdir()},
            )
            manifest_rows = [
                json.loads(line)
                for line in (fixture.probe_root / "split.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([PROBE._probe_manifest_row()], manifest_rows)
            report = json.loads(
                (fixture.probe_root / "source_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fixture.lock_hash, report["controls"]["models_lock"]["sha256"])
            self.assertEqual(fixture.config_hash, report["controls"]["effective_config_sha256"])
            self.assertFalse(list(fixture.probe_root.parent.glob(".*.staging-*")))

    def test_verifies_formal_failure_and_closes_trace_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()

            result = fixture.verify(summary)

            self.assertEqual("pass", result["status"])
            self.assertTrue(fixture.verification_report.is_file())
            report = json.loads(fixture.verification_report.read_text(encoding="utf-8"))
            self.assertTrue(report["bindings"]["formal_timing_contract_validated"])
            self.assertEqual(_sha(summary), report["inference_evidence"]["summary_sha256"])
            self.assertEqual(
                _sha(summary.parent / "predictions.jsonl"),
                report["inference_evidence"]["predictions_sha256"],
            )
            contract = PROBE._load_hard_gate_contract(fixture.root)
            gate = contract._audit_traceable_errors(
                fixture.root,
                fixture.lock_hash,
                fixture.harness,
                fixture.config_hash,
                require_formal_binding=True,
            )
            self.assertEqual("pass", gate["status"], gate["rows"])
            self.assertEqual(1, gate["current_lock_traceable_rows"])

    def test_rejects_timing_contract_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["timing"]["cold_start_included"] = False
            _write_json(summary, payload)

            with self.assertRaisesRegex(PROBE.TraceProbeError, "timing_cold_start"):
                fixture.verify(summary)

            self.assertFalse(fixture.verification_report.exists())

    def test_rejects_wrong_failure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["not_run"] = 1
            _write_json(summary, payload)

            with self.assertRaisesRegex(PROBE.TraceProbeError, "not_run"):
                fixture.verify(summary)

            self.assertFalse(fixture.verification_report.exists())

    def test_rejects_error_id_that_does_not_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()
            predictions = summary.parent / "predictions.jsonl"
            row = json.loads(predictions.read_text(encoding="utf-8"))
            row["error"]["error_id"] = "0" * 64
            predictions.write_bytes(_canonical_line(row))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["prediction_manifest_sha256"] = _sha(predictions)
            payload["prediction_set_sha256"] = _stable_prediction_set([row])
            _write_json(summary, payload)

            with self.assertRaisesRegex(
                PROBE.TraceProbeError, "error_id_mismatch|stably recomputed"
            ):
                fixture.verify(summary)

            self.assertFalse(fixture.verification_report.exists())

    def test_rejects_probe_byte_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()
            image = fixture.probe_root / "dataset" / PROBE.PROBE_IMAGE_NAME
            image.write_bytes(PROBE.PROBE_IMAGE_BYTES + b"tampered")

            with self.assertRaisesRegex(PROBE.TraceProbeError, "fixed undecodable"):
                fixture.verify(summary)

            self.assertFalse(fixture.verification_report.exists())

    def test_never_reuses_existing_probe_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.probe_root.mkdir(parents=True)
            sentinel = fixture.probe_root / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(PROBE.TraceProbeError, "refusing to reuse"):
                fixture.build()

            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))

    def test_never_overwrites_existing_verification_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.build()
            summary = fixture.make_failed_run()
            fixture.verification_report.parent.mkdir(parents=True)
            fixture.verification_report.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(PROBE.TraceProbeError, "refusing to reuse"):
                fixture.verify(summary)

            self.assertEqual(
                "keep", fixture.verification_report.read_text(encoding="utf-8")
            )

    def test_staging_failure_leaves_no_probe_or_temporary_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            original = PROBE._atomic_write_bytes
            calls = 0

            def fail_second_write(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected split write failure")
                original(path, payload)

            with mock.patch.object(PROBE, "_atomic_write_bytes", side_effect=fail_second_write):
                with self.assertRaisesRegex(OSError, "injected"):
                    fixture.build()

            self.assertFalse(fixture.probe_root.exists())
            self.assertFalse(list(fixture.probe_root.parent.glob(".*.staging-*")))


if __name__ == "__main__":
    unittest.main()
