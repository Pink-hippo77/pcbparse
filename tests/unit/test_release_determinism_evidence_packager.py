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
SCRIPT = WORKSPACE / "scripts" / "package_release_determinism_evidence.py"
AUDIT_SCRIPT = WORKSPACE / "scripts" / "audit_release_hard_gates.py"
SPEC = importlib.util.spec_from_file_location(
    "release_determinism_evidence_packager", SCRIPT
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT}")
PACKAGER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGER
SPEC.loader.exec_module(PACKAGER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_hash(value: object) -> str:
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
    return _compact_hash(
        [{field: row[field] for field in fields} for row in rows]
    )


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
        self.pipeline = root / "configs" / "pipeline.yaml"
        self.profile = root / "configs" / "profiles" / "competition.yaml"
        self.lock = root / "models" / "models.lock.json"
        self.official_profile = (
            root / "src" / "pcbparse" / "resources" / "official_schema.json"
        )
        self.dataset = root / "dataset-release50"
        self.manifest = root / "data" / "splits" / "release50.jsonl"
        self.output = root / "artifacts" / "determinism" / "formal-release50"

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
        self.model_paths: list[Path] = []
        model_entries = []
        for task in self.model_tasks:
            artifact = root / "models" / "candidates" / task / "model.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"fixture-{task}".encode("ascii"))
            self.model_paths.append(artifact)
            model_entries.append(
                {
                    "task": task,
                    "path": artifact.relative_to(self.lock.parent).as_posix(),
                    "sha256": _sha(artifact),
                    "status": "blessed",
                    "blessed": True,
                }
            )
        _write_json(
            self.lock,
            {"schema_version": "1.0", "models": model_entries},
        )
        _write_json(self.official_profile, {"version": "fixture-v1"})
        contract = PACKAGER._load_hard_gate_contract(root)
        self.runtime_source_manifest = _materialize_runtime_source_population(
            root, contract
        )

        self.cases: list[dict[str, str]] = []
        manifest_rows = []
        for number in range(1, 51):
            case_id = f"{number:04d}"
            relative_image = Path(case_id) / f"{case_id}.png"
            image_path = self.dataset / relative_image
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(f"fixture-image-{case_id}".encode("ascii"))
            image_sha256 = _sha(image_path)
            group = f"group-{case_id}"
            self.cases.append(
                {
                    "case_id": case_id,
                    "duplicate_group": group,
                    "image_path": str(image_path),
                    "image_sha256": image_sha256,
                }
            )
            manifest_rows.append(
                {
                    "sample_id": case_id,
                    "duplicate_group": group,
                    "duplicate_group_size": 1,
                    "split": "release50",
                    "files": {
                        "image": {
                            "path": relative_image.as_posix(),
                            "sha256": image_sha256,
                        }
                    },
                }
            )
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_bytes(
            b"".join(_canonical_line(row) for row in manifest_rows)
        )
        self.config_hash = self._config_hash()
        self.lock_hash = _sha(self.lock)

    def _config_hash(self) -> str:
        pipeline = json.loads(self.pipeline.read_text(encoding="utf-8"))
        profile = json.loads(self.profile.read_text(encoding="utf-8"))
        pipeline["run"]["profile"] = profile["name"]
        pipeline["run"]["artifact_policy"] = profile["artifact_policy"]
        pipeline["qwen"]["enabled"] = profile["qwen_enabled"]
        pipeline["validation"]["require_zero_errors"] = profile[
            "strict_validation"
        ]
        return _compact_hash(pipeline)

    def make_run(
        self,
        number: int,
        *,
        run_id: str | None = None,
        official_variant: str = "stable",
    ) -> Path:
        run_root = self.root / "work" / f"production-e2e-source-run{number}"
        official_root = run_root / "official"
        ir_root = run_root / "circuit_ir"
        official_root.mkdir(parents=True)
        ir_root.mkdir(parents=True)
        inference_run_id = run_id or str(number) * 64
        rows: list[dict[str, object]] = []
        shared_inputs = [
            str(self.pipeline),
            str(self.profile),
            str(self.lock),
            *(str(path) for path in self.model_paths),
            str(self.official_profile),
        ]
        for case in self.cases:
            case_id = case["case_id"]
            ir_path = ir_root / f"{case_id}.circuit-ir.json"
            official_path = official_root / f"{case_id}.json"
            _write_json(
                ir_path,
                {
                    "case_id": case_id,
                    "state": "SERIALIZED",
                    "metadata": {
                        "model_task_plan": self.model_task_plan,
                        "model_versions": {
                            task: "fixture-v1" for task in self.model_tasks
                        },
                        "quarantined_model_versions": {},
                    },
                },
            )
            _write_json(
                official_path,
                {
                    "components": {},
                    "pins": {},
                    "nets": {},
                    "fixture_variant": official_variant,
                    "case_id": case_id,
                },
            )
            official_sha256 = _sha(official_path)
            rows.append(
                {
                    "case_id": case_id,
                    "duplicate_group": case["duplicate_group"],
                    "image_sha256": case["image_sha256"],
                    "ir_path": f"circuit_ir/{case_id}.circuit-ir.json",
                    "ir_sha256": _sha(ir_path),
                    "official_path": f"official/{case_id}.json",
                    "official_sha256": official_sha256,
                    "official_output_sha256": official_sha256,
                    "provider_kind": "production",
                    "model_lock_sha256": self.lock_hash,
                    "config_hash": self.config_hash,
                    "inference_run_id": inference_run_id,
                    "sidecar_access": False,
                    "input_paths_read": [case["image_path"], *shared_inputs],
                    "elapsed_seconds": 0.01,
                    "status": "ok",
                    "error": None,
                }
            )
        predictions = run_root / "predictions.jsonl"
        predictions.write_bytes(b"".join(_canonical_line(row) for row in rows))
        case_ids = [case["case_id"] for case in self.cases]
        membership = {
            case["duplicate_group"]: [case["case_id"]] for case in self.cases
        }
        completed = datetime.now(timezone.utc) - timedelta(
            minutes=10 - number
        )
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
            "status": "ok",
            "all_successful": True,
            "split": "release50",
            "selected_cases": 50,
            "selected_duplicate_groups": 50,
            "attempted": 50,
            "succeeded": 50,
            "failed": 0,
            "not_run": 0,
            "fail_fast": False,
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
                "initialization_seconds": 0.2,
                "case_loop_seconds": 0.7,
                "evidence_finalize_seconds": 0.1,
                "total_elapsed_seconds": 1.0,
            },
            "config_hash": self.config_hash,
            "resolved_perception_config": {
                "learned_junctions_enabled": False
            },
            "resolved_perception_config_sha256": _compact_hash(
                {"learned_junctions_enabled": False}
            ),
            "model_task_plan": self.model_task_plan,
            "runtime_source_manifest": self.runtime_source_manifest,
            "model_lock_sha256": self.lock_hash,
            "prediction_manifest": "predictions.jsonl",
            "prediction_manifest_sha256": _sha(predictions),
            "prediction_set_sha256": _stable_prediction_set(rows),
            "split_manifest_sha256": _sha(self.manifest),
            "coverage": {
                "expected_case_ids_sha256": _compact_hash(case_ids),
                "expected_duplicate_group_membership_sha256": _compact_hash(
                    membership
                ),
                "selected_duplicate_groups": 50,
                "manifest_case_ids_sha256": _compact_hash(case_ids),
                "exact_case_coverage": True,
                "exact_duplicate_group_coverage": True,
                "duplicate_case_ids": [],
            },
            "control_inputs": {
                "dataset_root": str(self.dataset),
                "split_manifest": str(self.manifest),
                "pipeline_config": str(self.pipeline),
                "profile": str(self.profile),
                "official_profile": str(self.official_profile),
                "models_lock": str(self.lock),
            },
            "selection_control_paths_read": [str(self.manifest)],
            "readiness": {
                "status": "ready",
                "ready": True,
                "required_learned_tasks": self.model_tasks,
                "registered_learned_tasks": self.model_tasks,
                "verified_learned_tasks": self.model_tasks,
            },
        }
        summary_path = run_root / "summary.json"
        _write_json(summary_path, summary)
        return summary_path

    def package(self, summaries: list[Path]) -> dict[str, object]:
        return PACKAGER.package_release_determinism_evidence(
            workspace_root=self.root,
            source_summaries=summaries,
            release_manifest=self.manifest,
            split="release50",
            output_root=self.output,
        )


class ReleaseDeterminismEvidencePackagerTests(unittest.TestCase):
    def test_packages_flat_official_json_and_passes_hard_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [fixture.make_run(number) for number in range(1, 4)]

            result = fixture.package(summaries)

            self.assertTrue(result["ok"])
            environment_path = fixture.output / "environment.json"
            self.assertTrue(environment_path.is_file())
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            self.assertEqual("ok", environment["status"])
            self.assertTrue(environment["formal_release_evidence"])
            self.assertEqual(
                "current_lock_production_image_only_inference",
                environment["evidence_scope"],
            )
            self.assertEqual(3, len(environment["runs"]))
            for number in range(1, 4):
                run_dir = fixture.output / f"run{number}"
                files = sorted(path.name for path in run_dir.iterdir())
                self.assertEqual(50, len(files))
                self.assertEqual("0001.json", files[0])
                self.assertEqual("0050.json", files[-1])
                self.assertFalse(any(path.is_dir() for path in run_dir.iterdir()))

            contract = PACKAGER._load_hard_gate_contract(fixture.root)
            audited = contract._audit_determinism(
                fixture.root,
                fixture.output,
                environment_path,
                fixture.lock_hash,
                fixture.config_hash,
                require_formal_binding=True,
            )
            self.assertEqual("pass", audited["status"], audited["environment_problems"])

    def test_rejects_distinct_valid_run_with_different_official_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [
                fixture.make_run(1),
                fixture.make_run(2),
                fixture.make_run(3, official_variant="different"),
            ]

            with self.assertRaisesRegex(
                PACKAGER.DeterminismEvidenceError,
                "release identity|official JSON",
            ):
                fixture.package(summaries)

            self.assertFalse(fixture.output.exists())

    def test_rejects_duplicate_inference_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [
                fixture.make_run(1),
                fixture.make_run(2),
                fixture.make_run(3, run_id="2" * 64),
            ]

            with self.assertRaisesRegex(
                PACKAGER.DeterminismEvidenceError, "distinct inference_run_id"
            ):
                fixture.package(summaries)

            self.assertFalse(fixture.output.exists())

    def test_rejects_release_manifest_changed_after_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [fixture.make_run(number) for number in range(1, 4)]
            fixture.manifest.write_bytes(
                fixture.manifest.read_bytes() + b"\n"
            )

            with self.assertRaisesRegex(
                PACKAGER.DeterminismEvidenceError,
                "formal production evidence|release50 checks",
            ):
                fixture.package(summaries)

            self.assertFalse(fixture.output.exists())

    def test_rejects_production_profile_changed_after_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [fixture.make_run(number) for number in range(1, 4)]
            profile = json.loads(fixture.profile.read_text(encoding="utf-8"))
            profile["artifact_policy"] = "warn"
            _write_json(fixture.profile, profile)

            with self.assertRaisesRegex(
                PACKAGER.DeterminismEvidenceError,
                "formal production evidence|release50 checks",
            ):
                fixture.package(summaries)

            self.assertFalse(fixture.output.exists())

    def test_existing_destination_is_never_reused_or_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [fixture.make_run(number) for number in range(1, 4)]
            fixture.output.mkdir(parents=True)
            sentinel = fixture.output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(
                PACKAGER.DeterminismEvidenceError, "refusing to reuse or overwrite"
            ):
                fixture.package(summaries)

            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))

    def test_staging_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            summaries = [fixture.make_run(number) for number in range(1, 4)]

            with mock.patch.object(
                PACKAGER, "_atomic_write_bytes", side_effect=OSError("fixture failure")
            ):
                with self.assertRaisesRegex(OSError, "fixture failure"):
                    fixture.package(summaries)

            self.assertFalse(fixture.output.exists())
            leftovers = list(
                fixture.output.parent.glob(f".{fixture.output.name}.staging-*")
            )
            self.assertEqual([], leftovers)


if __name__ == "__main__":
    unittest.main()
