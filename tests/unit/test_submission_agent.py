from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from pcbparse.output.official_serializer import atomic_write_official_json
from pcbparse.submission_agent import RuntimeBundle, build_parser, run_submission


EMPTY_OUTPUT = {"components": {}, "pins": {}, "nets": {}}


class SubmissionAgentTests(unittest.TestCase):
    def test_parser_accepts_official_single_dash_flags(self) -> None:
        args = build_parser().parse_args(["-input", "/input", "-output", "/output"])
        self.assertEqual(Path("/input"), args.input_dir)
        self.assertEqual(Path("/output"), args.output_dir)

    def test_batch_writes_exact_official_layout_and_decision_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            Image.new("RGB", (8, 8), "white").save(inputs / "0025-Altium Designer.png")
            Image.new("RGB", (8, 8), "white").save(inputs / "中文样例.png")
            (inputs / "ignored.json").write_text("{}", encoding="utf-8")

            def fake_infer(_image_path, output_path, **_kwargs):
                atomic_write_official_json(EMPTY_OUTPUT, output_path)
                context = SimpleNamespace(
                    manifest=lambda: {
                        "model_versions": {"component": "test"},
                        "tokens": {"calls": 0, "input_tokens": 0, "output_tokens": 0},
                    }
                )
                return SimpleNamespace(
                    context=context,
                    diagnostics=[],
                    output_hash="a" * 64,
                    output=EMPTY_OUTPUT,
                    quality={
                        "status": "empty_result",
                        "strict_success": False,
                        "degraded_nonempty": False,
                        "empty_result": True,
                        "fallback_empty": False,
                        "failed_to_write": False,
                        "legitimate_empty_input": True,
                    },
                )

            with patch(
                "pcbparse.submission_agent._load_runtime",
                return_value=_runtime_bundle(),
            ), patch("pcbparse.submission_agent.infer_one", side_effect=fake_infer):
                report = run_submission(inputs, outputs, app_root=root)

            expected = {
                "0025-Altium Designer_Pred.json",
                "中文样例_Pred.json",
                "run.log",
            }
            self.assertEqual(
                expected,
                {path.name for path in (outputs / "final_results").iterdir()},
            )
            decision = json.loads(
                (outputs / "decision_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual("edarace-agent-decision-log", decision["kind"])
            self.assertEqual(2, decision["summary"]["total_cases"])
            self.assertEqual(0, decision["summary"]["fallback_cases"])
            self.assertEqual("ok", report["summary"]["status"])
            self.assertEqual(2, decision["summary"]["status_counts"]["empty_result"])
            self.assertEqual(
                2,
                sum(decision["summary"]["status_counts"].values()),
                "each case must carry exactly one primary status",
            )

    def test_case_failure_emits_schema_valid_empty_fallback_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            Image.new("RGB", (8, 8), "white").save(inputs / "case_a.png")
            Image.new("RGB", (8, 8), "white").save(inputs / "case_b.png")

            calls = 0

            def fake_infer(_image_path, output_path, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("synthetic failure")
                atomic_write_official_json(EMPTY_OUTPUT, output_path)
                return SimpleNamespace(
                    context=SimpleNamespace(
                        manifest=lambda: {
                            "model_versions": {},
                            "tokens": {
                                "calls": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                            },
                        }
                    ),
                    diagnostics=[],
                    output_hash="b" * 64,
                    output=EMPTY_OUTPUT,
                    quality={
                        "status": "empty_result",
                        "strict_success": False,
                        "degraded_nonempty": False,
                        "empty_result": True,
                        "fallback_empty": False,
                        "failed_to_write": False,
                        "legitimate_empty_input": True,
                    },
                )

            with patch(
                "pcbparse.submission_agent._load_runtime",
                return_value=_runtime_bundle(),
            ), patch("pcbparse.submission_agent.infer_one", side_effect=fake_infer):
                report = run_submission(inputs, outputs, app_root=root)

            self.assertEqual("degraded", report["summary"]["status"])
            self.assertEqual(1, report["summary"]["fallback_cases"])
            self.assertEqual(1, report["summary"]["status_counts"]["fallback_empty"])
            self.assertEqual(
                2,
                sum(report["summary"]["status_counts"].values()),
                "each case must carry exactly one primary status",
            )
            fallback = json.loads(
                (outputs / "final_results" / "case_a_Pred.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(EMPTY_OUTPUT, fallback)
            self.assertTrue((outputs / "final_results" / "case_b_Pred.json").is_file())

    def test_runtime_initialization_failure_still_covers_every_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            Image.new("RGB", (8, 8), "white").save(inputs / "only.png")

            with patch(
                "pcbparse.submission_agent._load_runtime",
                side_effect=RuntimeError("model load failed"),
            ):
                report = run_submission(inputs, outputs, app_root=root)

            self.assertEqual(1, report["summary"]["fallback_cases"])
            self.assertEqual(1, report["summary"]["status_counts"]["fallback_empty"])
            self.assertFalse(report["runtime"]["ready"])
            self.assertTrue((outputs / "final_results" / "only_Pred.json").is_file())

    def test_five_primary_statuses_from_real_orchestrator_and_submission(self) -> None:
        from pcbparse.config import load_config
        from pcbparse.ir.models import (
            BBox,
            Component,
            IRState,
            Pin,
            Point,
        )
        from pcbparse.pipeline.providers import EmptyImageProvider

        class ScenarioProvider(EmptyImageProvider):
            version = "five-status-fixture"

            def build_ir(self, input_path, ctx=None):
                ir = super().build_ir(input_path, ctx)
                ir.state = IRState.PERCEIVED
                stem = Path(input_path).stem
                if stem in {"case_crash", "case_writefail"}:
                    raise RuntimeError("synthetic provider failure")
                if stem == "case_ok":
                    ir.components = [
                        Component(
                            internal_id="c1",
                            key="R1",
                            category="resistor",
                            value="10k",
                            bbox=BBox(x1=2, y1=2, x2=12, y2=12),
                        )
                    ]
                    ir.pins = [
                        Pin(
                            internal_id="p1",
                            component_id="c1",
                            number="1",
                            name="A",
                            point=Point(x=8, y=8),
                        )
                    ]
                elif stem == "case_nokey":
                    ir.components = [
                        Component(
                            internal_id="c_missing",
                            key=None,
                            category="resistor",
                            value="1k",
                            bbox=BBox(x1=1, y1=1, x2=6, y2=6),
                        ),
                        Component(
                            internal_id="c2",
                            key="R2",
                            category="resistor",
                            value="2k",
                            bbox=BBox(x1=9, y1=1, x2=14, y2=6),
                        ),
                    ]
                    ir.pins = [
                        Pin(
                            internal_id="p2",
                            component_id="c2",
                            number="1",
                            name="A",
                            point=Point(x=11, y=3),
                        )
                    ]
                elif stem == "case_pruned_empty":
                    ir.pins = [
                        Pin(
                            internal_id="ownerless",
                            component_id=None,
                            number="1",
                            point=Point(x=8, y=8),
                        )
                    ]
                return ir

        config = load_config("configs/pipeline.yaml")
        config["run"]["submission_mode"] = "degraded"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            for name in (
                "case_ok.png",
                "case_nokey.png",
                "case_pruned_empty.png",
                "case_empty.png",
                "case_crash.png",
                "case_writefail.png",
            ):
                Image.new("RGB", (16, 16), "white").save(inputs / name)

            provider = ScenarioProvider()

            def scenario_runtime(*_args, **_kwargs):
                return RuntimeBundle(
                    provider=provider,
                    config=config,
                    readiness={
                        "ready": True,
                        "required_learned_tasks": [],
                        "verified_learned_tasks": [],
                    },
                    config_sha256="c" * 64,
                    model_lock_sha256="d" * 64,
                )

            import pcbparse.submission_agent as submission_module

            real_write = submission_module.atomic_write_official_json

            def flaky_write(payload, path, *args, **kwargs):
                if Path(path).name.startswith("case_writefail"):
                    raise OSError("synthetic disk failure")
                return real_write(payload, path, *args, **kwargs)

            with patch(
                "pcbparse.submission_agent._load_runtime",
                side_effect=scenario_runtime,
            ), patch(
                "pcbparse.submission_agent.atomic_write_official_json",
                side_effect=flaky_write,
            ):
                report = run_submission(inputs, outputs, app_root=root)

            statuses = {
                case["case_name"]: case["status"] for case in report["cases"]
            }
            self.assertEqual(
                {
                    "case_ok": "strict_success",
                    "case_nokey": "degraded_nonempty",
                    "case_pruned_empty": "empty_result",
                    "case_empty": "empty_result",
                    "case_crash": "fallback_empty",
                    "case_writefail": "failed_to_write",
                },
                statuses,
            )
            by_name = {case["case_name"]: case for case in report["cases"]}
            self.assertEqual(1, by_name["case_nokey"]["quality"]["pruned_components"])
            self.assertGreaterEqual(
                by_name["case_pruned_empty"]["quality"]["pruned_pins"], 1
            )
            self.assertTrue(
                by_name["case_empty"]["quality"]["legitimate_empty_input"]
            )
            self.assertFalse(
                by_name["case_pruned_empty"]["quality"]["legitimate_empty_input"]
            )
            self.assertTrue(by_name["case_ok"]["quality"]["strict_success"])
            self.assertEqual(
                6,
                sum(report["summary"]["status_counts"].values()),
                "each case must carry exactly one primary status",
            )
            self.assertEqual(6, report["summary"]["total_cases"])
            self.assertEqual(1, report["summary"]["failed_to_write_cases"])
            self.assertIsNone(by_name["case_writefail"]["output_file"])
            self.assertIn("write_error", by_name["case_writefail"])
            self.assertEqual("degraded", report["summary"]["status"])


def _runtime_bundle() -> RuntimeBundle:
    return RuntimeBundle(
        provider=object(),
        config={"run": {"max_total_seconds": 10.0}},
        readiness={
            "ready": True,
            "required_learned_tasks": ["component", "ocr", "pin", "wire"],
            "verified_learned_tasks": ["component", "ocr", "pin", "wire"],
        },
        config_sha256="c" * 64,
        model_lock_sha256="d" * 64,
    )


if __name__ == "__main__":
    unittest.main()
