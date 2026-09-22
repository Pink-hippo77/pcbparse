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
                )

            with patch(
                "pcbparse.submission_agent._load_runtime",
                return_value=_runtime_bundle(),
            ), patch("pcbparse.submission_agent.infer_one", side_effect=fake_infer):
                report = run_submission(inputs, outputs, app_root=root)

            self.assertEqual("degraded", report["summary"]["status"])
            self.assertEqual(1, report["summary"]["fallback_cases"])
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
            self.assertFalse(report["runtime"]["ready"])
            self.assertTrue((outputs / "final_results" / "only_Pred.json").is_file())


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
