from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"


class DockerReleaseContractTests(unittest.TestCase):
    def test_default_command_is_exact_competition_production_service(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        matches = re.findall(r"(?m)^CMD\s+(\[.*\])\s*$", text)
        self.assertEqual(1, len(matches))
        command = json.loads(matches[0])

        self.assertEqual(["python", "-m", "pcbparse.cli", "serve"], command[:4])
        pairs = dict(zip(command[4::2], command[5::2]))
        self.assertEqual("0.0.0.0", pairs["--host"])
        self.assertEqual("8080", pairs["--port"])
        self.assertEqual("/data/input", pairs["--input-root"])
        self.assertEqual("/data/output", pairs["--output-root"])
        self.assertEqual("/app/configs/pipeline.yaml", pairs["--config"])
        self.assertEqual(
            "/app/configs/profiles/competition.yaml", pairs["--profile"]
        )
        self.assertEqual("production", pairs["--provider"])
        self.assertEqual("/app/models/models.lock.json", pairs["--models-lock"])
        self.assertNotIn("--allow-unready-debug", command)

    def test_image_keeps_fail_closed_healthcheck_and_pinned_runtime(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime", text
        )
        self.assertIn("HEALTHCHECK", text)
        self.assertIn("http://127.0.0.1:8080/ready", text)
        self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]', text)
        self.assertNotIn("dataset_golden", text)
        self.assertNotIn("dataset_debug", text)
        self.assertNotIn("heuristic", text)


if __name__ == "__main__":
    unittest.main()
