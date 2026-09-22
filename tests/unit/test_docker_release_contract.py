from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
AGENT_SYSTEM = ROOT / "docker" / "agent_system"
EDARACE_REQUIREMENTS = ROOT / "requirements.edarace.txt"


class DockerReleaseContractTests(unittest.TestCase):
    def test_image_uses_official_one_shot_contract(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("FROM edarace-agent-base:ubuntu24", text)
        self.assertIn('ENTRYPOINT ["/submission/agent_system"]', text)
        self.assertIn("COPY docker/agent_system /submission/agent_system", text)
        self.assertIn("PCBPARSE_APP_ROOT=/submission/app", text)
        self.assertNotIn("EXPOSE 8080", text)
        self.assertNotIn("HEALTHCHECK", text)
        self.assertNotIn('CMD ["python", "-m", "pcbparse.cli", "serve"', text)
        self.assertNotIn("/data/input", text)
        self.assertNotIn("/data/output", text)

    def test_image_keeps_model_lineage_and_official_cuda_runtime(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("models/models.lock.json", text)
        self.assertIn("models/releases/v8-portable", text)
        self.assertIn("require_lineage=True", text)
        self.assertIn("configs/pipeline.edarace.yaml", text)
        self.assertIn("cuda_nvrtc/lib", text)
        self.assertIn("cudnn/lib", text)

        requirements = EDARACE_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("pydantic==2.13.4", requirements)
        self.assertIn("onnxruntime-gpu==1.21.1", requirements)
        self.assertNotIn("torch==", requirements)
        self.assertNotIn("torchvision==", requirements)

    def test_entrypoint_forwards_official_arguments_without_eval(self) -> None:
        text = AGENT_SYSTEM.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/usr/bin/env bash\n"))
        self.assertIn('exec python3 -m pcbparse.submission_agent "$@"', text)
        self.assertNotIn("eval ", text)
        self.assertNotIn("LLM_API_KEY=", text)


if __name__ == "__main__":
    unittest.main()
