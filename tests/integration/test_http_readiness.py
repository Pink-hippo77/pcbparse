from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from pcbparse.config import load_config
from pcbparse.pipeline.providers import EmptyImageProvider
from pcbparse.service.http_api import APIServerConfig, create_server


class HTTPReadinessTests(unittest.TestCase):
    def test_client_cannot_bypass_server_side_readiness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            Image.new("RGB", (20, 10), "white").save(input_root / "board.png")
            lock = root / "models.lock.json"
            tasks = []
            for task in ("component", "junction", "ocr", "pin", "wire"):
                weight = root / f"{task}.bin"
                weight.write_bytes(f"fixture-{task}".encode())
                tasks.append(
                    {
                        "task": task,
                        "path": weight.name,
                        "sha256": "0" * 64,
                        "status": "blessed",
                        "blessed": True,
                        "version": "1.0",
                        "license": "test-only",
                        "precision": "fp32",
                    }
                )
            lock.write_text(
                json.dumps({"schema_version": "1.0", "models": tasks}),
                encoding="utf-8",
            )
            server = create_server(
                "127.0.0.1",
                0,
                APIServerConfig(
                    input_root=input_root,
                    output_root=output_root,
                    pipeline_config=load_config("configs/pipeline.yaml"),
                    provider=EmptyImageProvider(),
                    official_profile="configs/official_schema.yaml",
                    models_lock=lock,
                    allow_unready_debug=False,
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(f"{base}/ready", timeout=5)
                self.assertEqual(503, caught.exception.code)
                report = json.loads(caught.exception.read())
                self.assertIn("sha256_mismatch", {item["code"] for item in report["issues"]})

                request = urllib.request.Request(
                    f"{base}/v1/infer",
                    data=json.dumps(
                        {
                            "input": "board.png",
                            "output": "result.json",
                            "allow_unready_debug": True,
                        }
                    ).encode(),
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as infer_error:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(503, infer_error.exception.code)
                self.assertFalse((output_root / "result.json").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
