from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path

from PIL import Image

from pcbparse.config import load_config
from pcbparse.pipeline.providers import EmptyImageProvider
from pcbparse.service.http_api import APIServerConfig, create_server


class HTTPAPITests(unittest.TestCase):
    def test_health_and_infer_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            Image.new("RGB", (20, 10), "white").save(input_root / "board.png")
            server = create_server(
                "127.0.0.1",
                0,
                APIServerConfig(
                    input_root=input_root,
                    output_root=output_root,
                    pipeline_config=load_config("configs/pipeline.yaml"),
                    provider=EmptyImageProvider(),
                    official_profile="configs/official_schema.yaml",
                    allow_unready_debug=True,
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    health = json.loads(response.read())
                self.assertEqual("ok", health["status"])
                self.assertFalse(health["ready"])
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(f"{base}/ready", timeout=5)
                self.assertEqual(503, caught.exception.code)

                body = json.dumps({"input": "board.png", "output": "result.json"}).encode()
                request = urllib.request.Request(
                    f"{base}/v1/infer",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.loads(response.read())
                self.assertEqual("debug", result["status"])
                self.assertEqual("unready_debug", result["execution_mode"])
                self.assertTrue((output_root / "result.json").is_file())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
