from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.knowledge.model_catalog import load_model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_binding_is_required_and_tampering_rejected_after_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            original = json.dumps({"schema_version": 1, "model_names": ["LM385", "lm334"]}).encode()
            path.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            config = OCRAssociationConfig(component_model_catalog_path=str(path), component_model_catalog_sha256=digest)
            self.assertEqual(frozenset({"LM385", "LM334"}), load_model_catalog(str(path), digest))
            self.assertEqual(config, OCRAssociationConfig.coerce(config.to_dict()))
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_model_catalog(str(path), digest)
            with self.assertRaises(ValueError):
                OCRAssociationConfig(component_model_catalog_path=str(path))

    def test_wrong_schema_and_empty_names_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for document in ({"schema_version": 2, "model_names": ["LM385"]},
                             {"schema_version": 1, "model_names": []},
                             {"schema_version": 1, "model_names": [5]}):
                payload = json.dumps(document).encode()
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    load_model_catalog(str(path), hashlib.sha256(payload).hexdigest())
