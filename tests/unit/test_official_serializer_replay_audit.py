from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    IRState,
    ImageInfo,
    Pin,
    Point,
    WireSegment,
)


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_PATH = WORKSPACE / "scripts" / "audit_official_serializer_replay.py"
SPEC = importlib.util.spec_from_file_location("official_serializer_replay_audit", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {SCRIPT_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class OfficialSerializerReplayAuditTests(unittest.TestCase):
    def test_realistic_frozen_ir_replay_is_hash_bound_and_scope_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "dataset" / "case.png"
            annotation_path = root / "dataset" / "case.json"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 80), "white").save(image_path)
            image_sha256 = _sha(image_path)
            _write_json(annotation_path, {"canvas": {"width": 100, "height": 80}})

            ir = CircuitIR(
                state=IRState.SERIALIZED,
                image=ImageInfo(
                    image_id="case",
                    path=str(image_path),
                    width=100,
                    height=80,
                    sha256=image_sha256,
                ),
                components=[
                    Component(
                        internal_id="component:1",
                        key="R1",
                        category="R",
                        bbox=BBox(x1=10, y1=20, x2=30, y2=40),
                    )
                ],
                pins=[
                    Pin(
                        internal_id="pin:1",
                        component_id="component:1",
                        number="1",
                        point=Point(x=30, y=30),
                    )
                ],
                segments=[
                    WireSegment(
                        internal_id="segment:1",
                        start=Point(x=30, y=30),
                        end=Point(x=50, y=30),
                    )
                ],
                electrical_nets=[
                    ElectricalNet(
                        internal_id="net:1",
                        pins=["pin:1"],
                        segments=["segment:1"],
                    )
                ],
            )
            source_run = root / "source-run"
            ir_path = source_run / "circuit_ir" / "case.circuit-ir.json"
            ir_path.parent.mkdir(parents=True)
            ir_path.write_text(ir.model_dump_json(), encoding="utf-8")
            historical_path = source_run / "official" / "case.json"
            _write_json(
                historical_path,
                {
                    "components": {
                        "R1": {
                            "type": "resistor",
                            "name": None,
                            "value": None,
                            "bbox": [10, 20, 30, 40],
                        }
                    },
                    "pins": {},
                    "nets": {},
                },
            )
            source_lock = source_run / "source.models.lock.json"
            _write_json(source_lock, {"schema_version": "historical-fixture"})
            source_lock_sha256 = _sha(source_lock)
            manifest_path = source_run / "predictions.jsonl"
            _write_json(
                manifest_path,
                {
                    "case_id": "case",
                    "status": "ok",
                    "provider_kind": "production",
                    "sidecar_access": False,
                    "model_lock_sha256": source_lock_sha256,
                    "config_hash": "c" * 64,
                    "image_sha256": image_sha256,
                    "ir_path": "circuit_ir/case.circuit-ir.json",
                    "ir_sha256": _sha(ir_path),
                    "official_path": "official/case.json",
                    "official_sha256": _sha(historical_path),
                },
            )
            _write_json(
                source_run / "summary.json",
                {
                    "kind": "production-image-only-inference",
                    "accuracy_mode": "image-only",
                    "execution_mode": "production",
                    "provider_kind": "production",
                    "sidecar_access": False,
                    "status": "ok",
                    "all_successful": True,
                    "control_inputs": {"models_lock": str(source_lock)},
                    "prediction_manifest_sha256": _sha(manifest_path),
                    "prediction_set_sha256": "e" * 64,
                },
            )
            current_lock = root / "models" / "models.lock.json"
            _write_json(current_lock, {"schema_version": "fixture"})
            current_lock_before = _sha(current_lock)
            profile = root / "official_schema.json"
            profile.write_bytes(
                (WORKSPACE / "src" / "pcbparse" / "resources" / "official_schema.json").read_bytes()
            )

            keyword_arguments = {
                "workspace_root": root,
                "source_run": source_run,
                "source_image": image_path,
                "source_annotation": annotation_path,
                "current_models_lock": current_lock,
                "official_profile": profile,
                "replay_root": root / "replay",
                "coordinate_report": root / "coordinate.json",
                "case_id": "case",
            }
            first = AUDIT.build_and_write_evidence(**keyword_arguments)
            first_report_bytes = (root / "coordinate.json").read_bytes()
            second = AUDIT.build_and_write_evidence(**keyword_arguments)
            second_report_bytes = (root / "coordinate.json").read_bytes()
            current_lock_after = _sha(current_lock)

            outputs = [
                (root / "replay" / f"run{number}" / "case.json").read_bytes()
                for number in range(1, 4)
            ]
            official = json.loads(outputs[0])

        self.assertEqual(AUDIT.EVIDENCE_SCOPE, first["evidence_scope"])
        self.assertFalse(first["current_lock_inference_executed"])
        self.assertFalse(first["models_loaded"])
        self.assertNotEqual(
            first["source_inference_model_lock_sha256"], first["model_lock_sha256"]
        )
        self.assertEqual(0.0, first["max_roundtrip_error_px"])
        self.assertEqual(5, first["points_checked"])
        self.assertTrue(first["strict_schema"]["validated"])
        self.assertEqual(0, first["strict_schema"]["extra_fields"])
        self.assertEqual(0, first["strict_schema"]["dangling_references"])
        self.assertEqual(1, len(set(outputs)))
        self.assertEqual(["type", "Name", "value", "bbox"], list(official["components"]["R1"]))
        self.assertEqual([10.0, 40.0, 30.0, 60.0], official["components"]["R1"]["bbox"])
        self.assertEqual({"x": 30.0, "y": 50.0}, official["pins"]["R1"]["pin_1"]["point"])
        self.assertEqual(first_report_bytes, second_report_bytes)
        self.assertEqual(first["audit_payload_sha256"], second["audit_payload_sha256"])
        self.assertEqual(current_lock_before, current_lock_after)
        payload = dict(first)
        recorded_hash = payload.pop("audit_payload_sha256")
        self.assertEqual(recorded_hash, AUDIT._json_sha256(payload))


if __name__ == "__main__":
    unittest.main()
