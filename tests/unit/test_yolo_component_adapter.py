from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.io import load_image
from pcbparse.perception.yolo_component_adapter import (
    YoloComponentAdapter,
    YoloComponentArtifactError,
    factory,
)


SHA = "a" * 64
CATEGORIES = [
    {
        "yolo_class_id": 0,
        "source_category_id": "0",
        "source_category_name": "V",
    },
    {
        "yolo_class_id": 1,
        "source_category_id": "35",
        "source_category_name": "Motor",
    },
]


class _Tensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def tolist(self) -> object:
        return self._values


class _Model:
    task = "detect"
    names = {0: "V", 1: "Motor"}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=_Tensor(
                [
                    [-3.0, 2.0, 20.0, 22.0],
                    [40.0, 10.0, 80.0, 50.0],
                    [5.0, 5.0, 5.0, 8.0],
                    [1.0, 1.0, 4.0, 4.0],
                ]
            ),
            conf=_Tensor([0.8, 0.9, 0.7, 0.1]),
            cls=_Tensor([0.0, 1.0, 0.0, 0.0]),
        )
        return [SimpleNamespace(boxes=boxes)]


class _BatchModel(_Model):
    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        sources = kwargs["source"]
        count = len(sources) if isinstance(sources, list) else 1
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=_Tensor([[2.0, 3.0, 20.0, 22.0]]),
                    conf=_Tensor([0.8]),
                    cls=_Tensor([0.0]),
                )
            )
            for _ in range(count)
        ]


class _Context:
    def __init__(self) -> None:
        self.model_versions: dict[str, str] = {}
        self.stages: list[str] = []
        self.time_budget = SimpleNamespace(check=self._check)

    def _check(self, *, stage: str) -> None:
        self.stages.append(stage)


class YoloComponentAdapterTests(unittest.TestCase):
    def _view(self, root: Path):
        path = root / "image.png"
        Image.new("RGB", (64, 48), color=(255, 255, 255)).save(path)
        return load_image(path, view_id="tile_000_000")

    def test_predict_preserves_category_and_local_coordinate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _Model()
            adapter = YoloComponentAdapter(
                model,
                checkpoint_sha256=SHA,
                category_table=CATEGORIES,
                device="cpu",
                score_threshold=0.25,
                nms_iou=0.55,
                max_detections=3,
                imgsz=640,
            )
            context = _Context()
            first = adapter.predict(self._view(Path(temporary)), context)
            second = adapter.predict(self._view(Path(temporary)), context)

        self.assertEqual(["Motor", "V"], [item.category for item in first])
        self.assertEqual(
            [item.model_dump(mode="json") for item in first],
            [item.model_dump(mode="json") for item in second],
        )
        self.assertEqual([0.0, 2.0, 20.0, 22.0], list(first[1].bbox.model_dump().values()))
        self.assertEqual([40.0, 10.0, 64.0, 48.0], list(first[0].bbox.model_dump().values()))
        self.assertEqual(
            (35, 1),
            (
                first[0].evidence[0].metadata["official_category_id"],
                first[0].evidence[0].metadata["model_label"],
            ),
        )
        self.assertEqual(
            "image_view_local_xyxy",
            first[0].evidence[0].metadata["coordinate_system"],
        )
        self.assertEqual(
            ["component_detector:before", "component_detector:after"] * 2,
            context.stages,
        )
        self.assertEqual(adapter.version, context.model_versions["component"])
        self.assertEqual(0.25, model.calls[0]["conf"])
        self.assertEqual(0.55, model.calls[0]["iou"])
        self.assertEqual(640, model.calls[0]["imgsz"])

    def test_predict_many_uses_one_model_batch_and_preserves_view_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "image.png"
            Image.new("RGB", (64, 48), color=(255, 255, 255)).save(path)
            views = [
                load_image(path, view_id="tile_000_000"),
                load_image(path, view_id="tile_000_001"),
            ]
            model = _BatchModel()
            adapter = YoloComponentAdapter(
                model,
                checkpoint_sha256=SHA,
                category_table=CATEGORIES,
                device="cpu",
                score_threshold=0.25,
                nms_iou=0.55,
                max_detections=3,
                imgsz=640,
            )
            context = _Context()
            predictions = adapter.predict_many(views, context)

        self.assertEqual(1, len(model.calls))
        self.assertEqual(2, model.calls[0]["batch"])
        self.assertEqual(2, len(model.calls[0]["source"]))
        self.assertEqual([1, 1], [len(items) for items in predictions])
        self.assertEqual(
            ["tile_000_000", "tile_000_001"],
            [items[0].evidence[0].view_id for items in predictions],
        )
        self.assertEqual(
            ["component_detector:before", "component_detector:after"],
            context.stages,
        )
        self.assertEqual(adapter.version, context.model_versions["component"])

    def test_category_binding_rejects_name_or_label_drift(self) -> None:
        model = _Model()
        model.names = {0: "V", 1: "Wrong"}
        with self.assertRaisesRegex(YoloComponentArtifactError, "names do not match"):
            YoloComponentAdapter(
                model,
                checkpoint_sha256=SHA,
                category_table=CATEGORIES,
                device="cpu",
                score_threshold=0.25,
                nms_iou=0.55,
                max_detections=3,
                imgsz=640,
            )
        with self.assertRaisesRegex(YoloComponentArtifactError, "contiguous"):
            YoloComponentAdapter(
                _Model(),
                checkpoint_sha256=SHA,
                category_table=[CATEGORIES[0], {**CATEGORIES[1], "yolo_class_id": 2}],
                device="cpu",
                score_threshold=0.25,
                nms_iou=0.55,
                max_detections=3,
                imgsz=640,
            )

    def test_per_class_threshold_filters_after_shared_population_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = _Model()
            adapter = YoloComponentAdapter(
                model,
                checkpoint_sha256=SHA,
                category_table=CATEGORIES,
                device="cpu",
                score_threshold=0.25,
                nms_iou=0.55,
                max_detections=3,
                imgsz=640,
                per_class_score_thresholds={"V": 0.85},
            )
            predictions = adapter.predict(self._view(Path(temporary)), None)
        self.assertEqual(["Motor"], [item.category for item in predictions])
        self.assertEqual(0.25, model.calls[0]["conf"])
        self.assertEqual({"V": 0.85}, adapter.per_class_score_thresholds)
        self.assertEqual(
            0.25,
            predictions[0].evidence[0].metadata["effective_score_threshold"],
        )

        for thresholds, message in [
            ({"unknown": 0.5}, "unknown per-class"),
            ({"V": 0.2}, "cannot be below"),
        ]:
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, message):
                    YoloComponentAdapter(
                        _Model(),
                        checkpoint_sha256=SHA,
                        category_table=CATEGORIES,
                        device="cpu",
                        score_threshold=0.25,
                        nms_iou=0.55,
                        max_detections=3,
                        imgsz=640,
                        per_class_score_thresholds=thresholds,
                    )
    def test_factory_verifies_checkpoint_and_inline_category_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"fixture-yolo-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            module = SimpleNamespace(YOLO=lambda _: _Model())
            with patch(
                "pcbparse.perception.yolo_component_adapter._require_ultralytics",
                return_value=module,
            ):
                adapter = factory(
                    checkpoint,
                    {
                        "task": "component",
                        "sha256": digest,
                        "category_table": CATEGORIES,
                        "inference": {
                            "device": "cpu",
                            "score_threshold": 0.2,
                            "nms_iou": 0.55,
                            "max_detections": 100,
                            "imgsz": 640,
                        },
                    },
                )
                self.assertEqual(digest, adapter.checkpoint_sha256)
                with self.assertRaisesRegex(YoloComponentArtifactError, "SHA-256"):
                    factory(
                        checkpoint,
                        {
                            "task": "component",
                            "sha256": "0" * 64,
                            "category_table": CATEGORIES,
                        },
                    )

    def test_factory_loads_hash_bound_runtime_contract_from_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"packaged-yolo-checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            evidence = root / "evidence"
            evidence.mkdir()
            runtime_contract = {
                "schema_version": 1,
                "kind": "pcbparse_yolo_component_runtime_contract",
                "checkpoint_sha256": checkpoint_sha,
                "nms_iou": 0.55,
                "imgsz": 768,
                "per_class_score_thresholds": {"V": 0.85},
                "classes": CATEGORIES,
            }
            contract_path = evidence / "yolo_runtime_contract.json"
            contract_path.write_text(json.dumps(runtime_contract), encoding="utf-8")
            contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            manifest = {
                "task": "component",
                "artifact_path": "model.pt",
                "artifact_sha256": checkpoint_sha,
                "evidence_attachments": [
                    {
                        "kind": "yolo_component_runtime_contract_v1",
                        "path": "evidence/yolo_runtime_contract.json",
                        "sha256": contract_sha,
                    }
                ],
            }
            (root / "bundle.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            model = _Model()
            module = SimpleNamespace(YOLO=lambda _: model)
            with patch(
                "pcbparse.perception.yolo_component_adapter._require_ultralytics",
                return_value=module,
            ):
                adapter = factory(
                    checkpoint,
                    {
                        "task": "component",
                        "sha256": checkpoint_sha,
                        "inference": {
                            "device": "cpu",
                            "score_threshold": 0.2,
                            "max_detections": 100,
                        },
                    },
                )
                adapter.predict(self._view(root), None)
            self.assertEqual(0.55, model.calls[0]["iou"])
            self.assertEqual(768, model.calls[0]["imgsz"])
            self.assertEqual({"V": 0.85}, adapter.per_class_score_thresholds)

            runtime_contract["imgsz"] = 640
            contract_path.write_text(json.dumps(runtime_contract), encoding="utf-8")
            with patch(
                "pcbparse.perception.yolo_component_adapter._require_ultralytics",
                return_value=module,
            ):
                with self.assertRaisesRegex(
                    YoloComponentArtifactError, "runtime contract SHA-256 mismatch"
                ):
                    factory(
                        checkpoint,
                        {
                            "task": "component",
                            "sha256": checkpoint_sha,
                            "inference": {
                                "device": "cpu",
                                "score_threshold": 0.2,
                                "max_detections": 100,
                            },
                        },
                    )


if __name__ == "__main__":
    unittest.main()
