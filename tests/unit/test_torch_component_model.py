from __future__ import annotations

import gc
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.io import load_image
from pcbparse.perception.torch_component_adapter import _entry_value, factory
from pcbparse.training.torch_component_model import (
    TorchComponentArtifactError,
    TorchComponentDependencyError,
    build_model,
    evaluate_map50,
    export_model,
    load_checkpoint,
    normalize_category_table,
    save_checkpoint,
    train_one_epoch,
    verify_export_consistency,
)


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


class ComponentAdapterEntryTests(unittest.TestCase):
    def test_nested_inference_is_honoured_and_conflicts_are_rejected(self) -> None:
        self.assertEqual(
            0.4,
            _entry_value(
                {"inference": {"score_threshold": 0.4}},
                "score_threshold",
                default=0.25,
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _entry_value(
                {
                    "score_threshold": 0.3,
                    "inference": {"score_threshold": 0.4},
                },
                "score_threshold",
                default=0.25,
            )
ONNX_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


class TorchComponentOptionalDependencyTests(unittest.TestCase):
    def test_modules_import_without_torch_and_operation_fails_closed(self) -> None:
        if TORCH_AVAILABLE:
            self.skipTest("no-torch dependency gate is exercised by the bundled runtime")
        with self.assertRaisesRegex(TorchComponentDependencyError, "venv-train"):
            build_model(2)

    def test_sparse_official_ids_map_to_contiguous_foreground_labels(self) -> None:
        categories = normalize_category_table(
            [
                {"id": 35, "name": "Motor", "supercategory": "pcb_component"},
                {"id": 0, "name": "V", "supercategory": "pcb_component"},
            ]
        )
        self.assertEqual(
            [
                (0, 1, 1, "V", "0"),
                (35, 2, 2, "Motor", "35"),
            ],
            [
                (
                    row["official_id"],
                    row["model_label"],
                    row["id"],
                    row["name"],
                    row["source_category_id"],
                )
                for row in categories
            ],
        )

    def test_materialized_mapping_and_metadata_are_preserved(self) -> None:
        categories = normalize_category_table(
            [
                {
                    "id": 1,
                    "name": "V",
                    "source_category_id": "0",
                    "supercategory": "pcb_component",
                },
                {
                    "id": 2,
                    "name": "Motor",
                    "source_category_id": "35",
                    "supercategory": "pcb_component",
                },
            ]
        )
        self.assertEqual(0, categories[0]["official_id"])
        self.assertEqual(1, categories[0]["model_label"])
        self.assertEqual("pcb_component", categories[0]["supercategory"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch tests require .venv-train")
class TorchComponentMetricAndExportTests(unittest.TestCase):
    def test_class_aware_one_to_one_ap50_uses_global_score_ranking(self) -> None:
        import torch

        class DeterministicDetector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images: list[object]) -> list[dict[str, object]]:
                rows = []
                for image in images:
                    device = image.device
                    rows.append(
                        {
                            "boxes": torch.tensor(
                                [
                                    [4.0, 4.0, 20.0, 20.0],
                                    [4.0, 4.0, 20.0, 20.0],
                                    [4.0, 4.0, 20.0, 20.0],
                                ],
                                device=device,
                            ),
                            "scores": torch.tensor([0.9, 0.8, 0.7], device=device),
                            "labels": torch.tensor([1, 1, 2], device=device),
                        }
                    )
                return rows

        image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        metrics = evaluate_map50(
            DeterministicDetector(),
            [(image, [target])],
            "cpu",
            category_table=[{"id": 0, "name": "V"}, {"id": 35, "name": "Motor"}],
        )
        self.assertAlmostEqual(1.0, metrics["map50"], places=7)
        self.assertEqual(1, metrics["true_positives"])
        self.assertEqual(2, metrics["false_positives"])
        self.assertEqual(0, metrics["false_negatives"])
        self.assertEqual([2], metrics["classes_without_ground_truth"])
        self.assertIsNone(metrics["per_class"]["2"]["ap50"])
        self.assertEqual(0, metrics["per_class"]["1"]["official_category_id"])
        self.assertEqual(35, metrics["per_class"]["2"]["official_category_id"])

    def test_torchscript_export_has_native_output_consistency(self) -> None:
        import torch

        class ExportableDetector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images: list[object]) -> list[dict[str, object]]:
                image = images[0]
                zero = image.sum() * 0.0 + self.bias
                return [
                    {
                        "boxes": torch.tensor(
                            [[2.0, 3.0, 12.0, 15.0]], device=image.device
                        )
                        + zero,
                        "scores": torch.tensor([0.75], device=image.device) + zero,
                        "labels": torch.tensor([1], dtype=torch.int64, device=image.device),
                    }
                ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "component.ts"
            image = torch.rand((1, 3, 24, 24), dtype=torch.float32)
            model = ExportableDetector().eval()
            artifact = export_model(model, path, image)
            self.assertTrue(path.is_file())
            self.assertTrue(Path(artifact["sidecar"]).is_file())
            result = verify_export_consistency(model, path, image)
            self.assertTrue(result["consistent"], result)
            self.assertEqual(1, result["detections"])

    @unittest.skipUnless(ONNX_RUNTIME_AVAILABLE, "ONNX smoke requires onnx + onnxruntime")
    def test_onnx_export_has_native_output_consistency(self) -> None:
        import torch

        class ExportableDetector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images: list[object]) -> list[dict[str, object]]:
                image = images[0]
                zero = image.sum() * 0.0 + self.bias
                return [
                    {
                        "boxes": torch.tensor(
                            [[1.0, 1.0, 10.0, 10.0]], device=image.device
                        )
                        + zero,
                        "scores": torch.tensor([0.6], device=image.device) + zero,
                        "labels": torch.tensor([1], dtype=torch.int64, device=image.device),
                    }
                ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "component.onnx"
            image = torch.rand((1, 3, 24, 24), dtype=torch.float32)
            model = ExportableDetector().eval()
            export_model(model, path, image, opset_version=18)
            result = verify_export_consistency(model, path, image)
            self.assertTrue(result["consistent"], result)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch tests require .venv-train")
class TorchComponentRealDetectorTests(unittest.TestCase):
    def test_coco_pretraining_loads_compatible_layers_and_audits_head_shapes(self) -> None:
        import torch

        categories = [{"id": 0, "name": "V", "supercategory": "pcb_component"}]
        source = build_model(num_classes=3, detector_kwargs={"min_size": 64, "max_size": 64})
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "source.pth"
            torch.save(source.state_dict(), state_path)
            target = build_model(
                category_table=categories,
                pretrained_state_path=state_path,
                strict_pretrained=False,
                detector_kwargs={"min_size": 64, "max_size": 64},
            )

        audit = target._pcbparse_pretrained_incompatibility
        mismatches = {item["key"] for item in audit["shape_mismatch"]}
        self.assertGreater(audit["loaded_keys"], 0)
        self.assertIn("roi_heads.box_predictor.cls_score.weight", mismatches)
        self.assertIn("roi_heads.box_predictor.bbox_pred.weight", mismatches)
        self.assertEqual(
            source.backbone.body["0"][0].weight.shape,
            target.backbone.body["0"][0].weight.shape,
        )
        self.assertTrue(
            torch.equal(
                source.backbone.body["0"][0].weight,
                target.backbone.body["0"][0].weight,
            )
        )

    def test_official_pretraining_remaps_legacy_fpn_and_rpn_keys(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        weights = (
            repository
            / "models"
            / "pretrained"
            / "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
        )
        if not weights.is_file():
            self.skipTest("official cached torchvision weights are unavailable")
        model = build_model(
            category_table=({"id": 0, "name": "V"}, {"id": 35, "name": "Motor"}),
            pretrained_state_path=weights,
            strict_pretrained=False,
            detector_kwargs={"min_size": 64, "max_size": 64},
        )
        audit = model._pcbparse_pretrained_incompatibility
        self.assertEqual(10, len(audit["remapped_keys"]))
        self.assertEqual([], audit["unexpected_keys"])
        self.assertFalse(
            any(key.startswith("backbone.fpn") for key in audit["missing_keys"])
        )

    def test_real_faster_rcnn_train_checkpoint_load_and_adapter(self) -> None:
        import torch

        torch.manual_seed(20260813)
        categories = [
            {"id": 0, "name": "V", "supercategory": "pcb_component"},
            {"id": 35, "name": "Motor", "supercategory": "pcb_component"},
        ]
        detector_kwargs = {
            "min_size": 64,
            "max_size": 64,
            "rpn_pre_nms_top_n_train": 40,
            "rpn_post_nms_top_n_train": 20,
            "rpn_pre_nms_top_n_test": 20,
            "rpn_post_nms_top_n_test": 10,
            "box_detections_per_img": 5,
        }
        model = build_model(category_table=categories, detector_kwargs=detector_kwargs)
        # A real detection-head/RPN optimization step; freezing only the
        # backbone keeps this CPU smoke bounded without replacing the model.
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
            momentum=0.0,
        )
        image = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        image[:, :, 12:45, 10:47] = 1.0
        image[:, :, 18:50, 48:62] = 0.5
        target = {
            "boxes": torch.tensor(
                [[10.0, 12.0, 47.0, 45.0], [48.0, 18.0, 62.0, 50.0]],
                dtype=torch.float32,
            ),
            # label 1 is official category 0 (V); label 2 is sparse official
            # category 35 (Motor).  Neither may ever become background.
            "labels": torch.tensor([1, 2], dtype=torch.int64),
            "image_id": torch.tensor([1], dtype=torch.int64),
            "area": torch.tensor([1221.0, 448.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
        }
        training = train_one_epoch(
            model,
            [(image, [target])],
            optimizer,
            "cpu",
            grad_clip_norm=10.0,
        )
        self.assertEqual(1, training["num_examples"])
        self.assertGreater(training["losses"]["loss_total"], 0.0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "component.ckpt"
            artifact = save_checkpoint(
                checkpoint_path,
                model,
                category_table=categories,
                lineage={
                    "dataset_binding_sha256": "a" * 64,
                    "training_config_sha256": "b" * 64,
                    "split": "train",
                },
                optimizer=optimizer,
                epoch=1,
                metrics={"train_loss": training["losses"]["loss_total"]},
            )
            self.assertTrue(checkpoint_path.is_file())
            loaded = load_checkpoint(
                checkpoint_path,
                expected_sha256=artifact["sha256"],
                device="cpu",
            )
            self.assertEqual(1, loaded.epoch)
            self.assertEqual(
                [(0, 1, "V"), (35, 2, "Motor")],
                [
                    (row["official_id"], row["model_label"], row["name"])
                    for row in loaded.category_table
                ],
            )
            self.assertEqual(artifact["sha256"], loaded.checkpoint_sha256)
            for parameter in loaded.model.backbone.parameters():
                parameter.requires_grad_(False)
            resume_optimizer = torch.optim.SGD(
                [
                    parameter
                    for parameter in loaded.model.parameters()
                    if parameter.requires_grad
                ],
                lr=1e-3,
                momentum=0.0,
            )
            self.assertIsNotNone(loaded.optimizer_state)
            resume_optimizer.load_state_dict(loaded.optimizer_state)
            del resume_optimizer
            del loaded
            del model
            del optimizer
            gc.collect()

            image_path = root / "synthetic.png"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(image_path)
            view = load_image(image_path, view_id="tile_000_000")
            adapter = factory(
                checkpoint_path,
                {
                    "task": "component",
                    "sha256": artifact["sha256"],
                    "device": "cpu",
                    "score_threshold": 0.0,
                    "max_detections": 3,
                },
            )
            first = adapter.predict(view, None)
            second = adapter.predict(view, None)
            self.assertGreater(len(first), 0)
            self.assertLessEqual(len(first), 3)
            self.assertEqual(
                [item.model_dump(mode="json") for item in first],
                [item.model_dump(mode="json") for item in second],
            )
            for component in first:
                self.assertIn(component.category, {"V", "Motor"})
                self.assertIsNone(component.key)
                self.assertEqual("", component.name)
                self.assertEqual("", component.value)
                self.assertGreaterEqual(component.bbox.x1, 0.0)
                self.assertLessEqual(component.bbox.x2, 64.0)
                self.assertEqual("tile_000_000", component.evidence[0].view_id)
                self.assertEqual(
                    "image_view_local_xyxy",
                    component.evidence[0].metadata["coordinate_system"],
                )
                official_id = component.evidence[0].metadata[
                    "official_category_id"
                ]
                model_label = component.evidence[0].metadata["model_label"]
                expected = {"V": (0, 1), "Motor": (35, 2)}[component.category]
                self.assertEqual(expected, (official_id, model_label))

    def test_legacy_v1_checkpoint_fails_closed(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.ckpt"
            torch.save(
                {
                    "format": "pcbparse-torch-component-v1",
                    "category_table": [{"id": 1, "name": "V"}],
                    "model_state": {"placeholder": torch.zeros(())},
                },
                path,
            )
            with self.assertRaisesRegex(
                TorchComponentArtifactError, "lacks the explicit official_id/model_label"
            ):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
