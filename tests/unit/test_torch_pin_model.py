from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import BBox
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.protocols import PinProvider
from pcbparse.perception.torch_pin_adapter import create_pin_provider
from pcbparse.training.torch_component_pin import PinKeypointDataset, pin_keypoint_collate
from pcbparse.training.torch_pin_model import (
    PinCheckpointError,
    PinLossConfig,
    TinyPinHeatmapNet,
    compute_pin_metrics,
    decode_pin_heatmaps,
    export_pin_model,
    load_pin_checkpoint,
    pin_eval_step,
    pin_losses,
    pin_train_step,
    save_pin_checkpoint,
)


try:
    import torch
except ImportError:  # core runtime intentionally keeps torch optional
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
LINEAGE = {
    "binding_sha256": "a" * 64,
    "split_manifest_sha256": "b" * 64,
}


@unittest.skipUnless(torch is not None, "optional PyTorch runtime is not installed")
class TorchPinModelTests(unittest.TestCase):
    def _batch(self, batch_size: int = 2) -> tuple[object, dict[str, object]]:
        images = torch.rand(batch_size, 3, 64, 64)
        heatmap = torch.zeros(batch_size, 1, 16, 16)
        offset = torch.zeros(batch_size, 2, 16, 16)
        orientation = torch.zeros(batch_size, 2, 16, 16)
        mask = torch.zeros(batch_size, 1, 16, 16)
        points = []
        visibility = []
        for index in range(batch_size):
            x, y = 5 + index, 6
            heatmap[index, 0, y, x] = 1.0
            offset[index, :, y, x] = torch.tensor([0.25, 0.5])
            orientation[index, :, y, x] = torch.tensor([1.0, 0.0])
            mask[index, 0, y, x] = 1.0
            points.append(torch.tensor([[x + 0.25, y + 0.5]]))
            visibility.append(torch.tensor([2]))
        return images, {
            "heatmap": heatmap,
            "offset": offset,
            "orientation": orientation,
            "regression_mask": mask,
            "tip_points_heatmap": points,
            "visibility": visibility,
        }

    def test_model_contract_loss_train_and_eval_steps(self) -> None:
        torch.manual_seed(7)
        model = TinyPinHeatmapNet(base_channels=4)
        batch = self._batch()
        outputs = model(batch[0])
        self.assertEqual((2, 1, 16, 16), tuple(outputs["heatmap_logits"].shape))
        self.assertEqual((2, 2, 16, 16), tuple(outputs["offset"].shape))
        self.assertEqual((2, 2, 16, 16), tuple(outputs["orientation"].shape))
        losses = pin_losses(outputs, batch[1])
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        before = next(model.parameters()).detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train = pin_train_step(
            model, batch, optimizer, max_grad_norm=5.0, device="cpu"
        )
        self.assertTrue(math.isfinite(train["loss"]))
        self.assertGreater(train["gradient_norm"], 0.0)
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
        evaluated = pin_eval_step(
            model, batch, score_threshold=0.999, heatmap_stride=4
        )
        for key in ("pck@2", "pck@3", "pck@5", "count_mae", "loss"):
            self.assertIn(key, evaluated)
            self.assertTrue(math.isfinite(evaluated[key]))

    def test_exact_subpixel_prediction_has_perfect_pck_and_count(self) -> None:
        _, target = self._batch(batch_size=1)
        logits = torch.full((1, 1, 16, 16), -20.0)
        offsets = torch.zeros((1, 2, 16, 16))
        orientations = torch.zeros((1, 2, 16, 16))
        logits[0, 0, 6, 5] = 20.0
        offsets[0, :, 6, 5] = torch.tensor([0.25, 0.5])
        orientations[0, :, 6, 5] = torch.tensor([1.0, 0.0])
        metrics = compute_pin_metrics(
            {
                "heatmap_logits": logits,
                "offset": offsets,
                "orientation": orientations,
            },
            target,
            score_threshold=0.9,
            heatmap_stride=4,
        )
        self.assertEqual(1.0, metrics["pck@2"])
        self.assertEqual(1.0, metrics["pck@3"])
        self.assertEqual(1.0, metrics["pck@5"])
        self.assertEqual(1.0, metrics["detection_precision@max_pck"])
        self.assertEqual(1.0, metrics["detection_recall@max_pck"])
        self.assertEqual(0.0, metrics["count_mae"])
        self.assertEqual(1.0, metrics["count_exact_accuracy"])

    def test_component_box_prior_keeps_only_the_boundary_band(self) -> None:
        logits = torch.full((1, 1, 8, 8), -20.0)
        offsets = torch.zeros((1, 2, 8, 8))
        orientations = torch.zeros((1, 2, 8, 8))
        logits[0, 0, 4, 2] = 6.0  # on the left boundary
        logits[0, 0, 4, 4] = 8.0  # higher-scoring, deep interior false peak

        decoded = decode_pin_heatmaps(
            {
                "heatmap_logits": logits,
                "offset": offsets,
                "orientation": orientations,
            },
            score_threshold=0.9,
            nms_radius=0.0,
            max_detections=1,
            box_rects=[(2.0, 2.0, 6.0, 6.0)],
            box_margin=0.5,
            box_prior_mode="boundary_band_v2",
        )

        self.assertEqual(1, len(decoded[0]))
        self.assertEqual((2, 4), decoded[0][0]["anchor"])

        legacy = decode_pin_heatmaps(
            {
                "heatmap_logits": logits,
                "offset": offsets,
                "orientation": orientations,
            },
            score_threshold=0.9,
            nms_radius=0.0,
            max_detections=1,
            box_rects=[(2.0, 2.0, 6.0, 6.0)],
            box_margin=0.5,
            box_prior_mode="filled_rect_v1",
        )
        self.assertEqual((4, 4), legacy[0][0]["anchor"])

    def test_count_loss_penalizes_excess_heatmap_mass(self) -> None:
        torch.manual_seed(3)
        _, target = self._batch(batch_size=1)
        shape = (1, 1, 16, 16)
        diffuse = {
            "heatmap_logits": torch.full(shape, -2.0),
            "offset": torch.zeros((1, 2, 16, 16)),
            "orientation": torch.zeros((1, 2, 16, 16)),
        }
        sharp = {
            "heatmap_logits": torch.full(shape, -20.0),
            "offset": torch.zeros((1, 2, 16, 16)),
            "orientation": torch.zeros((1, 2, 16, 16)),
        }
        for row in range(len(target["tip_points_heatmap"][0])):
            x = int(target["tip_points_heatmap"][0][row][0])
            y = int(target["tip_points_heatmap"][0][row][1])
            sharp["heatmap_logits"][0, 0, y, x] = 20.0
        config = PinLossConfig(count_loss_weight=1.0, count_target_multiplier=1.0)
        diffuse_losses = pin_losses(diffuse, target, config=config)
        sharp_losses = pin_losses(sharp, target, config=config)
        self.assertGreater(float(diffuse_losses["count_loss"]), 0.0)
        self.assertLess(float(sharp_losses["count_loss"]), 1e-5)

    def test_checkpoint_roundtrip_and_lineage_fail_closed(self) -> None:
        model = TinyPinHeatmapNet(base_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin.ckpt"
            report = save_pin_checkpoint(
                path,
                model,
                optimizer=optimizer,
                epoch=3,
                global_step=17,
                lineage=LINEAGE,
                metrics={"pck@5": 0.75},
            )
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), report["sha256"])
            restored, payload = load_pin_checkpoint(
                path, expected_lineage=LINEAGE, map_location="cpu"
            )
            self.assertEqual(3, payload["epoch"])
            sample = torch.rand(1, 3, 32, 32)
            expected = model(sample)
            actual = restored(sample)
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assertTrue(torch.equal(expected[name], actual[name]))
            with self.assertRaisesRegex(PinCheckpointError, "differs"):
                load_pin_checkpoint(
                    path,
                    expected_lineage={
                        "binding_sha256": "c" * 64,
                        "split_manifest_sha256": "b" * 64,
                    },
                )

    def test_torchscript_and_onnx_exports_match_eager(self) -> None:
        model = TinyPinHeatmapNet(base_channels=2).eval()
        samples = [torch.rand(1, 3, 32, 40), torch.rand(2, 3, 40, 48)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripted = export_pin_model(
                model,
                root / "pin.ts",
                samples,
                export_format="torchscript",
                tolerance=1e-5,
            )
            self.assertLessEqual(scripted["maximum_absolute_error"], 1e-5)
            try:
                import onnxruntime  # noqa: F401
                import onnx  # noqa: F401
            except ImportError:
                self.skipTest("ONNX export dependencies are unavailable")
            onnx_report = export_pin_model(
                model,
                root / "pin.onnx",
                samples,
                export_format="onnx",
                tolerance=1e-4,
            )
            self.assertLessEqual(onnx_report["maximum_absolute_error"], 1e-4)

    def test_torchscript_adapter_restores_local_geometry_without_semantics(self) -> None:
        class FixedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                heatmap = torch.full((1, 1, 8, 8), -20.0)
                heatmap[0, 0, 2, 3] = 20.0
                offset = torch.zeros((1, 2, 8, 8))
                offset[0, :, 2, 3] = torch.tensor([0.25, 0.5])
                orientation = torch.zeros((1, 2, 8, 8))
                orientation[0, :, 2, 3] = torch.tensor([1.0, 0.0])
                self.register_buffer("fixed_heatmap", heatmap)
                self.register_buffer("fixed_offset", offset)
                self.register_buffer("fixed_orientation", orientation)

            def forward(self, image: object) -> dict[str, object]:
                batch = image.shape[0]
                return {
                    "heatmap_logits": self.fixed_heatmap.expand(batch, -1, -1, -1),
                    "offset": self.fixed_offset.expand(batch, -1, -1, -1),
                    "orientation": self.fixed_orientation.expand(batch, -1, -1, -1),
                }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.ts"
            export_pin_model(
                FixedModel(), path, torch.zeros(1, 3, 32, 32), export_format="torchscript"
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            provider = create_pin_provider(
                path,
                {
                    "task": "pin",
                    "sha256": digest,
                    "version": "fixed-pin-v1",
                    "artifact_format": "torchscript",
                    "device": "cpu",
                    "input_size": [32, 32],
                    "score_threshold": 0.9,
                    "base_length_pixels": 8,
                },
            )
            self.assertIsInstance(provider, PinProvider)
            view = ImageView(
                view_id="component:C1:full",
                image=Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)),
                transform=TransformChain(
                    [AffineTransform.identity("component:C1:full")]
                ),
            )
            component_view = ComponentView(
                component_id="C1",
                component_bbox=BBox(x1=0, y1=0, x2=32, y2=32),
                full_roi=view,
            )
            wide_view = ImageView(
                view_id="component:C2:full",
                image=Image.fromarray(np.full((32, 64, 3), 255, dtype=np.uint8)),
                transform=TransformChain(
                    [AffineTransform.identity("component:C2:full")]
                ),
            )
            wide_component = ComponentView(
                component_id="C2",
                component_bbox=BBox(x1=0, y1=0, x2=64, y2=32),
                full_roi=wide_view,
            )
            padding_peak_view = ImageView(
                view_id="component:C3:full",
                image=Image.fromarray(np.full((16, 64, 3), 255, dtype=np.uint8)),
                transform=TransformChain(
                    [AffineTransform.identity("component:C3:full")]
                ),
            )
            padding_peak_component = ComponentView(
                component_id="C3",
                component_bbox=BBox(x1=0, y1=0, x2=64, y2=16),
                full_roi=padding_peak_view,
            )
            pins = provider.predict(
                [component_view, wide_component, padding_peak_component], None
            )
            # C3's only peak is in the white top letterbox padding and must not
            # become a geometric Pin candidate.
            self.assertEqual(2, len(pins))
            pin = next(item for item in pins if item.component_id == "C1")
            self.assertEqual("C1", pin.component_id)
            self.assertAlmostEqual(13.0, pin.point.x, places=5)
            self.assertAlmostEqual(10.0, pin.point.y, places=5)
            self.assertAlmostEqual(5.0, pin.base.x, places=5)
            self.assertAlmostEqual(10.0, pin.base.y, places=5)
            self.assertEqual("R", pin.orientation)
            self.assertIsNone(pin.number)
            self.assertEqual("", pin.name)
            self.assertEqual(["component:C1:full"], [item.view_id for item in pin.evidence])
            wide_pin = next(item for item in pins if item.component_id == "C2")
            self.assertAlmostEqual(26.0, wide_pin.point.x, places=5)
            self.assertAlmostEqual(4.0, wide_pin.point.y, places=5)
            self.assertAlmostEqual(18.0, wide_pin.base.x, places=5)
            self.assertEqual("R", wide_pin.orientation)

    def test_real_materialized_dataset_sample_forward_smoke(self) -> None:
        index = (
            ROOT
            / "artifacts"
            / "training"
            / "materialized"
            / "dataset300-v2"
            / "component-pin"
            / "pin_index.jsonl"
        )
        source = ROOT / "Dataset300" / "Dataset300"
        if not index.is_file() or not source.is_dir():
            self.skipTest("official materialized Pin dataset is not present")
        dataset = PinKeypointDataset(
            index,
            source,
            view_mode="roi",
            sample_ids=["0001"],
            include_negative=False,
        )
        self.assertGreater(len(dataset), 0)
        batch = pin_keypoint_collate([dataset[0]])
        model = TinyPinHeatmapNet(base_channels=2).eval()
        with torch.no_grad():
            losses = pin_losses(model(batch[0]), batch[1])
        self.assertTrue(math.isfinite(float(losses["loss"])))
        self.assertEqual(
            tuple(batch[1]["heatmap"].shape),
            (1, 1, 128, 128),
        )


if __name__ == "__main__":
    unittest.main()
