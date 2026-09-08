from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pcbparse.io import ImageView
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.torch_wire_adapter import (
    TorchWireAdapter,
    TorchWireAdapterError,
    create_wire_provider,
)
from pcbparse.training.torch_wire_model import (
    TorchWireRuntimeError,
    WireCheckpointError,
    build_tiny_unet,
    compute_wire_loss,
    export_wire_model,
    load_wire_checkpoint,
    save_wire_checkpoint,
    verify_wire_export,
    wire_centerline_metrics,
    wire_eval_step,
    wire_train_step,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ONNX_RUNTIME_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None


class TorchWireLazyImportTests(unittest.TestCase):
    def test_import_is_lightweight_and_runtime_failure_is_actionable(self) -> None:
        self.assertNotIn("torch", sys.modules if not TORCH_AVAILABLE else {})
        with mock.patch(
            "pcbparse.training.torch_wire_model.importlib.import_module",
            side_effect=ImportError("fixture without torch"),
        ):
            with self.assertRaisesRegex(TorchWireRuntimeError, "PyTorch is required"):
                build_tiny_unet()

    def test_adapter_fails_before_runtime_loading_on_bad_artifact(self) -> None:
        with self.assertRaisesRegex(TorchWireAdapterError, "missing or empty"):
            TorchWireAdapter(REPOSITORY_ROOT / "does-not-exist.pt", {})


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training runtime is not installed")
class TorchWireModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        torch.manual_seed(17)

    def test_synthetic_train_eval_and_ignore_pixels_are_fully_masked(self) -> None:
        torch = self.torch
        model = build_tiny_unet(base_channels=4)
        image = torch.rand(2, 3, 25, 29)
        target = torch.zeros(2, 1, 25, 29)
        target[0, 0, 12, 3:26] = 1.0
        target[1, 0, 3:22, 14] = 1.0
        ignore = torch.zeros_like(target)
        ignore[:, :, 0:5, 0:8] = 1.0
        fixed_logits = model(image).detach()

        target_changed_only_inside_ignore = target.clone()
        target_changed_only_inside_ignore[ignore.bool()] = 1.0
        first = compute_wire_loss(fixed_logits, target, ignore)
        second = compute_wire_loss(
            fixed_logits, target_changed_only_inside_ignore, ignore
        )
        for name in ("loss", "bce", "dice", "cldice"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

        completely_ignored = torch.ones_like(ignore)
        ignored_logits = fixed_logits.clone().requires_grad_(True)
        zero_loss = compute_wire_loss(
            ignored_logits, target, completely_ignored
        )["loss"]
        self.assertEqual(0.0, float(zero_loss.detach()))
        zero_loss.backward()
        self.assertFalse(bool(ignored_logits.grad.abs().any()))

        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        before = model.head.weight.detach().clone()
        batch = {"image": image, "target": target, "ignore": ignore}
        train = wire_train_step(
            model,
            batch,
            optimizer,
            max_grad_norm=2.0,
            tolerance_px=1,
        )
        self.assertFalse(torch.equal(before, model.head.weight.detach()))
        self.assertTrue(all(math.isfinite(value) for value in train.values()))
        self.assertEqual(float((1.0 - ignore).sum()), train["valid_pixels"])
        self.assertIn("skeleton_f1", train)
        self.assertIn("connectivity_score", train)

        evaluated = wire_eval_step(model, batch, tolerance_px=1)
        self.assertTrue(all(math.isfinite(value) for value in evaluated.values()))
        self.assertGreaterEqual(evaluated["centerline_f1"], 0.0)
        self.assertLessEqual(evaluated["centerline_f1"], 1.0)

    def test_perfect_centerline_and_connectivity_metrics(self) -> None:
        torch = self.torch
        target = torch.zeros(1, 1, 21, 21)
        target[0, 0, 10, 2:19] = 1.0
        probabilities = target.clone()
        metrics = wire_centerline_metrics(
            probabilities,
            target,
            from_logits=False,
            tolerance_px=0,
        )
        self.assertAlmostEqual(1.0, metrics["centerline_precision"], places=6)
        self.assertAlmostEqual(1.0, metrics["centerline_recall"], places=6)
        self.assertAlmostEqual(1.0, metrics["centerline_f1"], places=6)
        self.assertAlmostEqual(1.0, metrics["connectivity_score"], places=6)
        self.assertEqual(1.0, metrics["predicted_components"])
        self.assertEqual(1.0, metrics["target_components"])
        self.assertEqual(17.0, metrics["precision_hits"])
        self.assertEqual(17.0, metrics["recall_hits"])
        self.assertEqual(17.0, metrics["predicted_skeleton_pixels"])
        self.assertEqual(17.0, metrics["target_skeleton_pixels"])
        self.assertEqual(1.0, metrics["connectivity_score_sum"])
        self.assertEqual(0.0, metrics["component_count_error_sum"])

    def test_component_error_is_per_example_and_cannot_cancel_across_batch(self) -> None:
        torch = self.torch
        target = torch.zeros(2, 1, 9, 9)
        target[:, 0, 4, 1:8] = 1.0
        prediction = torch.zeros_like(target)
        prediction[0, 0, 2, 1:3] = 1.0
        prediction[0, 0, 6, 5:8] = 1.0

        metrics = wire_centerline_metrics(
            prediction,
            target,
            from_logits=False,
            tolerance_px=0,
            skeleton_iterations=1,
        )

        # Aggregate predicted and target component totals are both two, but
        # each example is wrong by one.  Taking abs(total_pred-total_target)
        # would incorrectly hide both errors.
        self.assertEqual(2.0, metrics["predicted_components"])
        self.assertEqual(2.0, metrics["target_components"])
        self.assertEqual(2.0, metrics["component_count_error_sum"])
        self.assertEqual(1.0, metrics["component_count_error"])
        self.assertEqual(0.5, metrics["connectivity_score_sum"])
        self.assertEqual(0.25, metrics["connectivity_score"])

    def test_checkpoint_round_trip_preserves_weights_optimizer_and_lineage(self) -> None:
        torch = self.torch
        model = build_tiny_unet(base_channels=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        image = torch.rand(1, 3, 17, 19)
        target = torch.zeros(1, 1, 17, 19)
        target[:, :, 8, :] = 1.0
        wire_train_step(
            model,
            {"image": image, "target": target, "ignore": torch.zeros_like(target)},
            optimizer,
        )
        model.eval()
        with torch.inference_mode():
            expected = model(image).clone()
        lineage = {
            "dataset": "dataset300-v2",
            "run_id": "a" * 64,
            "split_manifest_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "wire.ckpt"
            info = save_wire_checkpoint(
                checkpoint,
                model,
                optimizer=optimizer,
                epoch=3,
                global_step=27,
                lineage=lineage,
                metrics={"skeleton_f1": 0.75},
            )
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), info.sha256)
            restored_model = build_tiny_unet(base_channels=4)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-3)
            loaded = load_wire_checkpoint(
                checkpoint,
                expected_lineage=lineage,
                model=restored_model,
                optimizer=restored_optimizer,
            )
            self.assertEqual(3, loaded.epoch)
            self.assertEqual(27, loaded.global_step)
            self.assertTrue(loaded.optimizer_loaded)
            restored_model.eval()
            with torch.inference_mode():
                actual = restored_model(image)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            with self.assertRaisesRegex(WireCheckpointError, "lineage differs"):
                load_wire_checkpoint(checkpoint, expected_lineage={"run_id": "wrong"})

    def test_torchscript_export_and_adapter_preserve_local_shapes_and_order(self) -> None:
        torch = self.torch
        model = build_tiny_unet(base_channels=4)
        # A deterministic 0-logit model makes the configured threshold observable.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "wire.torchscript"
            report = export_wire_model(
                model,
                artifact,
                input_shape=(2, 3, 17, 19),
                format="torchscript",
            )
            self.assertTrue(report.passed)
            self.assertLessEqual(report.max_abs_error, report.tolerance)
            verified = verify_wire_export(
                model,
                artifact,
                format="torchscript",
                inputs=torch.rand(2, 3, 13, 15),
            )
            self.assertTrue(verified["passed"])
            entry = {
                "task": "wire",
                "version": "wire-synthetic-v1",
                "sha256": report.artifact_sha256,
                "artifact_format": "torchscript",
                "device": "cpu",
                "threshold": 0.51,
                "batch_size": 2,
            }
            adapter = create_wire_provider(artifact, entry)
            views = [
                _view("first", (19, 17), (220, 220, 220)),
                _view("second", (19, 17), (20, 30, 40)),
                _view("third", (11, 13), (100, 110, 120)),
            ]
            masks = adapter.segment(views)
            self.assertEqual([view.view_id for view in views], [mask.view_id for mask in masks])
            self.assertEqual([(17, 19), (17, 19), (13, 11)], [mask.mask.shape for mask in masks])
            self.assertTrue(all(mask.mask.dtype == np.bool_ for mask in masks))
            self.assertTrue(all(not bool(mask.mask.any()) for mask in masks))
            for mask, view in zip(masks, views, strict=True):
                self.assertEqual(view.view_id, mask.evidence[0].view_id)
                self.assertEqual(
                    [view.height, view.width],
                    mask.evidence[0].metadata["local_mask_shape"],
                )
                self.assertFalse(mask.evidence[0].metadata["vectorized"])
                self.assertEqual("wire-synthetic-v1", mask.evidence[0].metadata["model_version"])

    @unittest.skipUnless(
        ONNX_RUNTIME_AVAILABLE, "ONNX Runtime is not installed for consistency smoke"
    )
    def test_onnx_export_consistency_and_adapter(self) -> None:
        model = build_tiny_unet(base_channels=4)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "wire.onnx"
            report = export_wire_model(
                model,
                artifact,
                input_shape=(1, 3, 17, 19),
                format="onnx",
            )
            self.assertTrue(report.passed)
            self.assertLessEqual(report.max_abs_error, report.tolerance)
            adapter = TorchWireAdapter(
                artifact,
                {
                    "task": "wire",
                    "version": "wire-onnx-v1",
                    "sha256": report.artifact_sha256,
                    "device": "cpu",
                },
            )
            mask = adapter.segment([_view("dynamic", (23, 21), (80, 90, 100))])[0]
            self.assertEqual((21, 23), mask.mask.shape)
            self.assertEqual("onnxruntime", mask.evidence[0].metadata["runtime"])

    def test_real_frozen_wire_sample_small_training_smoke(self) -> None:
        """Exercise the current official-data materialization, not only fixtures."""

        torch = self.torch
        import os as _os

        override = _os.environ.get("PCB_WIRE_JUNCTION_MATERIALIZED_ROOT")
        if override:
            run_dir = (
                Path(override)
                / "runs"
                / "5233aeb838db659970f0ce73480753a2f4a4e79366612878986d3bf18bca610e"
            )
        else:
            run_dir = (
                REPOSITORY_ROOT
                / "artifacts"
                / "training"
                / "materialized"
                / "dataset300-v2"
                / "wire-junction"
                / "runs"
                / "5233aeb838db659970f0ce73480753a2f4a4e79366612878986d3bf18bca610e"
            )
        source_root = REPOSITORY_ROOT / "Dataset300" / "Dataset300"
        if not (run_dir / "wire_tiles.jsonl").is_file() or not source_root.is_dir():
            self.skipTest("current official-data wire freeze is not present")
        from pcbparse.training.torch_wire_junction import WireTorchDataset

        dataset = WireTorchDataset(run_dir, source_root=source_root, split="train")
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        # Keep the smoke small while retaining pixels loaded and hash-verified by
        # the real freeze.  Nearest-neighbour masks preserve ignore semantics.
        image = torch.nn.functional.interpolate(
            sample["image"].unsqueeze(0), size=(48, 48), mode="bilinear", align_corners=False
        )
        target = torch.nn.functional.interpolate(
            sample["target"].unsqueeze(0), size=(48, 48), mode="nearest"
        )
        ignore = torch.nn.functional.interpolate(
            sample["ignore"].unsqueeze(0), size=(48, 48), mode="nearest"
        )
        self.assertGreater(float((1.0 - ignore).sum()), 0.0)
        model = build_tiny_unet(base_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        result = wire_train_step(
            model,
            {"image": image, "target": target, "ignore": ignore},
            optimizer,
        )
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertGreater(result["valid_pixels"], 0.0)


def _view(view_id: str, size: tuple[int, int], colour: tuple[int, int, int]) -> ImageView:
    return ImageView(
        view_id=view_id,
        image=Image.new("RGB", size, colour),
        transform=TransformChain([AffineTransform.identity(view_id)]),
        sha256=hashlib.sha256(f"{view_id}:{size}:{colour}".encode()).hexdigest(),
    )


if __name__ == "__main__":
    unittest.main()
