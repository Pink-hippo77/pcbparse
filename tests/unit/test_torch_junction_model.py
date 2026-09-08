from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pcbparse.io import ImageView, load_image, tile_image_view
from pcbparse.ir.models import Point, WireSegment
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.torch_junction_adapter import (
    JunctionAdapterConfig,
    TorchJunctionProvider,
    build_junction_patches,
)
from pcbparse.training.synthetic_junction import (
    MixedJunctionTrainingDataset,
    SyntheticJunctionConfig,
    SyntheticJunctionDataset,
    SyntheticJunctionError,
    build_mixed_junction_training_dataset,
    generate_synthetic_junction_patch,
)
from pcbparse.training.torch_junction_model import (
    JunctionCheckpointError,
    build_junction_model,
    evaluate_junction_model,
    export_junction_model,
    junction_class_weights_from_counts,
    junction_metrics_from_confusion,
    junction_train_step,
    load_junction_checkpoint,
    save_junction_checkpoint,
)
from pcbparse.training.torch_wire_junction import JUNCTION_CLASSES, JunctionTorchDataset, junction_collate


ROOT = Path(__file__).resolve().parents[2]
_WIRE_ROOT = Path(
    __import__("os").environ.get(
        "PCB_WIRE_JUNCTION_MATERIALIZED_ROOT",
        ROOT / "artifacts/training/materialized/dataset300-v2/wire-junction",
    )
)
OFFICIAL_POINTER = _WIRE_ROOT / "latest.json"
HAS_TORCH = importlib.util.find_spec("torch") is not None


class SyntheticJunctionContractTests(unittest.TestCase):
    def test_all_six_classes_are_deterministic_distinct_and_auditable(self) -> None:
        config = SyntheticJunctionConfig(seed=19, patch_size=48)
        first = [generate_synthetic_junction_patch(name, 3, config=config) for name in JUNCTION_CLASSES]
        second = [generate_synthetic_junction_patch(name, 3, config=config) for name in JUNCTION_CLASSES]
        self.assertEqual([item.image_sha256 for item in first], [item.image_sha256 for item in second])
        self.assertEqual(6, len({item.image_sha256 for item in first}))
        arrays = [np.asarray(item.image, dtype=np.float32) for item in first]
        self.assertGreater(
            min(float(np.abs(left - right).mean()) for left, right in itertools.combinations(arrays, 2)),
            5.0,
        )
        self.assertEqual(list(JUNCTION_CLASSES), [item.label for item in first])
        for expected_id, item in enumerate(first):
            self.assertEqual(expected_id, item.label_id)
            self.assertEqual(config.policy_sha256, item.metadata["policy_sha256"])
            self.assertFalse(item.metadata["raw_unknown_crosses_used"])
            self.assertEqual("deterministic_synthetic_train_only", item.metadata["provenance"])
        connected = first[JUNCTION_CLASSES.index("connected_cross")].metadata["visual_contract"]
        disconnected = first[JUNCTION_CLASSES.index("non_connected_cross")].metadata["visual_contract"]
        self.assertTrue(connected["node_dot"])
        self.assertFalse(connected["bridge_gap"])
        self.assertFalse(disconnected["node_dot"])
        self.assertTrue(disconnected["underpass_visibly_gapped"])

        # Pixel evidence for the semantic cross distinction: at a clean,
        # unrotated crossing the vertical trace remains dark through a node,
        # while the corresponding underpass position is erased for a bridge.
        clean = SyntheticJunctionConfig(
            seed=19,
            maximum_center_jitter=0,
            maximum_rotation_degrees=0,
            maximum_gaussian_noise=0,
        )
        connected_asset = generate_synthetic_junction_patch("connected_cross", 3, config=clean)
        bridge_asset = generate_synthetic_junction_patch("non_connected_cross", 3, config=clean)

        def underpass_probe(asset):
            array = np.asarray(asset.image, dtype=np.float32)[:, :, 0]
            x, y = asset.metadata["parameters"]["center"]
            width = asset.metadata["parameters"]["line_width"]
            return float(array[round(y + 0.8 * width), round(x)])

        self.assertLess(underpass_probe(connected_asset), 80.0)
        self.assertGreater(underpass_probe(bridge_asset), 150.0)

    def test_train_only_gate_is_structural(self) -> None:
        with self.assertRaisesRegex(SyntheticJunctionError, "train-only"):
            SyntheticJunctionDataset(samples_per_class=1, split="val")
        with self.assertRaisesRegex(SyntheticJunctionError, "train-only"):
            SyntheticJunctionDataset(samples_per_class=1, split="test")


@unittest.skipUnless(HAS_TORCH, "PyTorch training environment is optional")
class TorchJunctionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        torch.manual_seed(11)

    def test_real_and_synthetic_mix_excludes_unknown_and_completes_schema(self) -> None:
        torch = self.torch

        class TrustedReal:
            split = "train"
            metadata = SimpleNamespace(run_id="f" * 64)

            def __init__(self) -> None:
                self.records = [
                    {"label": name, "label_status": "trusted_gt"}
                    for name in JUNCTION_CLASSES[:3]
                ]

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, index: int):
                name = self.records[index]["label"]
                return {
                    "image": torch.zeros((3, 64, 64)),
                    "label": torch.tensor(JUNCTION_CLASSES.index(name)),
                    "label_name": name,
                    "record_id": f"real:{index}",
                    "sample_id": f"real:{index}",
                    "split": "train",
                    "transform": {},
                    "source": {"kind": "trusted_real"},
                }

        mixed = build_mixed_junction_training_dataset(TrustedReal(), samples_per_class=2)
        self.assertIsInstance(mixed, MixedJunctionTrainingDataset)
        self.assertEqual(9, len(mixed))
        self.assertEqual(set(JUNCTION_CLASSES), set(mixed.class_counts))
        self.assertTrue(all(mixed.class_counts[name] > 0 for name in JUNCTION_CLASSES))
        self.assertFalse(mixed.lineage["raw_unknown_crosses_used"])
        self.assertEqual("connected_cross", mixed[3]["label_name"])

    def test_loss_balancing_backward_metrics_and_evaluation(self) -> None:
        torch = self.torch
        dataset = SyntheticJunctionDataset(samples_per_class=2, config={"patch_size": 32, "seed": 3})
        batch = junction_collate([dataset[index] for index in range(len(dataset))])
        model = build_junction_model(base_channels=4, input_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        weights = junction_class_weights_from_counts(dataset.class_counts)
        result = junction_train_step(model, batch, optimizer, class_weights=weights)
        self.assertGreater(result["loss"], 0.0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        perfect = junction_metrics_from_confusion(torch.eye(6, dtype=torch.long))
        self.assertEqual(1.0, perfect["macro_f1"])
        evaluation = evaluate_junction_model(model, [batch])
        self.assertIn("macro_f1", evaluation["metrics"])
        for name in JUNCTION_CLASSES:
            self.assertIn(f"recall/{name}", evaluation["metrics"])
        self.assertEqual(len(dataset), len(evaluation["predictions"]))

    def test_atomic_checkpoint_lineage_and_torchscript_onnx_consistency(self) -> None:
        torch = self.torch
        model = build_junction_model(base_channels=4, input_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lineage = {
            "dataset_run_id": "a" * 64,
            "synthetic_policy_sha256": "b" * 64,
            "raw_unknown_crosses_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "junction.ckpt"
            saved = save_junction_checkpoint(
                checkpoint,
                model,
                optimizer=optimizer,
                epoch=2,
                global_step=17,
                lineage=lineage,
                metrics={"macro_f1": 0.5},
            )
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), saved.sha256)
            loaded = load_junction_checkpoint(checkpoint, expected_lineage=lineage)
            self.assertEqual(2, loaded.epoch)
            self.assertEqual(JUNCTION_CLASSES, tuple(loaded.model.model_config["class_names"]))
            with self.assertRaisesRegex(JunctionCheckpointError, "lineage differs"):
                load_junction_checkpoint(checkpoint, expected_lineage={"wrong": True})

            script_report = export_junction_model(
                model,
                root / "junction.ts",
                format="torchscript",
                input_shape=(2, 3, 32, 32),
            )
            self.assertTrue(script_report.passed)
            self.assertEqual(0.0, script_report.max_abs_error)
            if importlib.util.find_spec("onnxruntime") is not None:
                onnx_report = export_junction_model(
                    model,
                    root / "junction.onnx",
                    format="onnx",
                    input_shape=(2, 3, 32, 32),
                    tolerance=1e-3,
                )
                self.assertTrue(onnx_report.passed)
                self.assertLessEqual(onnx_report.max_abs_error, 1e-3)

    def test_adapter_unique_view_coordinates_stable_id_and_ambiguous_threshold(self) -> None:
        torch = self.torch

        class ConstantClassifier(torch.nn.Module):
            def __init__(self, connected_bias: float) -> None:
                super().__init__()
                self.register_buffer("bias", torch.tensor([0.0, 0.0, 0.0, connected_bias, 0.0, 0.0]))

            def forward(self, image):
                return self.bias.unsqueeze(0).expand(image.shape[0], -1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "board.png"
            Image.new("RGB", (150, 80), "white").save(image_path)
            original = load_image(image_path)
            views = tile_image_view(original, tile_size=100, overlap=50)
            segments = [
                WireSegment(internal_id="horizontal", start=Point(x=20, y=40), end=Point(x=130, y=40)),
                WireSegment(internal_id="vertical", start=Point(x=75, y=5), end=Point(x=75, y=75)),
            ]
            patches = build_junction_patches(
                views,
                segments,
                config=JunctionAdapterConfig(input_size=32, patch_size_original_px=32),
            )
            crossing = next(
                item
                for item in patches
                if abs(item.original_point.x - 75) < 1e-6 and abs(item.original_point.y - 40) < 1e-6
            )
            # The second overlapping tile owns the point because its nearest
            # border is one pixel farther away (25 px versus 24 px).
            self.assertEqual("tile_000_001", crossing.view.view_id)
            self.assertAlmostEqual(75.0, crossing.view.to_original(crossing.local_point).x)

            high_path = root / "high.ts"
            traced = torch.jit.trace(ConstantClassifier(8.0), torch.zeros((1, 3, 32, 32)))
            traced.save(str(high_path))
            provider = TorchJunctionProvider(
                high_path,
                {
                    "task": "junction",
                    "artifact_format": "torchscript",
                    "device": "cpu",
                    "input_size": 32,
                    "patch_size_original_px": 32,
                    "score_threshold": 0.55,
                },
            )
            first = provider.classify(views, segments)
            second = provider.classify(views, segments)
            self.assertEqual([item.internal_id for item in first], [item.internal_id for item in second])
            predicted_cross = next(
                item
                for item in first
                if item.evidence[0].metadata["original_candidate_point"] == [75.0, 40.0]
            )
            self.assertEqual("connected_cross", predicted_cross.kind)
            self.assertEqual(1, len(predicted_cross.evidence))
            self.assertEqual("tile_000_001", predicted_cross.evidence[0].view_id)
            restored = crossing.view.to_original(predicted_cross.point)
            self.assertAlmostEqual(75.0, restored.x)
            self.assertAlmostEqual(40.0, restored.y)

            low_path = root / "low.ts"
            torch.jit.trace(ConstantClassifier(0.0), torch.zeros((1, 3, 32, 32))).save(str(low_path))
            low = TorchJunctionProvider(
                low_path,
                {
                    "task": "junction",
                    "artifact_format": "torchscript",
                    "device": "cpu",
                    "input_size": 32,
                    "patch_size_original_px": 32,
                    "score_threshold": 0.8,
                },
            )
            self.assertTrue(all(item.kind == "ambiguous" for item in low.classify(views, segments)))

    def test_patch_extent_respects_scaled_view_transform(self) -> None:
        scaled = ImageView(
            view_id="scaled",
            image=Image.new("RGB", (200, 120), "white"),
            transform=TransformChain(
                [
                    AffineTransform.scale_translate(
                        scale_x=2.0,
                        scale_y=2.0,
                        offset_x=0.0,
                        offset_y=0.0,
                        source_view="original",
                        target_view="scaled",
                    )
                ]
            ),
        )
        segments = [
            WireSegment(internal_id="s", start=Point(x=10, y=20), end=Point(x=80, y=20))
        ]
        patches = build_junction_patches(
            [scaled],
            segments,
            config=JunctionAdapterConfig(input_size=32, patch_size_original_px=32),
        )
        endpoint = next(item for item in patches if item.original_point.x == 10.0)
        self.assertEqual((20.0, 40.0), (endpoint.local_point.x, endpoint.local_point.y))
        left, top, right, bottom = endpoint.patch_box_local
        self.assertEqual(64.0, right - left)
        self.assertEqual(64.0, bottom - top)
        restored = endpoint.view.to_original(endpoint.local_point)
        self.assertEqual((10.0, 20.0), (restored.x, restored.y))

    def test_adapter_point_is_local_with_nonzero_origin_and_scale(self) -> None:
        torch = self.torch

        class ConstantClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "bias", torch.tensor([0.0, 0.0, 0.0, 8.0, 0.0, 0.0])
                )

            def forward(self, image):
                return self.bias.unsqueeze(0).expand(image.shape[0], -1)

        # This view covers original x=[50,100], y=[20,60] at 2x scale.
        # original (75,40) therefore has view-local coordinate (50,40).
        scaled_tile = ImageView(
            view_id="scaled-tile-x50-y20",
            image=Image.new("RGB", (100, 80), "white"),
            transform=TransformChain(
                [
                    AffineTransform.scale_translate(
                        scale_x=2.0,
                        scale_y=2.0,
                        offset_x=-100.0,
                        offset_y=-40.0,
                        source_view="original",
                        target_view="scaled-tile-x50-y20",
                    )
                ]
            ),
        )
        segments = [
            WireSegment(
                internal_id="horizontal",
                start=Point(x=55, y=40),
                end=Point(x=95, y=40),
            ),
            WireSegment(
                internal_id="vertical",
                start=Point(x=75, y=25),
                end=Point(x=75, y=55),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "junction.ts"
            torch.jit.trace(
                ConstantClassifier(), torch.zeros((1, 3, 32, 32))
            ).save(str(artifact))
            provider = TorchJunctionProvider(
                artifact,
                {
                    "task": "junction",
                    "artifact_format": "torchscript",
                    "device": "cpu",
                    "input_size": 32,
                    "patch_size_original_px": 32,
                    "score_threshold": 0.55,
                },
            )
            junctions = provider.classify([scaled_tile], segments)
        crossing = next(
            item
            for item in junctions
            if item.evidence[0].metadata["original_candidate_point"] == [75.0, 40.0]
        )
        self.assertEqual("scaled-tile-x50-y20", crossing.evidence[0].view_id)
        self.assertEqual((50.0, 40.0), (crossing.point.x, crossing.point.y))
        self.assertEqual(
            [50.0, 40.0], crossing.evidence[0].metadata["local_candidate_point"]
        )
        restored = scaled_tile.to_original(crossing.point)
        self.assertEqual((75.0, 40.0), (restored.x, restored.y))
        # CompositePerceptionProvider's _restore_junctions performs precisely
        # this single Evidence.view_id lookup and to_original conversion.
        self.assertNotEqual(
            (crossing.point.x, crossing.point.y), (restored.x, restored.y)
        )

    def test_official_frozen_real_smoke_preserves_unknown_gate(self) -> None:
        if not OFFICIAL_POINTER.is_file():
            self.skipTest("official frozen junction materialization is absent")
        pointer = json.loads(OFFICIAL_POINTER.read_text(encoding="utf-8"))
        run_dir = OFFICIAL_POINTER.parent / pointer["run_dir"]
        dataset = JunctionTorchDataset(run_dir, split="train")
        self.assertGreater(len(dataset), 0)
        self.assertGreater(dataset.metadata.excluded_unknown_records, 0)
        self.assertIn("connected_cross", dataset.metadata.missing_classes)
        self.assertTrue(all(row["label_status"] == "trusted_gt" for row in dataset.records))
        self.assertTrue(all(row["label"] != "UNKNOWN" for row in dataset.records))
        sample = dataset[0]
        self.assertEqual((3,), tuple(sample["image"].shape[:1]))
        self.assertIn(sample["label_name"], JUNCTION_CLASSES[:3])
        batch = junction_collate([dataset[index] for index in range(min(4, len(dataset)))])
        model = build_junction_model(base_channels=4)
        model.eval()
        with self.torch.inference_mode():
            logits = model(batch["image"])
        self.assertEqual((len(batch["record_id"]), 6), tuple(logits.shape))
        self.assertTrue(bool(self.torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
