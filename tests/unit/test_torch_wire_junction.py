from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pcbparse.training.torch_wire_junction import (
    DatasetGateError,
    FrozenDataError,
    JUNCTION_CLASSES,
    JunctionTorchDataset,
    TorchUnavailableError,
    WireTorchDataset,
    inspect_junction_dataset,
    inspect_wire_dataset,
    junction_class_weights,
    junction_collate,
    junction_sample_weights,
    make_junction_weighted_sampler,
    wire_collate,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FrozenDatasetInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_dir, self.source_root = _write_freeze(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_inspection_needs_no_torch_and_exposes_fail_closed_metadata(self) -> None:
        wire = inspect_wire_dataset(self.run_dir, split="train")
        self.assertEqual(3, wire.selected_records)
        self.assertEqual(2, wire.eligible_records)
        self.assertEqual(1, wire.excluded_ineligible_records)
        self.assertEqual(1, wire.excluded_other_splits)

        junction = inspect_junction_dataset(self.run_dir, split="train")
        self.assertEqual(4, junction.selected_records)
        self.assertEqual(3, junction.trusted_records)
        self.assertEqual(1, junction.excluded_unknown_records)
        self.assertEqual(2, junction.class_counts["endpoint"])
        self.assertEqual(1, junction.class_counts["bend"])
        self.assertEqual(
            ("t_junction", "connected_cross", "non_connected_cross", "noise"),
            junction.missing_classes,
        )
        self.assertFalse(junction.six_class_training_ready)
        with self.assertRaisesRegex(DatasetGateError, "missing classes"):
            junction.require_training_ready()

    def test_tensor_dataset_fails_clearly_when_torch_import_is_unavailable(self) -> None:
        with mock.patch(
            "pcbparse.training.torch_wire_junction.importlib.import_module",
            side_effect=ImportError("fixture has no torch"),
        ):
            with self.assertRaisesRegex(
                TorchUnavailableError, "PyTorch is required.*freeze inspection"
            ):
                WireTorchDataset(self.run_dir, source_root=self.source_root)
            with self.assertRaises(TorchUnavailableError):
                JunctionTorchDataset(self.run_dir)

    def test_manifest_hash_corruption_is_rejected(self) -> None:
        manifest = self.run_dir / "wire_tiles.jsonl"
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(FrozenDataError, "byte count mismatch"):
            inspect_wire_dataset(self.run_dir)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training runtime is not installed")
class TorchWireJunctionDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_dir, self.source_root = _write_freeze(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_wire_reads_lazy_tiles_filters_ineligible_and_collates_padding_as_ignore(self) -> None:
        dataset = WireTorchDataset(
            self.run_dir, source_root=self.source_root, split="train"
        )
        self.assertEqual(2, len(dataset))
        self.assertEqual(1, dataset.metadata.excluded_ineligible_records)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual((3, 4, 4), tuple(first["image"].shape))
        self.assertEqual((1, 4, 4), tuple(first["target"].shape))
        self.assertEqual((1, 4, 4), tuple(first["ignore"].shape))
        self.assertAlmostEqual(1.0, float(first["target"][0, 1, 1]))
        self.assertAlmostEqual(0.0, float(first["target"][0, 1, 2]))
        self.assertAlmostEqual(1.0, float(first["ignore"][0, 1, 2]))
        self.assertAlmostEqual(0.0, float(first["valid"][0, 1, 2]))
        self.assertEqual((3, 4, 2), tuple(second["image"].shape))

        batch = wire_collate([first, second])
        self.assertEqual((2, 3, 4, 4), tuple(batch["image"].shape))
        self.assertEqual((2, 1, 4, 4), tuple(batch["target"].shape))
        self.assertEqual([[4, 4], [4, 2]], batch["original_shapes"].tolist())
        # Padding is invalid supervision rather than a false background target.
        self.assertTrue(bool(batch["ignore"][1, :, :, 2:].all()))
        self.assertFalse(bool(batch["target"][1, :, :, 2:].any()))

    def test_wire_asset_hash_is_verified_lazily(self) -> None:
        dataset = WireTorchDataset(
            self.run_dir, source_root=self.source_root, split="train"
        )
        target_path = self.run_dir / dataset.records[0]["target_mask"]["path"]
        target_path.write_bytes(target_path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(FrozenDataError, "SHA-256 mismatch"):
            _ = dataset[0]

    def test_junction_excludes_unknown_returns_labels_and_weight_helpers_stay_gated(self) -> None:
        dataset = JunctionTorchDataset(self.run_dir, split="train")
        self.assertEqual(3, len(dataset))
        self.assertNotIn("UNKNOWN", [row["label"] for row in dataset.records])
        first = dataset[0]
        self.assertEqual((3, 4, 4), tuple(first["image"].shape))
        self.assertEqual(0, int(first["label"]))
        self.assertEqual("endpoint", first["label_name"])

        batch = junction_collate([dataset[index] for index in range(len(dataset))])
        self.assertEqual((3, 3, 4, 4), tuple(batch["image"].shape))
        self.assertEqual([0, 0, 2], batch["label"].tolist())

        weights = junction_sample_weights(dataset)
        self.assertEqual(3, int(weights.numel()))
        self.assertAlmostEqual(0.75, float(weights[0]), places=6)
        self.assertAlmostEqual(0.75, float(weights[1]), places=6)
        self.assertAlmostEqual(1.5, float(weights[2]), places=6)
        with self.assertRaises(DatasetGateError):
            junction_class_weights(dataset)
        partial_weights = junction_class_weights(dataset, require_complete=False)
        self.assertEqual(6, int(partial_weights.numel()))
        self.assertEqual(0.0, float(partial_weights[1]))
        with self.assertRaises(DatasetGateError):
            make_junction_weighted_sampler(dataset)
        sampler = make_junction_weighted_sampler(
            dataset, require_complete=False, num_samples=5
        )
        self.assertEqual(5, int(sampler.num_samples))
        with self.assertRaises(DatasetGateError):
            JunctionTorchDataset(
                self.run_dir, split="train", require_training_ready=True
            )


def _write_freeze(root: Path) -> tuple[Path, Path]:
    source_root = root / "source"
    source_path = source_root / "s1" / "image.png"
    source_path.parent.mkdir(parents=True)
    source = Image.new("RGB", (6, 4), (20, 40, 60))
    for x in range(6):
        source.putpixel((x, 1), (100 + x, 120, 140))
    source.save(source_path, format="PNG")

    run_dir = root / "run"
    run_dir.mkdir()
    run_id = "1" * 64
    wire_rows: list[dict[str, object]] = []
    wire_specs = (
        ("tile-a", "train", (0, 0, 4, 4), True),
        ("tile-b", "train", (4, 0, 6, 4), True),
        ("tile-ineligible", "train", (0, 0, 4, 4), False),
        ("tile-val", "val", (0, 0, 4, 4), True),
    )
    for name, split, bbox, eligible in wire_specs:
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        target = Image.new("L", (width, height), 0)
        ignore = Image.new("L", (width, height), 0)
        if eligible:
            for x in range(width):
                target.putpixel((x, 1), 255)
            if name == "tile-a":
                target.putpixel((2, 1), 0)
                ignore.putpixel((2, 1), 255)
        else:
            ignore = Image.new("L", (width, height), 255)
        target_rel = Path("wire/target_masks/s1") / f"{name}.png"
        ignore_rel = Path("wire/ignore_masks/s1") / f"{name}.png"
        (run_dir / target_rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / ignore_rel).parent.mkdir(parents=True, exist_ok=True)
        target.save(run_dir / target_rel, format="PNG")
        ignore.save(run_dir / ignore_rel, format="PNG")
        wire_rows.append(
            {
                "schema_version": 1,
                "record_id": f"s1:wire_tile:{name}",
                "sample_id": "s1",
                "split": split,
                "tile_bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
                "tile_size": [width, height],
                "target_mask": {
                    "path": target_rel.as_posix(),
                    "sha256": _sha256(run_dir / target_rel),
                    "foreground_pixels": (
                        width - 1 if name == "tile-a" else width if eligible else 0
                    ),
                },
                "ignore_mask": {
                    "path": ignore_rel.as_posix(),
                    "sha256": _sha256(run_dir / ignore_rel),
                    "foreground_pixels": (
                        1 if name == "tile-a" else 0 if eligible else width * height
                    ),
                    "regions": [],
                },
                "transform": {
                    "tile_to_image": [[1, 0, bbox[0]], [0, 1, bbox[1]], [0, 0, 1]],
                    "image_to_tile": [[1, 0, -bbox[0]], [0, 1, -bbox[1]], [0, 0, 1]],
                },
                "source": {
                    "image": "s1/image.png",
                    "image_sha256": _sha256(source_path),
                    "json": "s1/source.json",
                    "json_sha256": "2" * 64,
                    "derived_wire_record_ids": ["s1:edge:E1"],
                },
                "supervision": {
                    "eligible": eligible,
                    "ineligible_reason": None if eligible else "fixture_unusable_sparse",
                    "unusable_sparse_record_ids": [] if eligible else ["s1:edge:E0"],
                },
                "lineage": {
                    "run_id": run_id,
                    "geometry_sources": ["keyPointPaths", "keyPoints/skeleton"],
                    "dense_coordinates_loaded": False,
                },
            }
        )
    wire_manifest = run_dir / "wire_tiles.jsonl"
    _write_jsonl(wire_manifest, wire_rows)

    junction_specs = (
        ("endpoint-a", "train", "endpoint", "trusted_gt"),
        ("endpoint-b", "train", "endpoint", "trusted_gt"),
        ("bend", "train", "bend", "trusted_gt"),
        ("cross", "train", "UNKNOWN", "needs_review"),
        ("tee-val", "val", "t_junction", "trusted_gt"),
    )
    junction_rows: list[dict[str, object]] = []
    for index, (name, split, label, status) in enumerate(junction_specs):
        patch = Image.new("RGB", (4, 4), (20 * index, 30, 40))
        patch_rel = Path("junction/patches/s1") / f"{name}.png"
        (run_dir / patch_rel).parent.mkdir(parents=True, exist_ok=True)
        patch.save(run_dir / patch_rel, format="PNG")
        label_id = -1 if label == "UNKNOWN" else JUNCTION_CLASSES.index(label)
        junction_rows.append(
            {
                "schema_version": 1,
                "record_id": f"s1:junction:{name}",
                "sample_id": "s1",
                "split": split,
                "label": label,
                "label_id": label_id,
                "label_status": status,
                "review_reason": (
                    "cross_connectivity_not_encoded_by_gt"
                    if label == "UNKNOWN"
                    else None
                ),
                "patch": {
                    "path": patch_rel.as_posix(),
                    "sha256": _sha256(run_dir / patch_rel),
                    "size": [4, 4],
                },
                "transform": {
                    "patch_to_image": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "image_to_patch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "source": {
                    "image": "s1/image.png",
                    "image_sha256": _sha256(source_path),
                    "json": "s1/source.json",
                    "json_sha256": "2" * 64,
                    "source_refs": [],
                },
                "lineage": {"run_id": run_id, "derived_record_id": name},
            }
        )
    junction_manifest = run_dir / "junction_annotations.jsonl"
    _write_jsonl(junction_manifest, junction_rows)

    schema = {
        "schema_version": 1,
        "classes": [
            {"id": index, "name": name}
            for index, name in enumerate(JUNCTION_CLASSES)
        ],
        "unknown": {"id": -1, "name": "UNKNOWN", "trainable": False},
        "cross_policy": {"automatic_label": "UNKNOWN", "needs_review": True},
    }
    schema_path = run_dir / "junction_label_schema.json"
    schema_path.write_text(_pretty(schema), encoding="utf-8")
    train_counts = {
        "endpoint": 2,
        "t_junction": 0,
        "bend": 1,
        "connected_cross": 0,
        "non_connected_cross": 0,
        "noise": 0,
        "UNKNOWN": 1,
    }
    val_counts = {
        "endpoint": 0,
        "t_junction": 1,
        "bend": 0,
        "connected_cross": 0,
        "non_connected_cross": 0,
        "noise": 0,
        "UNKNOWN": 0,
    }
    trusted_global = {
        name: train_counts[name] + val_counts[name] for name in JUNCTION_CLASSES
    }
    summary = {
        "schema_version": 1,
        "kind": "wire_junction_materialization",
        "complete": True,
        "run_id": run_id,
        "materialization": {"ok": True},
        "wire": {"mask_semantics": {"foreground": 255, "background": 0}},
        "training_gate": {
            "ok": False,
            "required_classes": list(JUNCTION_CLASSES),
            "required_split": "train",
            "missing_classes_in_train": [
                name for name in JUNCTION_CLASSES if train_counts[name] == 0
            ],
            "blockers": [
                {
                    "code": "JUNCTION_CLASS_MISSING_FROM_TRAIN",
                    "class": name,
                }
                for name in JUNCTION_CLASSES
                if train_counts[name] == 0
            ]
            + [{"code": "JUNCTION_REVIEW_PENDING", "count": 1}],
        },
        "junction_class_counts": {
            "trusted": trusted_global,
            "UNKNOWN": 1,
            "by_split": {"train": train_counts, "val": val_counts},
        },
        "outputs": {
            "wire_tiles": _binding(run_dir, wire_manifest),
            "junction_annotations": _binding(run_dir, junction_manifest),
            "junction_label_schema": _binding(run_dir, schema_path),
        },
    }
    (run_dir / "summary.json").write_text(_pretty(summary), encoding="utf-8")
    return run_dir, source_root


def _binding(run_dir: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
