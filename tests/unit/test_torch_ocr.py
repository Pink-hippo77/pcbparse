from __future__ import annotations

import builtins
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from pcbparse.training.ocr_materialize import (
    OCRMaterializationOptions,
    materialize_ocr_training_data,
)
from pcbparse.training.torch_ocr import (
    CTCCharset,
    TorchOCRDataError,
    TorchOCRRecognitionDataset,
    ctc_collate,
    make_ctc_collate,
    require_torch,
)


try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch training runtime is not installed")
class TorchOCRDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        fixture = _make_fixture(self.root)
        self.source_root = fixture["source_root"]
        self.materialized = self.root / "materialized"
        materialize_ocr_training_data(
            **fixture,
            output_root=self.materialized,
            options=OCRMaterializationOptions(
                materialize_images=True,
                verify_all_source_hashes=True,
            ),
        )

    def test_materialized_and_lazy_crops_are_identical_raw_ctc_samples(self) -> None:
        materialized = TorchOCRRecognitionDataset(
            self.materialized,
            split="train",
            image_mode="materialized",
        )
        lazy = TorchOCRRecognitionDataset(
            self.materialized,
            split="train",
            image_mode="lazy",
            source_root=self.source_root,
        )

        self.assertEqual(2, len(materialized))
        self.assertEqual(2, len(lazy))
        first = materialized[0]
        lazy_first = lazy[0]
        self.assertEqual(" A惟 ", first["text"])
        self.assertEqual("0001:text:T1", first["record_id"])
        self.assertEqual([1, 48, 80], list(first["image"].shape))
        self.assertEqual(80, first["image_width"].item())
        self.assertEqual(4, first["target_length"].item())
        self.assertTrue(torch.equal(first["target"], lazy_first["target"]))
        self.assertTrue(torch.equal(first["image"], lazy_first["image"]))
        self.assertEqual("materialized", first["metadata"]["image_origin"])
        self.assertEqual("lazy", lazy_first["metadata"]["image_origin"])
        self.assertEqual(" A惟 ", materialized.charset.decode(first["target"].tolist()))
        self.assertNotIn(materialized.charset.blank_index, first["target"].tolist())
        self.assertEqual(len(materialized.charset.characters) + 1, materialized.charset.num_classes)

    def test_ctc_collate_pads_width_and_concatenates_targets(self) -> None:
        dataset = TorchOCRRecognitionDataset(
            self.materialized,
            split="train",
            image_mode="auto",
            source_root=self.source_root,
        )
        samples = [dataset[0], dataset[1]]

        batch = ctc_collate(samples)

        self.assertEqual([2, 1, 48, 80], list(batch["images"].shape))
        self.assertEqual([80, 32], batch["image_widths"].tolist())
        self.assertEqual([4, 1], batch["target_lengths"].tolist())
        self.assertEqual(5, int(batch["targets"].numel()))
        self.assertTrue(torch.all(batch["images"][1, :, :, 32:] == 1.0))
        self.assertEqual([" A惟 ", "B"], batch["texts"])
        self.assertEqual(
            ["0001:text:T1", "0001:text:T2"], batch["record_ids"]
        )
        configured = make_ctc_collate(pad_value=0.5)
        pickle.dumps(configured)
        configured_batch = configured(samples)
        self.assertTrue(torch.all(configured_batch["images"][1, :, :, 32:] == 0.5))

    def test_train_only_prefix_weights_upweight_tail_and_remain_capped(self) -> None:
        dataset = TorchOCRRecognitionDataset(
            self.materialized,
            split="train",
            image_mode="materialized",
        )
        dataset.records = (
            {"content": "R1"},
            {"content": "R2"},
            {"content": "R3"},
            {"content": "K1"},
            {"content": "plain-text"},
        )

        weights = dataset.alphanumeric_prefix_sampling_weights(
            power=0.5,
            maximum_weight=1.5,
        )

        self.assertEqual((1.0, 1.0, 1.0), weights[:3])
        self.assertEqual(1.5, weights[3])
        self.assertEqual(1.0, weights[4])
        dataset.split = "val"
        with self.assertRaisesRegex(TorchOCRDataError, "train split"):
            dataset.alphanumeric_prefix_sampling_weights()

    def test_excluded_long_text_is_absent_and_oov_fails_at_construction(self) -> None:
        index_path = self.materialized / "recognition_index.jsonl"
        rows = _read_jsonl(index_path)
        excluded = next(row for row in rows if row["record_id"] == "0001:text:T3")
        self.assertEqual("excluded", excluded["status"])
        self.assertEqual("over_max_text_length", excluded["exclusion_reason"])
        dataset = TorchOCRRecognitionDataset(
            self.materialized, split="train", image_mode="materialized"
        )
        self.assertNotIn(
            "0001:text:T3", [dataset[index]["record_id"] for index in range(len(dataset))]
        )

        # Validation contains Ω, which is intentionally absent from train-only
        # charset. The dataset must fail before any DataLoader worker starts.
        with self.assertRaisesRegex(TorchOCRDataError, r"charset-OOV.*U\+03A9"):
            TorchOCRRecognitionDataset(
                self.materialized, split="val", image_mode="materialized"
            )

    def test_commit_index_crop_and_source_tampering_fail_closed(self) -> None:
        index_path = self.materialized / "recognition_index.jsonl"
        index_path.write_bytes(index_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(TorchOCRDataError, "committed artifact.*SHA-256"):
            TorchOCRRecognitionDataset(
                self.materialized, split="train", image_mode="materialized"
            )

        self._rematerialize()
        dataset = TorchOCRRecognitionDataset(
            self.materialized, split="train", image_mode="materialized"
        )
        first_path = self.materialized / dataset.records[0]["output_image"]
        first_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(TorchOCRDataError, "materialized crop.*SHA-256"):
            dataset[0]

        self._rematerialize()
        lazy = TorchOCRRecognitionDataset(
            self.materialized,
            split="train",
            image_mode="lazy",
            source_root=self.source_root,
        )
        source_path = self.source_root / "0001" / "diagram.jpg"
        source_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(TorchOCRDataError, "source image.*SHA-256"):
            lazy[0]

    def _rematerialize(self) -> None:
        # Restore source pixels before rebuilding the frozen output.
        _write_source_image(self.source_root / "0001" / "diagram.jpg", (80, 40))
        source_manifest = self.root / "source.jsonl"
        rows = _read_jsonl(source_manifest)
        rows[0]["files"]["image"]["sha256"] = _sha(
            self.source_root / "0001" / "diagram.jpg"
        )
        _write_jsonl(source_manifest, rows)
        fixture = {
            "ocr_manifest": self.root / "derived" / "ocr_text.jsonl",
            "derived_summary": self.root / "derived" / "summary.json",
            "source_root": self.source_root,
            "source_manifest": source_manifest,
            "split_manifest": self.root / "split.jsonl",
        }
        materialize_ocr_training_data(
            **fixture,
            output_root=self.materialized,
            options=OCRMaterializationOptions(materialize_images=True),
        )


class TorchOCRImportAndCharsetTests(unittest.TestCase):
    def test_require_torch_fails_closed_without_importing_at_module_import(self) -> None:
        original_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ModuleNotFoundError("blocked in test")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(TorchOCRDataError, "PyTorch is required"):
                require_torch()

    def test_charset_mapping_is_codepoint_sorted_and_blank_zero(self) -> None:
        charset = CTCCharset((" ", "A", "惟"))
        self.assertEqual(0, charset.blank_index)
        self.assertEqual((1, 2, 3), charset.encode(" A惟"))
        self.assertEqual(" A惟", charset.decode((0, 1, 2, 3), remove_blank=True))
        self.assertEqual("A", charset.decode((2, 2), collapse_repeats=True))
        with self.assertRaisesRegex(TorchOCRDataError, "stable ascending"):
            CTCCharset(("B", "A"))
        with self.assertRaisesRegex(TorchOCRDataError, "charset-OOV"):
            charset.encode("Ω")


def _make_fixture(root: Path) -> dict[str, Path]:
    source_root = root / "source"
    derived_root = root / "derived"
    source_root.mkdir()
    derived_root.mkdir()
    specifications = (
        ("0001", "train", (80, 40), "diagram.jpg", "kicad"),
        ("0002", "val", (40, 30), "diagram.png", "datasheet"),
    )
    source_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for sample_id, split, size, filename, domain in specifications:
        sample_root = source_root / sample_id
        sample_root.mkdir()
        image_path = sample_root / filename
        _write_source_image(image_path, size)
        relative = f"{sample_id}/{filename}"
        group = f"group-{sample_id}"
        source_rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_class": domain,
                "duplicate_group": group,
                "canvas": {"width": size[0], "height": size[1]},
                "files": {
                    "image": {
                        "path": relative,
                        "sha256": _sha(image_path),
                        "width": size[0],
                        "height": size[1],
                        "detected_format": "PNG",
                    }
                },
            }
        )
        split_rows.append(
            {"sample_id": sample_id, "duplicate_group": group, "split": split}
        )
    source_manifest = root / "source.jsonl"
    split_manifest = root / "split.jsonl"
    _write_jsonl(source_manifest, source_rows)
    _write_jsonl(split_manifest, split_rows)

    common = {
        "schema_version": 1,
        "task": "ocr_text",
        "derivation_version": "torch-fixture-v1",
        "config_hash": "b" * 64,
        "sample_id": "0001",
        "source_domain": "kicad",
        "source_image": "0001/diagram.jpg",
        "duplicate_group": "group-0001",
        "split": "train",
        "state": "active",
        "category": "params",
        "details": "value",
        "normalization_applied": False,
    }
    ocr_rows = [
        {
            **common,
            "record_id": "0001:text:T1",
            "content": " A惟 ",
            "bbox": {"x1": 2, "y1": 2, "x2": 18, "y2": 10},
        },
        {
            **common,
            "record_id": "0001:text:T2",
            "content": "B",
            "bbox": {"x1": 30, "y1": 4, "x2": 34, "y2": 12},
        },
        {
            **common,
            "record_id": "0001:text:T3",
            "content": "x" * 65,
            "bbox": {"x1": 40, "y1": 4, "x2": 60, "y2": 12},
        },
        {
            **common,
            "record_id": "0002:text:T1",
            "sample_id": "0002",
            "source_domain": "datasheet",
            "source_image": "0002/diagram.png",
            "duplicate_group": "group-0002",
            "split": "val",
            "content": "Ω",
            "bbox": {"x1": 5, "y1": 5, "x2": 15, "y2": 15},
        },
    ]
    ocr_manifest = derived_root / "ocr_text.jsonl"
    _write_jsonl(ocr_manifest, ocr_rows)
    summary = {
        "schema_version": 1,
        "derivation_version": "torch-fixture-v1",
        "config_hash": "b" * 64,
        "sample_count": 2,
        "counts": {"ocr_text": 4},
        "task_files": {"ocr_text": "ocr_text.jsonl"},
        "config": {"split_manifest_sha256": _sha(split_manifest)},
    }
    summary_path = derived_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return {
        "ocr_manifest": ocr_manifest,
        "derived_summary": summary_path,
        "source_root": source_root,
        "source_manifest": source_manifest,
        "split_manifest": split_manifest,
    }


def _write_source_image(path: Path, size: tuple[int, int]) -> None:
    width, height = size
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    array = np.empty((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = (x * 3 + y * 5) % 251
    array[:, :, 1] = (x * 7 + y * 2) % 253
    array[:, :, 2] = (x * 11 + y * 13) % 255
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
