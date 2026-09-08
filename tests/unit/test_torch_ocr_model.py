from __future__ import annotations

import hashlib
import importlib.util
import json
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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.io import ImageView
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.torch_ocr_adapter import (
    TorchOCRAdapterError,
    _box_has_source_pixel_support,
    _entry_value,
    factory,
)
from pcbparse.training.torch_ocr import (
    CTCCharset,
    TorchOCRDataError,
    TorchOCRRecognitionDataset,
    ctc_collate,
)
from pcbparse.training.torch_ocr_detection import (
    TorchOCRDetectionDataset,
    _rasterize_annotations,
    detection_collate,
)
from pcbparse.training.torch_ocr_model import (
    OCRBundleError,
    TorchOCREvaluationDataError,
    TorchOCREvaluationDataset,
    TorchOCRRuntimeError,
    build_detector,
    build_recognizer,
    compute_detector_loss,
    ctc_beam_decode,
    default_ocr_preprocess,
    detection_box_metrics,
    detector_train_step,
    export_ocr_submodel,
    extract_text_boxes,
    load_ocr_bundle,
    ocr_evaluation_collate,
    recognition_output_lengths,
    recognizer_eval_step,
    recognizer_coverage_eval_step,
    recognizer_train_step,
    save_ocr_bundle,
    validate_ctc_batch,
    verify_ocr_export,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ONNX_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


class TorchOCRLazyImportTests(unittest.TestCase):
    def test_nested_inference_is_honoured_and_conflicts_are_rejected(self) -> None:
        self.assertEqual(
            0.4,
            _entry_value(
                {"inference": {"detector_threshold": 0.4}},
                "detector_threshold",
                default=0.5,
            ),
        )
        with self.assertRaisesRegex(TorchOCRAdapterError, "conflicts"):
            _entry_value(
                {
                    "detector_threshold": 0.3,
                    "inference": {"detector_threshold": 0.4},
                },
                "detector_threshold",
                default=0.5,
            )

    def test_model_operation_fails_actionably_when_torch_import_is_blocked(self) -> None:
        with mock.patch(
            "pcbparse.training.torch_ocr_model._DETECTOR_CLASS", None
        ), mock.patch(
            "pcbparse.training.torch_ocr_model.importlib.import_module",
            side_effect=ImportError("fixture without torch"),
        ):
            with self.assertRaisesRegex(TorchOCRRuntimeError, "venv-train"):
                build_detector()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch OCR tests require .venv-train")
class TorchOCRDetectionDatasetTests(unittest.TestCase):
    def test_target_shrink_separates_touching_boxes_without_changing_truth(self) -> None:
        row = {
            "annotations": [
                {
                    "record_id": "left",
                    "ignore": False,
                    "points": [[2, 2], [12, 2], [12, 12], [2, 12]],
                },
                {
                    "record_id": "right",
                    "ignore": False,
                    "points": [[12, 2], [22, 2], [22, 12], [12, 12]],
                },
            ]
        }
        merged, _, full_boxes, _ = _rasterize_annotations(
            row, width=24, height=16
        )
        separated, _, shrunk_truth, _ = _rasterize_annotations(
            row, width=24, height=16, target_shrink_ratio=0.12
        )

        self.assertEqual(1, len(extract_text_boxes(merged, min_area=1)))
        self.assertEqual(2, len(extract_text_boxes(separated, min_area=1)))
        self.assertEqual(full_boxes, shrunk_truth)
        self.assertEqual(
            [[2.0, 2.0, 12.0, 12.0], [12.0, 2.0, 22.0, 12.0]],
            shrunk_truth,
        )

    def test_target_shrink_ratio_rejects_unsafe_values(self) -> None:
        row = {"annotations": []}
        for value in (-0.01, 0.5, float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TorchOCRDataError, "target_shrink_ratio"
                ):
                    _rasterize_annotations(
                        row,
                        width=8,
                        height=8,
                        target_shrink_ratio=value,
                    )

    def test_lazy_tile_masks_resize_and_ignore_overrides_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialized, source = _make_detection_freeze(root)
            dataset = TorchOCRDetectionDataset(
                materialized,
                source_root=source,
                split="train",
                resize=16,
            )
            self.assertEqual(1, len(dataset))
            sample = dataset[0]
            self.assertEqual((3, 16, 16), tuple(sample["image"].shape))
            self.assertEqual((1, 16, 16), tuple(sample["target"].shape))
            self.assertEqual((1, 16, 16), tuple(sample["ignore"].shape))
            self.assertFalse(
                bool((sample["target"].bool() & sample["ignore"].bool()).any())
            )
            self.assertEqual((1, 4), tuple(sample["boxes"].shape))
            self.assertEqual((1, 4), tuple(sample["ignore_boxes"].shape))
            batch = detection_collate([sample, dataset[0]])
            self.assertEqual((2, 3, 16, 16), tuple(batch["images"].shape))
            self.assertEqual(["fixture:0:0", "fixture:0:0"], batch["tile_ids"])

    def test_current_official_freeze_lazy_small_training_smoke(self) -> None:
        materialized = (
            REPOSITORY_ROOT
            / "artifacts"
            / "training"
            / "materialized"
            / "dataset300-v2"
            / "ocr"
        )
        source = REPOSITORY_ROOT / "Dataset300" / "Dataset300"
        if not (materialized / "detection_index.jsonl").is_file() or not source.is_dir():
            self.skipTest("official OCR freeze is not present")
        dataset = TorchOCRDetectionDataset(
            materialized,
            source_root=source,
            split="train",
            resize=32,
            max_records=1,
        )
        sample = dataset[0]
        self.assertEqual((3, 32, 32), tuple(sample["image"].shape))
        self.assertGreater(float((1.0 - sample["ignore"]).sum()), 0.0)
        import torch

        model = build_detector(base_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        result = detector_train_step(
            model, detection_collate([sample]), optimizer
        )
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertGreater(result["valid_pixels"], 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch OCR tests require .venv-train")
class TorchOCRModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        torch.manual_seed(20260813)

    def test_adapter_rejects_subpixel_detector_fragments(self) -> None:
        self.assertTrue(_box_has_source_pixel_support(2.0, 3.0, 3.0, 4.0))
        self.assertFalse(_box_has_source_pixel_support(2.0, 3.0, 2.99, 4.0))
        self.assertFalse(_box_has_source_pixel_support(2.0, 3.0, 3.0, 3.99))
        self.assertFalse(
            _box_has_source_pixel_support(2.0, 3.0, float("inf"), 4.0)
        )

    def test_detector_train_ignore_metric_and_components(self) -> None:
        torch = self.torch
        model = build_detector(base_channels=2)
        images = torch.rand(2, 3, 20, 24)
        target = torch.zeros(2, 1, 20, 24)
        target[0, 0, 3:8, 2:10] = 1
        target[1, 0, 11:17, 13:21] = 1
        ignore = torch.zeros_like(target)
        ignore[:, :, :2, :4] = 1
        fixed = model(images).detach()
        changed = target.clone()
        changed[ignore.bool()] = 1
        first = compute_detector_loss(fixed, target, ignore)
        second = compute_detector_loss(fixed, changed, ignore)
        for key in ("loss", "bce", "dice"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        before = model.head.weight.detach().clone()
        result = detector_train_step(
            model,
            {"images": images, "target": target, "ignore": ignore},
            optimizer,
            max_grad_norm=2.0,
        )
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertFalse(torch.equal(before, model.head.weight.detach()))

        probabilities = target.numpy()
        boxes = [
            torch.tensor([[2, 3, 10, 8]], dtype=torch.float32),
            torch.tensor([[13, 11, 21, 17]], dtype=torch.float32),
        ]
        metrics = detection_box_metrics(
            probabilities,
            ground_truth_boxes=boxes,
            ignore_masks=ignore.numpy(),
            threshold=0.5,
            min_area=1,
        )
        self.assertEqual(1.0, metrics["hmean"])
        self.assertEqual(2, metrics["true_positives"])
        components = extract_text_boxes(probabilities[0, 0], min_area=1)
        self.assertEqual((2.0, 3.0, 10.0, 8.0), components[0].xyxy())

    def test_crnn_train_eval_and_strict_ctc_feasibility(self) -> None:
        torch = self.torch
        charset = CTCCharset(("A", "B"))
        model = build_recognizer(
            charset, base_channels=2, hidden_size=4
        )
        batch = _recognition_batch(torch)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        before = model.classifier.weight.detach().clone()
        training = recognizer_train_step(
            model, batch, optimizer, max_grad_norm=2.0
        )
        self.assertTrue(math.isfinite(training["loss"]))
        self.assertFalse(torch.equal(before, model.classifier.weight.detach()))
        evaluation = recognizer_eval_step(model, batch, charset)
        self.assertTrue(math.isfinite(evaluation["loss"]))
        self.assertGreaterEqual(evaluation["strict_exact"], 0.0)
        self.assertGreaterEqual(evaluation["cer"], 0.0)

        model.eval()
        short = torch.rand(1, 1, 48, 20)
        padded = torch.ones(2, 1, 48, 32)
        padded[0, :, :, :20] = short[0]
        padded[1] = torch.rand(1, 48, 32)
        with torch.inference_mode():
            alone = model(short, torch.tensor([20]))
            together = model(padded, torch.tensor([20, 32]))
        valid_steps = int(recognition_output_lengths(torch.tensor([20]))[0])
        torch.testing.assert_close(
            alone[:valid_steps, 0], together[:valid_steps, 0], rtol=1e-5, atol=1e-6
        )

        impossible_logits = torch.zeros(2, 1, charset.num_classes)
        repeated = torch.tensor([1, 1], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "needs 3 timesteps"):
            validate_ctc_batch(
                impossible_logits,
                repeated,
                torch.tensor([2]),
                torch.tensor([2]),
            )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            validate_ctc_batch(
                torch.zeros(4, 1, charset.num_classes),
                torch.tensor([1.0]),
                torch.tensor([1]),
                torch.tensor([4]),
            )
        self.assertEqual([8, 2], recognition_output_lengths(torch.tensor([32, 5])).tolist())


    def test_ctc_beam_candidates_are_bounded_and_deterministic(self) -> None:
        torch = self.torch
        charset = CTCCharset(("A", "B"))
        logits = torch.tensor(
            [
                [[0.0, 4.0, 1.0]],
                [[2.0, 4.0, 0.0]],
                [[4.0, 1.0, 0.0]],
            ]
        )
        lengths = torch.tensor([3], dtype=torch.long)

        self.assertEqual(
            [[]],
            ctc_beam_decode(logits, lengths, charset, candidate_count=0),
        )
        first = ctc_beam_decode(logits, lengths, charset, candidate_count=3)
        second = ctc_beam_decode(logits, lengths, charset, candidate_count=3)
        self.assertEqual(first, second)
        self.assertEqual(["A", "BA", "AA"], first[0])
        self.assertLessEqual(len(first[0]), 3)
        self.assertEqual(len(first[0]), len(set(first[0])))
        self.assertTrue(all(first[0]))

    def test_bundle_round_trip_and_adapter_real_two_stage_inference(self) -> None:
        torch = self.torch
        charset = CTCCharset(("A", "B"))
        detector = build_detector(base_channels=2)
        recognizer = build_recognizer(
            charset, base_channels=2, hidden_size=4
        )
        # Full-image text component plus an always-A CTC sequence makes the
        # adapter behavior deterministic without replacing either real model.
        with torch.no_grad():
            for parameter in detector.parameters():
                parameter.zero_()
            detector.head.bias.fill_(8.0)
            for parameter in recognizer.parameters():
                parameter.zero_()
            recognizer.classifier.bias[1] = 8.0
        preprocess = default_ocr_preprocess()
        preprocess["detector_size"] = [32, 32]
        preprocess["detector_min_area"] = 4
        lineage = {
            "dataset": "synthetic",
            "detection_index_sha256": "a" * 64,
            "recognition_index_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path = Path(temporary) / "ocr.bundle.pt"
            info = save_ocr_bundle(
                bundle_path,
                detector,
                recognizer,
                charset=charset,
                preprocess=preprocess,
                lineage=lineage,
                epoch=2,
                global_step=9,
                metrics={"strict_exact": 0.5, "hmean": 0.75},
            )
            self.assertEqual(
                hashlib.sha256(bundle_path.read_bytes()).hexdigest(), info.sha256
            )
            loaded = load_ocr_bundle(
                bundle_path,
                expected_sha256=info.sha256,
                expected_lineage=lineage,
            )
            self.assertEqual(charset.characters, loaded.charset.characters)
            self.assertEqual(2, loaded.epoch)
            with self.assertRaisesRegex(OCRBundleError, "lineage differs"):
                load_ocr_bundle(
                    bundle_path, expected_lineage={"dataset": "wrong"}
                )

            adapter = factory(
                bundle_path,
                {
                    "task": "ocr",
                    "sha256": info.sha256,
                    "device": "cpu",
                    "detector_threshold": 0.5,
                    "recognition_threshold": 0.5,
                    "detector_batch_size": 2,
                    "recognition_batch_size": 2,
                },
            )
            views = [
                _view("view:first", (24, 16), (255, 255, 255)),
                _view("view:second", (18, 12), (100, 100, 100)),
            ]
            first = adapter.recognize(views)
            second = adapter.recognize(views)
            self.assertEqual(
                [item.model_dump(mode="json") for item in first],
                [item.model_dump(mode="json") for item in second],
            )
            self.assertEqual(["A", "A"], [item.raw_text for item in first])
            self.assertEqual(
                ["view:first", "view:second"],
                [item.evidence[0].view_id for item in first],
            )
            self.assertEqual(2, len({item.evidence[0].evidence_id for item in first}))
            for item, view in zip(first, views, strict=True):
                self.assertEqual([], item.candidates)
                self.assertIsNone(item.resolved_text)
                self.assertEqual("unknown", item.category)
                self.assertEqual("", item.details)
                self.assertIsNone(item.component_id)
                self.assertIsNone(item.pin_id)
                self.assertEqual(0.0, item.bbox.x1)
                self.assertEqual(0.0, item.bbox.y1)
                self.assertEqual(float(view.width), item.bbox.x2)
                self.assertEqual(float(view.height), item.bbox.y2)
                self.assertEqual(1, len(item.evidence))
                self.assertFalse(
                    item.evidence[0].metadata["semantic_fields_inferred"]
                )
            with self.assertRaisesRegex(TorchOCRAdapterError, "view_id.*unique"):
                adapter.recognize([views[0], views[0]])

    def test_torchscript_exports_are_separate_and_consistent(self) -> None:
        torch = self.torch
        charset = CTCCharset(("A", "B"))
        detector = build_detector(base_channels=2)
        recognizer = build_recognizer(
            charset, base_channels=2, hidden_size=4
        )
        with tempfile.TemporaryDirectory() as temporary:
            detector_path = Path(temporary) / "detector.ts"
            recognizer_path = Path(temporary) / "recognizer.ts"
            detector_report = export_ocr_submodel(
                detector,
                detector_path,
                kind="detector",
                format="torchscript",
                input_shape=(1, 3, 17, 19),
            )
            recognizer_report = export_ocr_submodel(
                recognizer,
                recognizer_path,
                kind="recognizer",
                format="torchscript",
                input_shape=(1, 1, 48, 31),
            )
            self.assertTrue(detector_report.passed)
            self.assertTrue(recognizer_report.passed)
            self.assertNotEqual(detector_report.artifact_sha256, recognizer_report.artifact_sha256)
            self.assertTrue(
                verify_ocr_export(
                    detector,
                    detector_path,
                    format="torchscript",
                    inputs=torch.rand(1, 3, 15, 21),
                )["passed"]
            )
            self.assertTrue(
                verify_ocr_export(
                    recognizer,
                    recognizer_path,
                    format="torchscript",
                    inputs=torch.rand(1, 1, 48, 27),
                )["passed"]
            )

    @unittest.skipUnless(
        ONNX_RUNTIME_AVAILABLE, "ONNX + ONNX Runtime are required"
    )
    def test_detector_and_recognizer_onnx_consistency(self) -> None:
        charset = CTCCharset(("A", "B"))
        detector = build_detector(base_channels=2)
        recognizer = build_recognizer(
            charset, base_channels=2, hidden_size=4
        )
        with tempfile.TemporaryDirectory() as temporary:
            detector_path = Path(temporary) / "detector.onnx"
            recognizer_path = Path(temporary) / "recognizer.onnx"
            detector_report = export_ocr_submodel(
                detector,
                detector_path,
                kind="detector",
                format="onnx",
                input_shape=(1, 3, 16, 20),
            )
            recognizer_report = export_ocr_submodel(
                recognizer,
                recognizer_path,
                kind="recognizer",
                format="onnx",
                input_shape=(1, 1, 48, 28),
            )
            self.assertTrue(detector_report.passed)
            self.assertTrue(recognizer_report.passed)
            self.assertTrue(
                verify_ocr_export(
                    detector,
                    detector_path,
                    format="onnx",
                    inputs=self.torch.rand(2, 3, 19, 23),
                )["passed"]
            )
            self.assertTrue(
                verify_ocr_export(
                    recognizer,
                    recognizer_path,
                    format="onnx",
                    inputs=self.torch.rand(2, 1, 48, 36),
                )["passed"]
            )

    def test_official_recognition_lazy_sample_smoke(self) -> None:
        materialized = (
            REPOSITORY_ROOT
            / "artifacts"
            / "training"
            / "materialized"
            / "dataset300-v2"
            / "ocr"
        )
        source = REPOSITORY_ROOT / "Dataset300" / "Dataset300"
        if not (materialized / "recognition_index.jsonl").is_file() or not source.is_dir():
            self.skipTest("official OCR freeze is not present")
        dataset = TorchOCRRecognitionDataset(
            materialized,
            source_root=source,
            split="train",
            image_mode="lazy",
            image_height=16,
        )
        sample = dataset[0]
        self.assertEqual(1, sample["image"].shape[0])
        self.assertEqual(16, sample["image"].shape[1])
        self.assertGreater(sample["target"].numel(), 0)
        self.assertEqual("lazy", sample["metadata"]["image_origin"])
        batch = ctc_collate([sample])
        model = build_recognizer(
            dataset.charset, base_channels=2, hidden_size=4
        )
        optimizer = self.torch.optim.Adam(model.parameters(), lr=1.0e-3)
        training = recognizer_train_step(model, batch, optimizer)
        self.assertTrue(math.isfinite(training["loss"]))

    def test_train_only_charset_oov_evaluation_contract(self) -> None:
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            materialized, source = _make_oov_evaluation_freeze(Path(temporary))
            # The training Dataset must remain fail-closed and must not expand
            # its vocabulary from validation references.
            with self.assertRaisesRegex(TorchOCRDataError, "charset-OOV"):
                TorchOCRRecognitionDataset(
                    materialized,
                    source_root=source,
                    split="val",
                    image_mode="lazy",
                )
            dataset = TorchOCREvaluationDataset(
                materialized,
                source_root=source,
                split="val",
                image_mode="lazy",
            )
            self.assertEqual(("A", "B"), dataset.charset.characters)
            self.assertNotIn("Ω", dataset.charset.characters)
            self.assertEqual(
                {
                    "samples": 3,
                    "eligible": 2,
                    "oov_samples": 1,
                    "oov_characters": 1,
                    "oov_unique_characters": 1,
                    "oov_character_values": ["Ω"],
                    "oov_codepoints": ["U+03A9"],
                    "coverage": 2 / 3,
                    "charset_source": "train_only",
                },
                dataset.coverage_summary(),
            )
            samples = [dataset[index] for index in range(len(dataset))]
            oov = next(item for item in samples if not item["eligible"])
            self.assertEqual("AΩ", oov["text"])
            self.assertEqual(["Ω"], oov["oov_characters"])
            self.assertEqual(0, oov["target"].numel())
            batch = ocr_evaluation_collate(samples)
            model = build_recognizer(
                dataset.charset, base_channels=2, hidden_size=4
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.classifier.bias[1] = 8.0
            metrics = recognizer_coverage_eval_step(
                model, batch, dataset.charset
            )
            self.assertEqual(3, metrics["samples"])
            self.assertEqual(2, metrics["eligible"])
            self.assertEqual(1, metrics["oov_samples"])
            self.assertEqual(1, metrics["oov_characters"])
            self.assertEqual(["Ω"], metrics["oov_character_values"])
            self.assertEqual(["U+03A9"], metrics["oov_codepoints"])
            self.assertEqual(2 / 3, metrics["coverage"])
            self.assertEqual(1, metrics["exact_count"])
            self.assertEqual(1 / 3, metrics["strict_exact"])
            self.assertEqual(0.5, metrics["strict_exact_eligible"])
            self.assertEqual(1, metrics["loss_eligible"])
            self.assertEqual(1, metrics["loss_ineligible_ctc"])
            self.assertEqual(1 / 3, metrics["loss_coverage"])
            self.assertEqual(0.5, metrics["loss_coverage_eligible"])
            self.assertTrue(math.isfinite(metrics["loss"]))
            # The normal entry point recognizes the evaluation contract too.
            routed = recognizer_eval_step(model, batch, dataset.charset)
            self.assertEqual(metrics["coverage"], routed["coverage"])
            self.assertEqual(metrics["strict_exact"], routed["strict_exact"])

            # Even a self-consistent forged report cannot add a validation-only
            # character to charset.txt: the evaluation loader re-derives the
            # vocabulary from included train rows.
            charset_path = materialized / "charset.txt"
            charset_path.write_text("A\nB\nΩ\n", encoding="utf-8", newline="\n")
            report_path = materialized / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["charset"]["training_characters"] = ["A", "B", "Ω"]
            report["charset"]["training_character_count"] = 3
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            manifest_path = materialized / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["charset.txt"] = hashlib.sha256(
                charset_path.read_bytes()
            ).hexdigest()
            manifest["artifacts"]["report.json"] = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(
                TorchOCREvaluationDataError,
                "exactly the character set derived.*train",
            ):
                TorchOCREvaluationDataset(
                    materialized,
                    source_root=source,
                    split="val",
                    image_mode="lazy",
                )

    def test_official_val_test_oov_rows_are_retained_and_reported(self) -> None:
        materialized = (
            REPOSITORY_ROOT
            / "artifacts"
            / "training"
            / "materialized"
            / "dataset300-v2"
            / "ocr"
        )
        source = REPOSITORY_ROOT / "Dataset300" / "Dataset300"
        if not (materialized / "recognition_index.jsonl").is_file() or not source.is_dir():
            self.skipTest("official OCR freeze is not present")
        expected = {
            "val": (4277, 17, 23),
            "test": (5861, 18, 28),
        }
        for split, (samples, oov_samples, unique_oov) in expected.items():
            dataset = TorchOCREvaluationDataset(
                materialized,
                source_root=source,
                split=split,
                image_mode="lazy",
                image_height=16,
            )
            summary = dataset.coverage_summary()
            self.assertEqual(825, len(dataset.charset.characters))
            self.assertEqual(samples, summary["samples"])
            self.assertEqual(oov_samples, summary["oov_samples"])
            self.assertEqual(unique_oov, summary["oov_unique_characters"])
            self.assertEqual(samples - oov_samples, summary["eligible"])
            self.assertAlmostEqual(
                (samples - oov_samples) / samples, summary["coverage"]
            )
            self.assertTrue(
                set(dataset.oov_characters).isdisjoint(dataset.charset.characters)
            )
            oov_index = next(
                index
                for index, row in enumerate(dataset.records)
                if row["oov_characters"]
            )
            sample = dataset[oov_index]
            self.assertFalse(sample["eligible"])
            self.assertEqual(0, sample["target"].numel())
            self.assertTrue(sample["oov_characters"])
            model = build_recognizer(
                dataset.charset, base_channels=2, hidden_size=4
            )
            metrics = recognizer_coverage_eval_step(
                model,
                ocr_evaluation_collate([sample]),
                dataset.charset,
            )
            self.assertEqual(0, metrics["eligible"])
            self.assertEqual(1, metrics["oov_samples"])
            self.assertEqual(0.0, metrics["coverage"])
            self.assertEqual(0.0, metrics["strict_exact"])
            self.assertIsNone(metrics["loss"])
            self.assertEqual(0.0, metrics["loss_coverage"])


def _recognition_batch(torch: object) -> dict[str, object]:
    images = torch.rand(2, 1, 48, 32)
    return {
        "images": images,
        "targets": torch.tensor([1, 2, 1, 1], dtype=torch.long),
        "target_lengths": torch.tensor([2, 2], dtype=torch.long),
        "image_widths": torch.tensor([32, 32], dtype=torch.long),
        "texts": ["AB", "AA"],
    }


def _make_detection_freeze(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    materialized = root / "ocr"
    source.mkdir()
    materialized.mkdir()
    image_path = source / "fixture.png"
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(24, dtype=np.uint8)[:, None]
    Image.fromarray(pixels, mode="RGB").save(image_path)
    source_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    row = {
        "schema_version": 1,
        "tile_id": "fixture:0:0",
        "sample_id": "fixture",
        "split": "train",
        "source_domain": "synthetic",
        "source_image": "fixture.png",
        "source_image_sha256": source_sha,
        "source_box": {"left": 0, "top": 0, "right": 32, "bottom": 24},
        "output_size": [32, 32],
        "valid_size": [32, 24],
        "padding": {"right": 0, "bottom": 8},
        "positive_count": 1,
        "ignored_count": 1,
        "empty_tile": False,
        "spec_sha256": "c" * 64,
        "annotations": [
            {
                "record_id": "fixture:text:1",
                "transcription": "A",
                "ignore": False,
                "points": [[2, 2], [16, 2], [16, 12], [2, 12]],
            },
            {
                "record_id": "fixture:text:2",
                "transcription": "###",
                "ignore": True,
                "ignore_reason": "fixture",
                "points": [[10, 8], [20, 8], [20, 16], [10, 16]],
            },
        ],
        "output_image": "det/train/fixture.png",
        "output_image_sha256": None,
        "transform": {},
        "source_format": "PNG",
    }
    index_path = materialized / "detection_index.jsonl"
    index_path.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    policy = {
        "bbox_points": "top_left_top_right_bottom_right_bottom_left_clockwise",
        "detection_tile_overlap": 4,
        "detection_tile_size": 32,
        "empty_detection_tiles_retained": True,
        "partial_detection_box": "clip_intersection_and_mark_###_ignore",
    }
    report_path = materialized / "report.json"
    report_path.write_text(
        json.dumps({"schema_version": 1, "policy": policy}, sort_keys=True),
        encoding="utf-8",
    )
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(
        json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "pcbparse_ocr_paddle_materialization",
        "complete": True,
        "materialize_images": False,
        "counts": {"detection_tiles": 1},
        "artifacts": {
            "detection_index.jsonl": index_sha,
            "report.json": report_sha,
        },
        "policy_sha256": policy_sha,
    }
    (materialized / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return materialized, source


def _make_oov_evaluation_freeze(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    materialized = root / "ocr-evaluation"
    source.mkdir()
    materialized.mkdir()
    image_path = source / "fixture.png"
    pixels = np.full((24, 48, 3), 255, dtype=np.uint8)
    pixels[5:18, 4:18] = 20
    pixels[5:18, 28:44] = 80
    Image.fromarray(pixels, mode="RGB").save(image_path)
    source_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def row(
        record_id: str,
        content: str,
        left: int,
        right: int,
        split: str = "val",
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_id": record_id,
            "sample_id": "fixture",
            "split": split,
            "status": "included",
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "text_length_codepoints": len(content),
            "normalization_applied": False,
            "exclusion_reason": None,
            "source_domain": "synthetic",
            "source_image": "fixture.png",
            "source_image_sha256": source_sha,
            "output_image": f"rec/{split}/{record_id.replace(':', '_')}.png",
            "output_image_sha256": None,
            "crop_box": {"left": left, "top": 2, "right": right, "bottom": 22},
            "vertical_long": False,
        }

    rows = [
        row("train:charset", "AB", 0, 22, "train"),
        row("val:ctc-long", "AAAAA", 22, 24),
        row("val:eligible", "A", 0, 22),
        row("val:oov", "AΩ", 24, 48),
    ]
    index_path = materialized / "recognition_index.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    charset_path = materialized / "charset.txt"
    charset_path.write_text("A\nB\n", encoding="utf-8", newline="\n")
    policy = {
        "max_text_length": 64,
        "overlength": "exclude_not_truncate",
        "raw_text": "preserved_without_strip_normalization_or_truncation",
    }
    report = {
        "schema_version": 1,
        "policy": policy,
        "charset": {
            "training_characters": ["A", "B"],
            "training_character_count": 2,
            "oov_by_split": {
                "val": {
                    "characters": ["Ω"],
                    "codepoints": ["U+03A9"],
                    "character_count": 1,
                    "record_ids": ["val:oov"],
                    "record_count": 1,
                }
            },
        },
    }
    report_path = materialized / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (index_path, charset_path, report_path)
    }
    policy_sha = hashlib.sha256(
        json.dumps(
            policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "pcbparse_ocr_paddle_materialization",
        "complete": True,
        "materialize_images": False,
        "counts": {
            "recognition_index": 4,
            "recognition_included": 4,
            "recognition_excluded": 0,
        },
        "artifacts": artifacts,
        "policy_sha256": policy_sha,
    }
    (materialized / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return materialized, source


def _view(
    view_id: str, size: tuple[int, int], colour: tuple[int, int, int]
) -> ImageView:
    return ImageView(
        view_id=view_id,
        image=Image.new("RGB", size, colour),
        transform=TransformChain([AffineTransform.identity(view_id)]),
        sha256=hashlib.sha256(f"{view_id}:{size}:{colour}".encode()).hexdigest(),
    )


if __name__ == "__main__":
    unittest.main()
