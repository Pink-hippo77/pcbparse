from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pcbparse.training.runner import TrainingBackendResult
from pcbparse.training.torch_backend import (
    TorchSmokeRunComplete,
    TorchTrainingError,
    _MaterializationBinding,
    _TaskOutcome,
    _RunLimits,
    _component_detector_kwargs,
    _execute_task,
    _data_loader,
    _evaluate_gates,
    _gate_selection_summary,
    _initialization_lineage,
    _load_pin_pretrained_checkpoint,
    _load_ocr_pretrained_bundle,
    _ocr_detail_metric_name,
    _ocr_inventory,
    _ocr_val_detail_index,
    _model_selection_policy,
    _pin_detection_summary,
    _prepare_run_output,
    _restored_selection_key,
    _restore_training_state,
    _run_epochs,
    _save_training_state,
    _selection_is_improved,
    _set_cosine_epoch_learning_rate,
    _verify_task_materialization,
    _wire_centerline_summary,
    _wire_evaluate,
    run_torch_task,
)
from pcbparse.training.torch_train_cli import build_parser


_HASHES = {
    name: hashlib.sha256(name.encode("ascii")).hexdigest()
    for name in (
        "config",
        "summary",
        "source",
        "split",
        "component",
        "pin",
        "ocr",
        "wire",
        "junction",
        "training",
        "binding",
    )
}
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ComponentDetectorContractTests(unittest.TestCase):
    def test_declared_score_threshold_reaches_torchvision_roi_heads(self) -> None:
        task_config = SimpleNamespace(
            input_size=(768, 768),
            component_score_threshold=0.0,
            component_box_detections_per_img=300,
        )
        limits = _RunLimits(
            smoke=False,
            epochs=1,
            max_train_batches=None,
            max_val_batches=None,
            export_samples=2,
        )

        detector_kwargs = _component_detector_kwargs(task_config, limits)

        self.assertEqual(0.0, detector_kwargs["box_score_thresh"])
        self.assertEqual(300, detector_kwargs["box_detections_per_img"])


class ModelSelectionPolicyTests(unittest.TestCase):
    def test_secondary_gate_improvement_beats_small_primary_regression(self) -> None:
        gates = (
            SimpleNamespace(
                metric="normalized_exact_match", operator="gte", threshold=0.65
            ),
            SimpleNamespace(metric="detector_hmean", operator="gte", threshold=0.45),
            SimpleNamespace(
                metric="recognition_designator_exact_match",
                operator="gte",
                threshold=0.9,
            ),
        )
        epoch_two = _gate_selection_summary(
            gates,
            {
                "normalized_exact_match": 0.6883,
                "detector_hmean": 0.3517,
                "recognition_designator_exact_match": 0.912,
            },
        )
        epoch_three = _gate_selection_summary(
            gates,
            {
                "normalized_exact_match": 0.6858,
                "detector_hmean": 0.3701,
                "recognition_designator_exact_match": 0.906,
            },
        )

        self.assertLess(tuple(epoch_three["key"]), tuple(epoch_two["key"]))
        self.assertEqual(1, epoch_three["failed_gate_count"])

    def test_all_gate_pass_uses_primary_metric_as_tie_breaker(self) -> None:
        gates = (
            SimpleNamespace(metric="f1", operator="gte", threshold=0.5),
            SimpleNamespace(metric="ratio", operator="lte", threshold=0.7),
        )
        lower_primary = _gate_selection_summary(gates, {"f1": 0.60, "ratio": 0.60})
        higher_primary = _gate_selection_summary(gates, {"f1": 0.65, "ratio": 0.69})

        self.assertTrue(lower_primary["all_gates_passed"])
        self.assertLess(tuple(higher_primary["key"]), tuple(lower_primary["key"]))

    def test_missing_secondary_metric_fails_closed(self) -> None:
        gates = (
            SimpleNamespace(metric="f1", operator="gte", threshold=0.5),
            SimpleNamespace(metric="recall", operator="gte", threshold=0.4),
        )
        with self.assertRaisesRegex(TorchTrainingError, "recall"):
            _gate_selection_summary(gates, {"f1": 0.7})

    def test_single_gate_preserves_legacy_direction_and_epsilon(self) -> None:
        for operator, best_value, tiny_value, improved_value in (
            ("gte", 0.6, 0.6 + 0.5e-12, 0.6 + 2.0e-12),
            ("lte", 0.6, 0.6 - 0.5e-12, 0.6 - 2.0e-12),
        ):
            with self.subTest(operator=operator):
                gates = (
                    SimpleNamespace(metric="score", operator=operator, threshold=0.5),
                )
                best = _gate_selection_summary(gates, {"score": best_value})
                best_key = tuple(float(item) for item in best["key"])
                tiny = _gate_selection_summary(gates, {"score": tiny_value})
                improved = _gate_selection_summary(gates, {"score": improved_value})

                self.assertFalse(
                    _selection_is_improved(
                        gates,
                        tiny,
                        best_key=best_key,
                        best_primary_value=best_value,
                    )
                )
                self.assertTrue(
                    _selection_is_improved(
                        gates,
                        improved,
                        best_key=best_key,
                        best_primary_value=best_value,
                    )
                )

    def test_multi_gate_key_applies_each_declared_priority(self) -> None:
        gates = (
            SimpleNamespace(metric="primary", operator="gte", threshold=1.0),
            SimpleNamespace(metric="secondary", operator="gte", threshold=1.0),
            SimpleNamespace(metric="tertiary", operator="lte", threshold=1.0),
        )
        two_small_failures = _gate_selection_summary(
            gates, {"primary": 1.1, "secondary": 0.9, "tertiary": 1.1}
        )
        one_large_failure = _gate_selection_summary(
            gates, {"primary": 1.1, "secondary": 0.0, "tertiary": 0.9}
        )
        lower_total = _gate_selection_summary(
            gates, {"primary": 0.8, "secondary": 0.9, "tertiary": 0.9}
        )
        higher_total = _gate_selection_summary(
            gates, {"primary": 0.8, "secondary": 1.0, "tertiary": 1.2}
        )

        self.assertLess(
            tuple(one_large_failure["key"]), tuple(two_small_failures["key"])
        )
        self.assertEqual(
            lower_total["maximum_normalized_violation"],
            higher_total["maximum_normalized_violation"],
        )
        self.assertLess(
            lower_total["total_normalized_violation"],
            higher_total["total_normalized_violation"],
        )
        self.assertLess(tuple(lower_total["key"]), tuple(higher_total["key"]))

    def test_selection_pass_state_matches_formal_gate_evaluator(self) -> None:
        gates = (
            SimpleNamespace(metric="quality", operator="gte", threshold=0.7),
            SimpleNamespace(metric="error", operator="lte", threshold=0.2),
        )
        for metrics in (
            {"quality": 0.8, "error": 0.1},
            {"quality": 0.8, "error": 0.3},
        ):
            with self.subTest(metrics=metrics):
                selection = _gate_selection_summary(gates, metrics)
                formal_rows, formal_passed = _evaluate_gates(
                    SimpleNamespace(evaluation_gates=gates), metrics
                )
                self.assertEqual(formal_passed, selection["all_gates_passed"])
                self.assertEqual(
                    [row["metric"] for row in formal_rows if not row["passed"]],
                    [row["metric"] for row in selection["failed_gates"]],
                )

    def test_resume_replay_validates_best_epoch_and_stale_count(self) -> None:
        gates = (
            SimpleNamespace(metric="primary", operator="gte", threshold=0.65),
            SimpleNamespace(metric="detector", operator="gte", threshold=0.45),
        )
        history = [
            {"epoch": 1, "val": {"primary": 0.6785, "detector": 0.3463}},
            {"epoch": 2, "val": {"primary": 0.6883, "detector": 0.3517}},
            {"epoch": 3, "val": {"primary": 0.6858, "detector": 0.3701}},
        ]

        key = _restored_selection_key(
            gates,
            history,
            best_epoch=3,
            best_value=0.6858,
            stale_epochs=0,
        )
        self.assertEqual(
            tuple(_gate_selection_summary(gates, history[2]["val"])["key"]), key
        )
        with self.assertRaisesRegex(TorchTrainingError, "best epoch"):
            _restored_selection_key(
                gates,
                history,
                best_epoch=2,
                best_value=0.6883,
                stale_epochs=1,
            )
        with self.assertRaisesRegex(TorchTrainingError, "stale epoch"):
            _restored_selection_key(
                gates,
                history,
                best_epoch=3,
                best_value=0.6858,
                stale_epochs=1,
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
    def test_epoch_loop_selects_ocr_e3_and_keeps_early_stop_consistent(self) -> None:
        import torch

        gates = (
            SimpleNamespace(
                metric="normalized_exact_match", operator="gte", threshold=0.65
            ),
            SimpleNamespace(metric="detector_hmean", operator="gte", threshold=0.45),
            SimpleNamespace(
                metric="recognition_designator_exact_match",
                operator="gte",
                threshold=0.9,
            ),
        )
        validation = {
            1: {
                "normalized_exact_match": 0.6785129763853168,
                "detector_hmean": 0.34625195424997945,
                "recognition_designator_exact_match": 0.9148351648351648,
            },
            2: {
                "normalized_exact_match": 0.6883329436520926,
                "detector_hmean": 0.35172127747822474,
                "recognition_designator_exact_match": 0.9120879120879121,
            },
            3: {
                "normalized_exact_match": 0.6857610474631751,
                "detector_hmean": 0.37008913832948165,
                "recognition_designator_exact_match": 0.9065934065934066,
            },
            4: {
                "normalized_exact_match": 0.68,
                "detector_hmean": 0.34,
                "recognition_designator_exact_match": 0.91,
            },
        }
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        best_checkpoint_epochs: list[int] = []

        def train_epoch(epoch: int) -> dict[str, float]:
            with torch.no_grad():
                model.weight.fill_(float(epoch))
            return {"loss": float(5 - epoch)}

        def save_checkpoint(path: Path, epoch: int, _metrics: object) -> None:
            if path.name == "best_checkpoint.pt":
                best_checkpoint_epochs.append(epoch)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = _run_epochs(
                torch=torch,
                model=model,
                optimizer=optimizer,
                task_config=SimpleNamespace(
                    evaluation_gates=gates,
                    gradient_accumulation=1,
                ),
                training_config=SimpleNamespace(
                    runtime=SimpleNamespace(
                        early_stopping_patience=1,
                        checkpoint_every_epochs=1,
                    )
                ),
                limits=_RunLimits(
                    smoke=True,
                    epochs=5,
                    max_train_batches=1,
                    max_val_batches=1,
                    export_samples=1,
                ),
                train_epoch=train_epoch,
                validate=lambda epoch: validation[epoch],
                save_checkpoint=save_checkpoint,
                output=root,
                log_path=root / "training_log.jsonl",
                device=torch.device("cpu"),
                scaler=SimpleNamespace(),
                loaders=(),
                lineage={},
                resume_checkpoint=None,
            )

        self.assertEqual(3, result.best_epoch)
        self.assertEqual(4, result.completed_epochs)
        self.assertTrue(result.early_stopped)
        self.assertEqual([1, 2, 3], best_checkpoint_epochs)
        self.assertEqual(3.0, float(model.weight.detach().item()))

    def test_policy_lineage_declares_full_order_and_gates(self) -> None:
        gates = (
            SimpleNamespace(metric="quality", operator="gte", threshold=0.7),
            SimpleNamespace(metric="error", operator="lte", threshold=0.2),
        )
        policy = _model_selection_policy(SimpleNamespace(evaluation_gates=gates))
        self.assertEqual("all_gate_normalized_violation_v1", policy["name"])
        self.assertEqual(4, len(policy["ordering"]))
        self.assertEqual(
            ["quality", "error"], [row["metric"] for row in policy["gates"]]
        )
        self.assertEqual(
            1e-12, policy["comparison"]["single_gate_absolute_epsilon"]
        )
        self.assertIn("threshold", policy["violation_normalization"])


class PinDetectionSummaryTests(unittest.TestCase):
    def test_micro_metrics_expose_false_positive_cost_hidden_by_pck(self) -> None:
        metrics = _pin_detection_summary(
            matched_total=50.0,
            pred_total=200.0,
            gt_total=100.0,
        )

        self.assertEqual(0.25, metrics["pin_precision_4px"])
        self.assertEqual(0.5, metrics["pin_recall_4px"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["pin_f1_4px"])
        self.assertEqual(2.0, metrics["pred_to_gt_ratio"])

    def test_empty_detection_population_is_perfect_only_when_both_sides_empty(self) -> None:
        empty = _pin_detection_summary(
            matched_total=0.0,
            pred_total=0.0,
            gt_total=0.0,
        )
        missed = _pin_detection_summary(
            matched_total=0.0,
            pred_total=0.0,
            gt_total=3.0,
        )

        self.assertEqual(1.0, empty["pin_f1_4px"])
        self.assertEqual(0.0, missed["pin_f1_4px"])

    def test_impossible_match_total_fails_closed(self) -> None:
        with self.assertRaises(TorchTrainingError):
            _pin_detection_summary(
                matched_total=2.0,
                pred_total=1.0,
                gt_total=2.0,
            )


class WireCenterlineSummaryTests(unittest.TestCase):
    def test_micro_summary_weights_pixels_instead_of_tiles(self) -> None:
        metrics = _wire_centerline_summary(
            precision_hits=90.0,
            recall_hits=80.0,
            predicted_skeleton_pixels=100.0,
            target_skeleton_pixels=200.0,
        )

        self.assertEqual(0.9, metrics["centerline_precision"])
        self.assertEqual(0.4, metrics["centerline_recall"])
        self.assertAlmostEqual(2.0 * 0.9 * 0.4 / 1.3, metrics["skeleton_f1"])

    def test_empty_population_convention_and_impossible_hits_fail_closed(self) -> None:
        empty = _wire_centerline_summary(
            precision_hits=0.0,
            recall_hits=0.0,
            predicted_skeleton_pixels=0.0,
            target_skeleton_pixels=0.0,
        )
        missed = _wire_centerline_summary(
            precision_hits=0.0,
            recall_hits=0.0,
            predicted_skeleton_pixels=0.0,
            target_skeleton_pixels=3.0,
        )

        self.assertEqual(1.0, empty["skeleton_f1"])
        self.assertEqual(0.0, missed["skeleton_f1"])
        with self.assertRaises(TorchTrainingError):
            _wire_centerline_summary(
                precision_hits=2.0,
                recall_hits=1.0,
                predicted_skeleton_pixels=1.0,
                target_skeleton_pixels=1.0,
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch runtime is not installed")
    def test_wire_evaluate_aggregates_raw_counts_globally(self) -> None:
        import torch

        class IdentityModel:
            def eval(self) -> None:
                return None

            def __call__(self, image):
                return image[:, :1]

        loader = [
            {
                "image": torch.zeros(1, 3, 2, 2),
                "target": torch.zeros(1, 1, 2, 2),
                "ignore": torch.zeros(1, 1, 2, 2),
            },
            {
                "image": torch.zeros(1, 3, 2, 2),
                "target": torch.zeros(1, 1, 2, 2),
                "ignore": torch.zeros(1, 1, 2, 2),
            },
        ]
        metric_rows = iter(
            [
                {
                    "centerline_precision": 1.0,
                    "centerline_recall": 1.0,
                    "centerline_f1": 1.0,
                    "skeleton_f1": 1.0,
                    "precision_hits": 0.0,
                    "recall_hits": 0.0,
                    "predicted_skeleton_pixels": 0.0,
                    "target_skeleton_pixels": 0.0,
                    "connectivity_score": 1.0,
                    "connectivity_score_sum": 1.0,
                    "predicted_components": 0.0,
                    "target_components": 0.0,
                    "component_count_error": 0.0,
                    "component_count_error_sum": 0.0,
                },
                {
                    "centerline_precision": 0.5,
                    "centerline_recall": 0.2,
                    "centerline_f1": 2.0 / 7.0,
                    "skeleton_f1": 2.0 / 7.0,
                    "precision_hits": 5.0,
                    "recall_hits": 4.0,
                    "predicted_skeleton_pixels": 10.0,
                    "target_skeleton_pixels": 20.0,
                    "connectivity_score": 0.5,
                    "connectivity_score_sum": 0.5,
                    "predicted_components": 4.0,
                    "target_components": 1.0,
                    "component_count_error": 3.0,
                    "component_count_error_sum": 3.0,
                },
            ]
        )
        loss_rows = iter((1.0, 3.0))

        metrics, examples = _wire_evaluate(
            torch,
            IdentityModel(),
            loader,
            torch.device("cpu"),
            None,
            lambda *_args, **_kwargs: {"loss": torch.tensor(next(loss_rows))},
            lambda *_args, **_kwargs: next(metric_rows),
            max_batches=None,
            predictions_path=None,
        )

        self.assertEqual(2, examples)
        self.assertEqual(0.5, metrics["centerline_precision"])
        self.assertEqual(0.2, metrics["centerline_recall"])
        self.assertAlmostEqual(2.0 / 7.0, metrics["skeleton_f1"])
        self.assertEqual(0.75, metrics["connectivity_score"])
        self.assertEqual(1.5, metrics["component_count_error"])
        self.assertEqual(2.0, metrics["validation_loss"])


class CosineEpochLearningRateTests(unittest.TestCase):
    def test_absolute_epoch_schedule_does_not_depend_on_prior_state(self) -> None:
        optimizer = SimpleNamespace(param_groups=[{"lr": 99.0}])

        first = _set_cosine_epoch_learning_rate(
            optimizer,
            epoch=1,
            planned_epochs=10,
            base_learning_rate=1.0,
            minimum_learning_rate=0.1,
        )
        fifth = _set_cosine_epoch_learning_rate(
            optimizer,
            epoch=5,
            planned_epochs=10,
            base_learning_rate=1.0,
            minimum_learning_rate=0.1,
        )
        resumed_fifth = _set_cosine_epoch_learning_rate(
            SimpleNamespace(param_groups=[{"lr": -1.0}]),
            epoch=5,
            planned_epochs=10,
            base_learning_rate=1.0,
            minimum_learning_rate=0.1,
        )

        self.assertEqual(1.0, first)
        self.assertAlmostEqual(fifth, resumed_fifth)
        self.assertLess(fifth, first)
        self.assertAlmostEqual(fifth, optimizer.param_groups[0]["lr"])

    def test_invalid_epoch_fails_closed(self) -> None:
        with self.assertRaises(TorchTrainingError):
            _set_cosine_epoch_learning_rate(
                SimpleNamespace(param_groups=[{"lr": 1.0}]),
                epoch=0,
                planned_epochs=10,
                base_learning_rate=1.0,
                minimum_learning_rate=0.1,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
class WeightedDataLoaderTests(unittest.TestCase):
    def test_component_weights_use_checkpointed_loader_generator(self) -> None:
        import torch

        runtime = SimpleNamespace(
            seed=17,
            dataloader_workers=0,
            pin_memory=False,
        )
        loader = _data_loader(
            torch,
            [0, 1, 2],
            SimpleNamespace(batch_size=1),
            SimpleNamespace(runtime=runtime),
            shuffle=True,
            collate_fn=lambda rows: rows,
            device=torch.device("cpu"),
            sample_weights=(1.0, 2.0, 4.0),
        )

        self.assertIs(loader.generator, loader.sampler.generator)
        self.assertEqual(3, loader.sampler.num_samples)
        self.assertEqual([1.0, 2.0, 4.0], loader.sampler.weights.tolist())

    def test_component_weights_fail_closed_on_wrong_cardinality(self) -> None:
        import torch

        runtime = SimpleNamespace(
            seed=17,
            dataloader_workers=0,
            pin_memory=False,
        )
        with self.assertRaisesRegex(TorchTrainingError, "one non-empty weight"):
            _data_loader(
                torch,
                [0, 1, 2],
                SimpleNamespace(batch_size=1),
                SimpleNamespace(runtime=runtime),
                shuffle=True,
                collate_fn=lambda rows: rows,
                device=torch.device("cpu"),
                sample_weights=(1.0, 2.0),
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
class OCRPretrainedWarmStartTests(unittest.TestCase):
    def _bundle_fixture(self, root: Path):
        import torch

        from pcbparse.training.torch_ocr import CTCCharset
        from pcbparse.training.torch_ocr_model import (
            build_detector,
            build_recognizer,
            default_ocr_preprocess,
            save_ocr_bundle,
        )

        charset = CTCCharset(("A", "B"))
        torch.manual_seed(31)
        source_detector = build_detector(base_channels=2)
        source_recognizer = build_recognizer(
            charset, base_channels=2, hidden_size=4
        )
        preprocess = default_ocr_preprocess()
        path = root / "ocr.pt"
        info = save_ocr_bundle(
            path,
            source_detector,
            source_recognizer,
            charset=charset,
            preprocess=preprocess,
            lineage={
                "task": "ocr",
                "architecture": "tiny_unet_crnn_ctc_v1",
                "input_size": [48, 32],
            },
            epoch=7,
            global_step=91,
        )
        return (
            torch,
            charset,
            source_detector,
            source_recognizer,
            preprocess,
            path,
            info,
        )

    def test_byte_bound_bundle_loads_only_strict_weights_and_records_audit(self) -> None:
        from pcbparse.training.torch_ocr_model import build_detector, build_recognizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                torch,
                charset,
                source_detector,
                source_recognizer,
                preprocess,
                path,
                info,
            ) = self._bundle_fixture(root)
            torch.manual_seed(99)
            target_detector = build_detector(base_channels=2)
            target_recognizer = build_recognizer(
                charset, base_channels=2, hidden_size=4
            )

            audit = _load_ocr_pretrained_bundle(
                detector=target_detector,
                recognizer=target_recognizer,
                charset=charset,
                preprocess={**preprocess, "detector_target_shrink_ratio": 0.12},
                initialization={
                    "mode": "pretrained_local_artifact",
                    "resolved_path": str(path),
                    "sha256": info.sha256,
                },
                expected_architecture="tiny_unet_crnn_ctc_v1",
                expected_input_size=(48, 32),
            )

            for source, target in (
                (source_detector, target_detector),
                (source_recognizer, target_recognizer),
            ):
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(value, target.state_dict()[name])
            self.assertEqual(info.sha256, audit["checkpoint_sha256_verified"])
            self.assertEqual(7, audit["source_epoch"])
            self.assertTrue(audit["strict_state_dict"])
            self.assertFalse(audit["optimizer_state_restored"])
            self.assertEqual(
                "weights_only_warm_start_new_optimizer", audit["policy"]
            )

    def test_charset_preprocess_and_digest_mismatches_fail_closed(self) -> None:
        from pcbparse.training.torch_ocr import CTCCharset
        from pcbparse.training.torch_ocr_model import build_detector, build_recognizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, charset, _, _, preprocess, path, info = self._bundle_fixture(root)

            def attempt(*, target_charset=charset, target_preprocess=preprocess, sha=info.sha256):
                return _load_ocr_pretrained_bundle(
                    detector=build_detector(base_channels=2),
                    recognizer=build_recognizer(
                        target_charset, base_channels=2, hidden_size=4
                    ),
                    charset=target_charset,
                    preprocess=target_preprocess,
                    initialization={
                        "mode": "pretrained_local_artifact",
                        "resolved_path": str(path),
                        "sha256": sha,
                    },
                    expected_architecture="tiny_unet_crnn_ctc_v1",
                    expected_input_size=(48, 32),
                )

            with self.assertRaisesRegex(TorchTrainingError, "charset differs"):
                attempt(target_charset=CTCCharset(("A", "C")))
            with self.assertRaisesRegex(TorchTrainingError, "preprocessing"):
                attempt(target_preprocess={**preprocess, "detector_size": [256, 256]})
            with self.assertRaisesRegex(TorchTrainingError, "SHA-256 mismatch"):
                attempt(sha="0" * 64)


class OCRValidationDetailJoinTests(unittest.TestCase):
    def test_hash_bound_val_join_canonicalizes_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "ocr_text.jsonl"
            rows = [
                {
                    "record_id": "v:1",
                    "sample_id": "v",
                    "split": "val",
                    "task": "ocr_text",
                    "details": "designator",
                    "content": "R1",
                },
                {
                    "record_id": "v:2",
                    "sample_id": "v",
                    "split": "val",
                    "task": "ocr_text",
                    "details": "pinNumber",
                    "content": "2",
                },
                {
                    "record_id": "t:1",
                    "sample_id": "t",
                    "split": "test",
                    "task": "ocr_text",
                    "details": "designator",
                    "content": "C1",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
            digest = _digest(manifest)
            inventory = SimpleNamespace(
                split="val",
                rows=(
                    {"record_id": "v:1", "sample_id": "v", "content": "R1"},
                    {"record_id": "v:2", "sample_id": "v", "content": "2"},
                ),
            )
            preflight = {
                "workspace_root": str(root),
                "tasks": {"ocr": {"path": str(manifest), "sha256": digest}},
            }

            details, audit = _ocr_val_detail_index(
                preflight, inventory, expected_sha256=digest
            )

            self.assertEqual({"v:1": "designator", "v:2": "pinnumber"}, details)
            self.assertEqual(0, audit["test_rows_used"])
            self.assertEqual(
                {"designator": 1, "pinnumber": 1}, audit["support_by_detail"]
            )

    def test_val_detail_join_rejects_manifest_or_schema_drift(self) -> None:
        self.assertEqual("unknown", _ocr_detail_metric_name(""))
        with self.assertRaisesRegex(TorchTrainingError, "unsupported"):
            _ocr_detail_metric_name("new competition field")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "ocr_text.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "record_id": "v:1",
                        "sample_id": "v",
                        "split": "val",
                        "task": "ocr_text",
                        "details": "designator",
                        "content": "C1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            digest = _digest(manifest)
            inventory = SimpleNamespace(
                split="val",
                rows=(
                    {"record_id": "v:1", "sample_id": "v", "content": "R1"},
                ),
            )
            preflight = {
                "workspace_root": str(root),
                "tasks": {"ocr": {"path": str(manifest), "sha256": digest}},
            }
            with self.assertRaisesRegex(TorchTrainingError, "differs"):
                _ocr_val_detail_index(
                    preflight, inventory, expected_sha256=digest
                )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
class ResumeConfigCompatibilityTests(unittest.TestCase):
    class _Scaler:
        def __init__(self) -> None:
            self.state: dict[str, object] = {}

        def state_dict(self) -> dict[str, object]:
            return dict(self.state)

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.state = dict(state)

    def test_additive_noop_schema_hash_can_resume_but_unknown_hash_cannot(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume_checkpoint.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = self._Scaler()
            generator = torch.Generator().manual_seed(17)
            loader = SimpleNamespace(generator=generator)
            task_config = SimpleNamespace(
                architecture="tiny_pin_unet_v1",
                input_size=(384, 384),
            )
            limits = SimpleNamespace(smoke=False, epochs=2)
            legacy_hash = "a" * 64
            current_hash = "b" * 64
            common_lineage = {
                "task": "pin",
                "materialization_commit_id": "binding-v1",
                "task_manifest_sha256": "c" * 64,
                "runtime": {"device": "cpu", "effective_precision": "fp32"},
                "model_selection": {
                    "name": "all_gate_normalized_violation_v1",
                    "ordering": [
                        "failed_gate_count_ascending",
                        "maximum_normalized_violation_ascending",
                        "total_normalized_violation_ascending",
                        "primary_metric_best",
                    ],
                    "primary_metric": "pck_4px",
                    "primary_operator": "gte",
                    "gates": [
                        {
                            "metric": "pck_4px",
                            "operator": "gte",
                            "threshold": 0.5,
                        }
                    ],
                },
            }
            _save_training_state(
                torch=torch,
                path=path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loaders=(loader,),
                task_config=task_config,
                limits=limits,
                lineage={**common_lineage, "training_config_sha256": legacy_hash},
                completed_epoch=1,
                best_epoch=1,
                best_value=0.4,
                best_model_state=model.state_dict(),
                stale_epochs=0,
                history=({"epoch": 1},),
            )

            restored = _restore_training_state(
                torch=torch,
                path=path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loaders=(loader,),
                task_config=task_config,
                limits=limits,
                lineage={
                    **common_lineage,
                    "training_config_sha256": current_hash,
                    "resume_compatible_training_config_sha256s": [
                        current_hash,
                        legacy_hash,
                    ],
                },
            )
            self.assertEqual(1, restored["completed_epoch"])

            with self.assertRaisesRegex(
                TorchTrainingError,
                "training_config_sha256 differs",
            ):
                _restore_training_state(
                    torch=torch,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loaders=(loader,),
                    task_config=task_config,
                    limits=limits,
                    lineage={
                        **common_lineage,
                        "training_config_sha256": current_hash,
                        "resume_compatible_training_config_sha256s": [current_hash],
                    },
                )

            legacy_path = Path(directory) / "legacy_resume_checkpoint.pt"
            legacy_document = torch.load(path, map_location="cpu", weights_only=False)
            mismatched_policy_path = Path(directory) / "mismatched_policy_resume.pt"
            mismatched_policy_document = dict(legacy_document)
            mismatched_policy_document["model_selection"] = {
                **mismatched_policy_document["model_selection"],
                "name": "different_policy",
            }
            torch.save(mismatched_policy_document, mismatched_policy_path)
            with self.assertRaisesRegex(TorchTrainingError, "model_selection differs"):
                _restore_training_state(
                    torch=torch,
                    path=mismatched_policy_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loaders=(loader,),
                    task_config=task_config,
                    limits=limits,
                    lineage={
                        **common_lineage,
                        "training_config_sha256": current_hash,
                        "resume_compatible_training_config_sha256s": [
                            current_hash,
                            legacy_hash,
                        ],
                    },
                )

            legacy_document["schema"] = "pcbparse.torch_training_resume.v1"
            legacy_document.pop("model_selection", None)
            torch.save(legacy_document, legacy_path)
            with self.assertRaisesRegex(TorchTrainingError, "historical v1"):
                _restore_training_state(
                    torch=torch,
                    path=legacy_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loaders=(loader,),
                    task_config=task_config,
                    limits=limits,
                    lineage={
                        **common_lineage,
                        "training_config_sha256": current_hash,
                        "resume_compatible_training_config_sha256s": [
                            current_hash,
                            legacy_hash,
                        ],
                    },
                )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
class PinWarmStartTests(unittest.TestCase):
    def test_pin_warm_start_is_hash_bound_strict_and_weights_only(self) -> None:
        from pcbparse.training.torch_pin_model import (
            TinyPinHeatmapNet,
            load_pin_checkpoint,
            save_pin_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin-source.pt"
            source = TinyPinHeatmapNet(base_channels=16)
            lineage = {
                "binding_sha256": "a" * 64,
                "task": "pin",
                "architecture": "tiny_pin_unet_v1",
                "input_size": [384, 384],
            }
            save_pin_checkpoint(
                path,
                source,
                lineage=lineage,
                epoch=6,
                global_step=42,
                metrics={"pin_f1_4px": 0.5},
            )
            target = TinyPinHeatmapNet(base_channels=16)
            audit = _load_pin_pretrained_checkpoint(
                model=target,
                load_pin_checkpoint=load_pin_checkpoint,
                initialization={
                    "mode": "pretrained_local_artifact",
                    "resolved_path": str(path),
                    "sha256": _digest(path),
                },
                expected_architecture="tiny_pin_unet_v1",
                expected_input_size=(384, 384),
            )

            self.assertEqual(6, audit["source_epoch"])
            self.assertFalse(audit["optimizer_state_restored"])
            self.assertGreater(audit["state_elements_loaded"], 0)
            for name, value in source.state_dict().items():
                self.assertTrue(value.equal(target.state_dict()[name]))

            with self.assertRaisesRegex(TorchTrainingError, "dimensions differ"):
                _load_pin_pretrained_checkpoint(
                    model=TinyPinHeatmapNet(base_channels=8),
                    load_pin_checkpoint=load_pin_checkpoint,
                    initialization={
                        "mode": "pretrained_local_artifact",
                        "resolved_path": str(path),
                        "sha256": _digest(path),
                    },
                    expected_architecture="tiny_pin_unet_v1",
                    expected_input_size=(384, 384),
                )


def _task_config(
    task: str = "pin", *, threshold: float = 0.5, epochs: int = 2
) -> SimpleNamespace:
    dataset_task = {
        "component": "component_detection",
        "pin": "pin_keypoints",
        "ocr": "ocr_text",
        "wire": "wire_paths",
        "junction": "junction_patches",
    }[task]
    metric = {
        "component": "bbox_map50",
        "pin": "pck_4px",
        "ocr": "normalized_exact_match",
        "wire": "skeleton_f1",
        "junction": "macro_f1",
    }[task]
    architecture = {
        "component": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pin": "tiny_pin_unet_v1",
        "ocr": "tiny_unet_crnn_ctc_v1",
        "wire": "tiny_wire_unet_v1",
        "junction": "tiny_junction_cnn_v1",
    }[task]
    input_size = {
        "component": (32, 32),
        "pin": (32, 32),
        "ocr": (48, 32),
        "wire": (32, 32),
        "junction": (32, 32),
    }[task]
    return SimpleNamespace(
        dataset_task=dataset_task,
        architecture=architecture,
        pretrained_artifact=None,
        epochs=epochs,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        input_size=input_size,
        precision="fp32",
        export_format="onnx",
        export_opset=17,
        export_max_abs_error=1e-3,
        export_min_samples=2,
        evaluation_gates=(
            SimpleNamespace(metric=metric, operator="gte", threshold=threshold),
        ),
    )


def _training_config(task: str = "pin", **task_kwargs: object) -> SimpleNamespace:
    selected = _task_config(task, **task_kwargs)
    tasks = {task: selected}
    dataset = SimpleNamespace(
        config_hash=_HASHES["config"],
        summary_sha256=_HASHES["summary"],
        source_manifest_sha256=_HASHES["source"],
        split_manifest_sha256=_HASHES["split"],
        task_manifest_sha256={
            "component_detection": _HASHES["component"],
            "pin_keypoints": _HASHES["pin"],
            "ocr_text": _HASHES["ocr"],
            "wire_paths": _HASHES["wire"],
            "junction_patches": _HASHES["junction"],
        },
    )
    runtime = SimpleNamespace(
        seed=7,
        deterministic_algorithms=True,
        max_vram_mb=1024,
        dataloader_workers=0,
        pin_memory=False,
        checkpoint_every_epochs=1,
        early_stopping_patience=1,
    )
    return SimpleNamespace(dataset=dataset, runtime=runtime, tasks=tasks)


def _preflight(root: Path, config: SimpleNamespace, task: str) -> dict[str, object]:
    task_config = config.tasks[task]
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "workspace_root": str(root),
        "training_config_sha256": _HASHES["training"],
        "dataset": {"config_hash": config.dataset.config_hash},
        "paths": {"source_root": str(source_root)},
        "tasks": {
            task: {
                "sha256": config.dataset.task_manifest_sha256[
                    task_config.dataset_task
                ]
            }
        },
    }


def _binding(root: Path, task: str) -> _MaterializationBinding:
    task_root = root / "materialized"
    task_root.mkdir(parents=True, exist_ok=True)
    return _MaterializationBinding(
        task=task,
        root=task_root,
        commit_kind="test",
        commit_id=_HASHES["binding"],
        commit_sha256=_HASHES["summary"],
        lineage={"test": True},
        task_root=task_root,
    )


def _fake_outcome(
    output: Path,
    *,
    task: str = "pin",
    metric: float = 0.8,
    planned_epochs: int = 2,
    completed_epochs: int = 2,
    early_stopped: bool = False,
) -> _TaskOutcome:
    artifact = output / "model.onnx"
    export = output / "verified.onnx"
    predictions = output / "predictions.jsonl"
    metric_name = {
        "component": "bbox_map50",
        "pin": "pck_4px",
        "ocr": "normalized_exact_match",
        "wire": "skeleton_f1",
    }[task]
    adapter = {
        "component": "pcbparse.perception.torch_component_adapter:factory",
        "pin": "pcbparse.perception.torch_pin_adapter:factory",
        "ocr": "pcbparse.perception.torch_ocr_adapter:factory",
        "wire": "pcbparse.perception.torch_wire_adapter:factory",
    }[task]
    return _TaskOutcome(
        artifact_path=artifact,
        export_path=export,
        adapter_entrypoint=adapter,
        artifact_precision="fp32",
        export_backend="fixture-onnx",
        export_sample_count=2,
        export_max_abs_error=0.0,
        metrics={metric_name: metric},
        predictions_path=predictions,
        planned_epochs=planned_epochs,
        completed_epochs=completed_epochs,
        early_stopped=early_stopped,
        best_epoch=completed_epochs,
        limitations=("fixture",),
    )


class TorchBackendContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "candidate"
        self.environment = patch.dict(
            os.environ,
            {
                "PCBPARSE_TRAIN_DEVICE": "cpu",
                "PCBPARSE_TRAIN_PRECISION": "fp32",
            },
            clear=False,
        )
        self.environment.start()
        for name in (
            "PCBPARSE_TRAIN_SMOKE",
            "PCBPARSE_TRAIN_SMOKE_EPOCHS",
            "PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES",
            "PCBPARSE_TRAIN_SMOKE_VAL_BATCHES",
            "PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES",
            "PCBPARSE_TRAIN_RESUME_CHECKPOINT",
        ):
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def _patch_runtime(self, task: str, outcome: _TaskOutcome):
        fake_torch = SimpleNamespace(__version__="test")

        def execute(**_kwargs: object) -> _TaskOutcome:
            outcome.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            outcome.artifact_path.write_bytes(b"model")
            outcome.export_path.write_bytes(b"export")
            outcome.predictions_path.write_text(
                '{"id":"sample"}\n', encoding="utf-8"
            )
            return outcome

        return (
            patch(
                "pcbparse.training.torch_backend._verify_task_materialization",
                return_value=_binding(self.root, task),
            ),
            patch("pcbparse.training.torch_backend._require_torch", return_value=fake_torch),
            patch(
                "pcbparse.training.torch_backend._configure_runtime",
                return_value=(SimpleNamespace(type="cpu", __str__=lambda self: "cpu"), "fp32"),
            ),
            patch("pcbparse.training.torch_backend._execute_task", side_effect=execute),
        )

    def test_formal_result_matches_framework_neutral_contract(self) -> None:
        config = _training_config("pin")
        report = _preflight(self.root, config, "pin")
        outcome = _fake_outcome(self.output)
        patches = self._patch_runtime("pin", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            raw = run_torch_task(
                "pin", config.tasks["pin"], config, report, self.output
            )
        result = TrainingBackendResult.model_validate(raw)
        self.assertEqual("pcbparse.perception.torch_pin_adapter:factory", result.adapter_entrypoint)
        self.assertTrue(result.export_consistency.passed)
        self.assertEqual(2, result.export_consistency.sample_count)
        self.assertEqual(0.8, result.evaluation.metrics["pck_4px"])
        self.assertEqual(
            "Official Dataset300 competition terms; redistribution not granted",
            result.license,
        )
        self.assertTrue(any("license" in item for item in result.limitations))
        status = json.loads((self.output / "torch_run_status.json").read_text("utf-8"))
        self.assertTrue(status["registerable"])
        self.assertFalse(status["smoke"])
        self.assertEqual(
            "all_gate_normalized_violation_v1",
            status["lineage"]["model_selection"]["name"],
        )
        self.assertEqual(
            "pck_4px", status["lineage"]["model_selection"]["primary_metric"]
        )
        self.assertIn(
            "model_selection_policy",
            status["lineage"]["resume"]["restored_state"],
        )
        self.assertEqual(
            "pcbparse.torch_training_resume.v2",
            status["lineage"]["resume"]["checkpoint_schema"],
        )
        self.assertFalse((self.root / "models.lock.json").exists())

    def test_failed_metric_gate_returns_no_registerable_contract(self) -> None:
        config = _training_config("pin", threshold=0.9)
        report = _preflight(self.root, config, "pin")
        outcome = _fake_outcome(self.output, metric=0.2)
        patches = self._patch_runtime("pin", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(TorchTrainingError, "validation gate failed"):
                run_torch_task("pin", config.tasks["pin"], config, report, self.output)
        status = json.loads((self.output / "torch_run_status.json").read_text("utf-8"))
        self.assertEqual("gate_failed", status["status"])
        self.assertFalse(status["registerable"])

    def test_smoke_completion_is_structurally_non_registerable(self) -> None:
        config = _training_config("pin")
        report = _preflight(self.root, config, "pin")
        outcome = _fake_outcome(
            self.output, planned_epochs=1, completed_epochs=1
        )
        os.environ.update(
            {
                "PCBPARSE_TRAIN_SMOKE": "1",
                "PCBPARSE_TRAIN_SMOKE_EPOCHS": "1",
                "PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES": "1",
                "PCBPARSE_TRAIN_SMOKE_VAL_BATCHES": "1",
                "PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES": "1",
            }
        )
        patches = self._patch_runtime("pin", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(TorchSmokeRunComplete):
                run_torch_task("pin", config.tasks["pin"], config, report, self.output)
        status = json.loads((self.output / "torch_run_status.json").read_text("utf-8"))
        self.assertEqual("smoke_complete", status["status"])
        self.assertFalse(status["formal_evidence"])
        self.assertFalse(status["registerable"])

    def test_configured_early_stop_can_produce_formal_candidate(self) -> None:
        config = _training_config("pin", epochs=4)
        report = _preflight(self.root, config, "pin")
        outcome = _fake_outcome(
            self.output,
            planned_epochs=4,
            completed_epochs=2,
            early_stopped=True,
        )
        patches = self._patch_runtime("pin", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            raw = run_torch_task(
                "pin", config.tasks["pin"], config, report, self.output
            )
        TrainingBackendResult.model_validate(raw)
        status = json.loads((self.output / "torch_run_status.json").read_text("utf-8"))
        self.assertEqual("candidate_evidence_ready", status["status"])
        self.assertTrue(status["registerable"])
        self.assertEqual("configured_early_stopping", status["termination"]["reason"])
        self.assertEqual(4, status["planned_epochs"])
        self.assertEqual(2, status["completed_epochs"])
        self.assertEqual(2, status["best_epoch"])

    def test_external_epoch_truncation_is_not_registerable(self) -> None:
        config = _training_config("pin", epochs=4)
        report = _preflight(self.root, config, "pin")
        outcome = _fake_outcome(
            self.output,
            planned_epochs=4,
            completed_epochs=2,
            early_stopped=False,
        )
        patches = self._patch_runtime("pin", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(TorchTrainingError, "external truncation"):
                run_torch_task(
                    "pin", config.tasks["pin"], config, report, self.output
                )
        status = json.loads((self.output / "torch_run_status.json").read_text("utf-8"))
        self.assertEqual("externally_truncated", status["status"])
        self.assertFalse(status["registerable"])

    def test_ocr_formal_result_uses_implemented_two_stage_adapter(self) -> None:
        config = _training_config("ocr")
        report = _preflight(self.root, config, "ocr")
        outcome = _fake_outcome(self.output, task="ocr")
        patches = self._patch_runtime("ocr", outcome)
        with patches[0], patches[1], patches[2], patches[3]:
            raw = run_torch_task(
                "ocr", config.tasks["ocr"], config, report, self.output
            )
        result = TrainingBackendResult.model_validate(raw)
        self.assertEqual(
            "pcbparse.perception.torch_ocr_adapter:factory",
            result.adapter_entrypoint,
        )
        self.assertEqual(0.8, result.evaluation.metrics["normalized_exact_match"])

    def test_ocr_dispatch_calls_real_trainer_entrypoint(self) -> None:
        expected = _fake_outcome(self.output, task="ocr")
        with patch(
            "pcbparse.training.torch_backend._train_ocr", return_value=expected
        ) as trainer:
            actual = _execute_task(task="ocr", marker="dispatch-smoke")
        self.assertIs(expected, actual)
        trainer.assert_called_once_with(task="ocr", marker="dispatch-smoke")

    def test_unknown_runtime_architecture_fails_before_torch_import(self) -> None:
        config = _training_config("wire")
        config.tasks["wire"].architecture = "different_unet"
        report = _preflight(self.root, config, "wire")
        with patch("pcbparse.training.torch_backend._require_torch") as require_torch:
            with self.assertRaisesRegex(TorchTrainingError, "architecture must be"):
                run_torch_task(
                    "wire", config.tasks["wire"], config, report, self.output
                )
        require_torch.assert_not_called()

    def test_formal_nonempty_output_requires_explicit_resume(self) -> None:
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "partial.log").write_text("partial", encoding="utf-8")
        limits = SimpleNamespace(smoke=False)
        with self.assertRaisesRegex(TorchTrainingError, "refusing to overwrite"):
            _prepare_run_output(
                occupied, workspace_root=self.root, limits=limits
            )

    def test_cli_exposes_resume_checkpoint(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--config",
                "train.json",
                "--task",
                "wire",
                "--output",
                "candidate",
                "--resume-checkpoint",
                "candidate/resume_checkpoint.pt",
            ]
        )
        self.assertEqual(
            "candidate/resume_checkpoint.pt", parsed.resume_checkpoint
        )

    def test_junction_six_class_gate_is_mandatory(self) -> None:
        config = _training_config("junction")
        report = _preflight(self.root, config, "junction")
        with patch(
            "pcbparse.training.torch_backend._verify_task_materialization",
            return_value=_binding(self.root, "junction"),
        ), patch(
            "pcbparse.training.torch_backend._reject_unready_junction",
            side_effect=TorchTrainingError("missing classes: connected_cross, noise"),
        ):
            with self.assertRaisesRegex(TorchTrainingError, "missing classes"):
                run_torch_task(
                    "junction", config.tasks["junction"], config, report, self.output
                )


class MaterializationBindingTests(unittest.TestCase):
    def test_component_pretrained_artifact_is_byte_bound_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights" / "component.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"immutable pretrained bytes")
            artifact = SimpleNamespace(
                path="weights/component.pt",
                sha256=_digest(weights),
                source="torchvision FasterRCNN MobileNet V3 COCO",
                license="BSD-3-Clause",
            )
            task_config = SimpleNamespace(pretrained_artifact=artifact)

            lineage = _initialization_lineage("component", task_config, root)

            self.assertEqual("pretrained_local_artifact", lineage["mode"])
            self.assertEqual(_digest(weights), lineage["sha256"])
            self.assertEqual("BSD-3-Clause", lineage["license"])
            weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(TorchTrainingError, "SHA-256 mismatch"):
                _initialization_lineage("component", task_config, root)

    def test_ocr_inventory_uses_train_only_charset_and_exposes_val_oov(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "record_id": "train-1",
                    "sample_id": "sample-train",
                    "split": "train",
                    "status": "included",
                    "content": "AB",
                },
                {
                    "record_id": "val-1",
                    "sample_id": "sample-val-1",
                    "split": "val",
                    "status": "included",
                    "content": "A",
                },
                {
                    "record_id": "val-oov",
                    "sample_id": "sample-val-oov",
                    "split": "val",
                    "status": "included",
                    "content": "AΩ",
                },
            ]
            (root / "recognition_index.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = {
                "charset": {
                    "training_characters": ["A", "B"],
                    "oov_by_split": {
                        "val": {
                            "record_ids": ["val-oov"],
                            "characters": ["Ω"],
                        },
                        "test": {"record_ids": [], "characters": []},
                    },
                },
                "recognition": {
                    "excluded_records": [
                        {"record_id": "val-excluded", "split": "val"}
                    ]
                },
            }
            (root / "report.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )
            charset = SimpleNamespace(characters=("A", "B"))

            train = _ocr_inventory(root, charset, split="train")
            val = _ocr_inventory(root, charset, split="val")

            self.assertEqual(0, train.oov_records)
            self.assertEqual(2, val.encodable_records + val.oov_records)
            self.assertEqual(1, val.oov_records)
            self.assertEqual(("Ω",), val.oov_characters)
            self.assertEqual(("val-oov",), val.oov_record_ids)
            self.assertEqual(1, val.excluded_records)

    def test_component_commit_and_derived_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialized = (
                root
                / "artifacts"
                / "training"
                / "materialized"
                / "dataset300-v2"
                / "component-pin"
            )
            materialized.mkdir(parents=True)
            component = materialized / "component_coco.json"
            pin = materialized / "pin_index.jsonl"
            component.write_text("{}\n", encoding="utf-8")
            pin.write_text("{}\n", encoding="utf-8")
            config = _training_config("component")
            summary = {
                "materializer_version": "component-pin-index-v1",
                "binding_sha256": _HASHES["binding"],
                "lineage": {
                    "derivation_config_sha256": _HASHES["config"],
                    "inputs": {
                        "source_manifest": {"sha256": _HASHES["source"]},
                        "split_manifest": {"sha256": _HASHES["split"]},
                        "summary": {"sha256": _HASHES["summary"]},
                        "component_manifest": {"sha256": _HASHES["component"]},
                        "pin_manifest": {"sha256": _HASHES["pin"]},
                    },
                },
                "outputs": {
                    "component_coco.json": {"sha256": _digest(component)},
                    "pin_index.jsonl": {"sha256": _digest(pin)},
                },
            }
            (materialized / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            binding = _verify_task_materialization(
                "component", config, {}, root
            )
            self.assertEqual(_HASHES["binding"], binding.commit_id)
            pin.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(TorchTrainingError, "SHA-256 mismatch"):
                _verify_task_materialization("component", config, {}, root)

    def test_cli_exposes_explicit_smoke_controls(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--config",
                "train.json",
                "--task",
                "wire",
                "--output",
                "candidate",
                "--smoke",
                "--smoke-train-batches",
                "2",
            ]
        )
        self.assertTrue(parsed.smoke)
        self.assertEqual(2, parsed.smoke_train_batches)

    def test_bare_cuda_cli_device_is_normalized_to_cuda_zero(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in this runtime")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available in this runtime")
        config = _training_config("wire")
        with patch.dict(
            os.environ,
            {"PCBPARSE_TRAIN_DEVICE": "cuda", "PCBPARSE_TRAIN_PRECISION": "fp16"},
            clear=False,
        ):
            from pcbparse.training.torch_backend import _configure_runtime

            device, precision = _configure_runtime(
                torch, config, config.tasks["wire"]
            )
        self.assertEqual("cuda:0", str(device))
        self.assertEqual("fp16", precision)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
