from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_component_validation_strata.py"
)
_SPEC = importlib.util.spec_from_file_location("component_stratified_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(
    *, sample_id: str, split: str, record_id: str, category: str = "R"
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "sample_id": sample_id,
        "split": split,
        "state": "active",
        "source_domain": "kicad",
        "category": category,
        "category_id": "3",
        "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
    }


def _fixture(root: Path, *, include_test_prediction: bool = False) -> dict[str, Path]:
    paths = {
        "predictions_path": root / "predictions.jsonl",
        "status_path": root / "status.json",
        "component_coco_path": root / "component_coco.json",
        "materialization_summary_path": root / "summary.json",
        "component_manifest_path": root / "component.jsonl",
        "source_manifest_path": root / "source.jsonl",
        "split_manifest_path": root / "split.jsonl",
    }
    binding = "a" * 64
    image_hashes = {"v": "b" * 64, "t": "c" * 64}
    _write_jsonl(
        paths["split_manifest_path"],
        [
            {"sample_id": "v", "split": "val"},
            {"sample_id": "t", "split": "test"},
        ],
    )
    _write_jsonl(
        paths["source_manifest_path"],
        [
            {
                "sample_id": sample_id,
                "source_class": "kicad",
                "files": {
                    "image": {
                        "path": f"{sample_id}/image.png",
                        "sha256": image_hashes[sample_id],
                    }
                },
            }
            for sample_id in ("v", "t")
        ],
    )
    _write_jsonl(
        paths["component_manifest_path"],
        [
            _record(sample_id="v", split="val", record_id="v:component:C_1"),
            _record(sample_id="t", split="test", record_id="t:component:C_1"),
        ],
    )
    lineage = {
        "inputs": {
            "component_manifest": {
                "path": "component.jsonl",
                "sha256": _sha(paths["component_manifest_path"]),
            },
            "source_manifest": {
                "path": "source.jsonl",
                "sha256": _sha(paths["source_manifest_path"]),
            },
            "split_manifest": {
                "path": "split.jsonl",
                "sha256": _sha(paths["split_manifest_path"]),
            },
        }
    }
    images = [
        {
            "id": index,
            "sample_id": sample_id,
            "split": split,
            "view_id": f"component:{sample_id}:full",
            "view_kind": "full",
            "binding_sha256": binding,
            "source_image": f"{sample_id}/image.png",
            "source_image_sha256": image_hashes[sample_id],
            "source_size": [100, 100],
        }
        for index, (sample_id, split) in enumerate(
            (("v", "val"), ("t", "test")), start=1
        )
    ]
    annotations = [
        {
            "id": index,
            "image_id": index,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "area": 400.0,
            "ignore": 0,
            "source_record_id": f"{sample_id}:component:C_1",
            "source_bbox": [10.0, 10.0, 30.0, 30.0],
        }
        for index, sample_id in enumerate(("v", "t"), start=1)
    ]
    coco = {
        "info": {"binding_sha256": binding, "lineage": lineage},
        "categories": [
            {"id": 1, "name": "R", "source_category_id": "3"}
        ],
        "images": images,
        "annotations": annotations,
    }
    _write_json(paths["component_coco_path"], coco)
    summary = {
        "binding_sha256": binding,
        "lineage": lineage,
        "outputs": {
            "component_coco.json": {"sha256": _sha(paths["component_coco_path"])}
        },
    }
    _write_json(paths["materialization_summary_path"], summary)
    prediction_rows = [
        {
            "sample_id": "v",
            "view_id": "component:v:full",
            "boxes": [[10.0, 10.0, 30.0, 30.0]],
            "scores": [0.9],
            "labels": [1],
        }
    ]
    if include_test_prediction:
        prediction_rows.append(
            {
                "sample_id": "t",
                "view_id": "component:t:full",
                "boxes": [[10.0, 10.0, 30.0, 30.0]],
                "scores": [0.8],
                "labels": [1],
            }
        )
    _write_jsonl(paths["predictions_path"], prediction_rows)
    status = {
        "task": "component",
        "status": "candidate_evidence_ready",
        "formal_evidence": True,
        "smoke": False,
        "finished_at_utc": "2026-08-24T00:00:00Z",
        "predictions": {"sha256": _sha(paths["predictions_path"])},
        "lineage": {
            "task": "component",
            "formal_evidence": True,
            "smoke": False,
            "binding_sha256": binding,
            "materialization_commit_id": binding,
            "materialization_commit_sha256": _sha(
                paths["materialization_summary_path"]
            ),
            "source_manifest_sha256": _sha(paths["source_manifest_path"]),
            "split_manifest_sha256": _sha(paths["split_manifest_path"]),
            "task_manifest_sha256": _sha(paths["component_manifest_path"]),
        },
        "metrics": {"bbox_map50": 1.0, "precision50": 1.0, "recall50": 1.0},
    }
    _write_json(paths["status_path"], status)
    return paths


def _tiled_fixture(
    root: Path, *, include_test_prediction: bool = False
) -> dict[str, Path]:
    paths = _fixture(root)
    parent_status_path = paths["status_path"]
    parent = json.loads(parent_status_path.read_text(encoding="utf-8"))
    model_path = root / "model.pt"
    model_path.write_bytes(b"component-tiled-fixture")
    parent["artifact"] = {"path": model_path.name, "sha256": _sha(model_path)}
    _write_json(parent_status_path, parent)

    tiled_predictions_path = root / "tiled_predictions.jsonl"
    tiled_rows = [
        {
            "sample_id": "v",
            "source_image_sha256": "b" * 64,
            "shape": [100, 100],
            "tile_count": 1,
            "candidate_count_before_cross_tile_nms": 1,
            "predictions": [
                {
                    "box": [10.0, 10.0, 30.0, 30.0],
                    "category": "R",
                    "score": 0.9,
                    "source_view_id": "original",
                }
            ],
        }
    ]
    if include_test_prediction:
        tiled_rows.append(
            {
                "sample_id": "t",
                "source_image_sha256": "c" * 64,
                "shape": [100, 100],
                "tile_count": 1,
                "candidate_count_before_cross_tile_nms": 1,
                "predictions": [
                    {
                        "box": [10.0, 10.0, 30.0, 30.0],
                        "category": "R",
                        "score": 0.9,
                        "source_view_id": "original",
                    }
                ],
            }
        )
    _write_jsonl(tiled_predictions_path, tiled_rows)
    tiled_status_path = root / "tiled_status.json"
    tiled_status = {
        "schema_version": 1,
        "kind": audit.TILED_DIAGNOSTIC_KIND,
        "task": "component",
        "status": audit.DIAGNOSTIC_STATUS,
        "formal_evidence": False,
        "registerable": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "smoke": False,
        "finished_at_utc": "2026-08-24T00:01:00Z",
        "device": "cpu",
        "protocol": {
            "tile_size": 512,
            "tile_overlap": 64,
            "tile_stride": 448,
            "score_threshold": 0.2,
            "component_nms_iou": 0.55,
            "match_iou": 0.5,
            "max_detections_per_tile": 300,
            "thresholds_searched": False,
            "tiling_configs_searched": False,
            "nms_configs_searched": False,
        },
        "selection_boundary": {
            "split": "val",
            "available_val_samples": 1,
            "selected_val_samples": 1,
            "selected_sample_ids": ["v"],
            "train_samples_loaded": 0,
            "train_images_decoded": 0,
            "train_annotations_scored": 0,
            "test_samples_loaded": 0,
            "test_images_decoded": 0,
            "test_annotations_scored": 0,
        },
        "metrics": {
            "bbox_map50": 1.0,
            "precision50": 1.0,
            "recall50": 1.0,
            "f1_50": 1.0,
        },
        "performance": {
            "sample_count": len(tiled_rows),
            "tile_count": len(tiled_rows),
            "candidate_count_before_cross_tile_nms": len(tiled_rows),
            "prediction_count_after_cross_tile_nms": len(tiled_rows),
        },
        "artifacts": {
            "predictions": {
                "path": tiled_predictions_path.name,
                "sha256": _sha(tiled_predictions_path),
                "row_count": len(tiled_rows),
            }
        },
        "lineage": {
            "parent_status": {
                "path": parent_status_path.name,
                "sha256": _sha(parent_status_path),
                "status": parent["status"],
            },
            "checkpoint": {
                "path": model_path.name,
                "sha256": _sha(model_path),
            },
            "component_coco": {
                "path": paths["component_coco_path"].name,
                "sha256": _sha(paths["component_coco_path"]),
            },
            "materialization_summary": {
                "path": paths["materialization_summary_path"].name,
                "sha256": _sha(paths["materialization_summary_path"]),
                "binding_sha256": "a" * 64,
            },
        },
        "interpretation_boundary": {
            "training_gate_reused": False,
            "parent_gate_status_changed": False,
            "models_lock_read": False,
            "models_lock_modified": False,
            "validation_configuration_tuned": False,
            "test_split_used": False,
            "full_validation_conclusion_allowed": True,
        },
    }
    _write_json(tiled_status_path, tiled_status)
    paths["predictions_path"] = tiled_predictions_path
    paths["status_path"] = tiled_status_path
    return paths


class ComponentStratifiedMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = [
            {
                "model_label": 1,
                "official_category_id": 3,
                "name": "R",
            },
            {
                "model_label": 2,
                "official_category_id": 4,
                "name": "C",
            },
        ]
        self.view_order = {"v1": 0, "v2": 1}

    def test_class_aware_matching_reports_tp_fp_fn_f1_and_ap50(self) -> None:
        truths = [
            {
                "view_id": "v1",
                "label": 1,
                "annotation_id": 1,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
            },
            {
                "view_id": "v2",
                "label": 1,
                "annotation_id": 2,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
            },
        ]
        predictions = [
            {
                "view_id": "v1",
                "label": 1,
                "prediction_index": 0,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
                "score": 0.9,
            },
            {
                "view_id": "v1",
                "label": 1,
                "prediction_index": 1,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
                "score": 0.8,
            },
            {
                "view_id": "v1",
                "label": 2,
                "prediction_index": 2,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
                "score": 0.7,
            },
        ]
        result = audit.evaluate_stratum(
            truths,
            predictions,
            categories=self.categories,
            view_order=self.view_order,
        )
        self.assertEqual(2, result["support"])
        self.assertEqual(1, result["true_positives"])
        self.assertEqual(2, result["false_positives"])
        self.assertEqual(1, result["false_negatives"])
        self.assertAlmostEqual(1.0 / 3.0, result["precision50"])
        self.assertAlmostEqual(0.5, result["recall50"])
        self.assertAlmostEqual(0.4, result["f1_50"])
        self.assertAlmostEqual(51.0 / 101.0, result["map50"])
        self.assertIsNone(result["per_class"]["2"]["ap50"])

    def test_size_range_ignores_matches_to_out_of_bin_truth(self) -> None:
        truths = [
            {
                "view_id": "v1",
                "label": 1,
                "annotation_id": 1,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
            },
            {
                "view_id": "v1",
                "label": 1,
                "annotation_id": 2,
                "box": (100.0, 100.0, 220.0, 220.0),
                "area": 14400.0,
            },
        ]
        predictions = [
            {
                "view_id": "v1",
                "label": 1,
                "prediction_index": 0,
                "box": (0.0, 0.0, 10.0, 10.0),
                "area": 100.0,
                "score": 0.9,
            },
            {
                "view_id": "v1",
                "label": 1,
                "prediction_index": 1,
                "box": (100.0, 100.0, 220.0, 220.0),
                "area": 14400.0,
                "score": 0.8,
            },
            {
                "view_id": "v1",
                "label": 1,
                "prediction_index": 2,
                "box": (20.0, 20.0, 25.0, 25.0),
                "area": 25.0,
                "score": 0.7,
            },
        ]
        small = audit.evaluate_stratum(
            truths,
            predictions,
            categories=self.categories,
            view_order=self.view_order,
            area_range=(0.0, 1024.0),
        )
        self.assertEqual(1, small["support"])
        self.assertEqual(1, small["true_positives"])
        self.assertEqual(1, small["false_positives"])
        self.assertEqual(1, small["ignored_predictions"])
        self.assertEqual(0, small["false_negatives"])

        large = audit.evaluate_stratum(
            truths,
            predictions,
            categories=self.categories,
            view_order=self.view_order,
            area_range=(9216.0, None),
        )
        self.assertEqual(1, large["support"])
        self.assertEqual(1, large["true_positives"])
        self.assertEqual(0, large["false_positives"])
        self.assertEqual(2, large["ignored_predictions"])


class ComponentStratifiedEvidenceTests(unittest.TestCase):
    def test_hash_bound_build_uses_only_exact_frozen_val_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _fixture(root)
            report = audit.build_audit(workspace=root, **paths)
        self.assertEqual(0, report["selection_boundary"]["test_rows_used"])
        self.assertEqual(1, report["selection_boundary"]["val_truth_rows_used"])
        self.assertEqual(1, report["overall"]["support"])
        self.assertEqual(1.0, report["overall"]["map50"])
        self.assertTrue(
            all(report["lineage_checks"]["run_metric_reproduction"].values())
        )
        self.assertEqual(64, len(report["audit_payload_sha256"]))

    def test_prediction_from_test_view_is_rejected_instead_of_scored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _fixture(root, include_test_prediction=True)
            with self.assertRaisesRegex(
                ValueError, "exact frozen-val full-view population"
            ):
                audit.build_audit(workspace=root, **paths)

    def test_diagnostic_replay_requires_and_hash_verifies_formal_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _fixture(root)
            parent_path = paths["status_path"]
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            diagnostic_path = root / "diagnostic_status.json"
            diagnostic = {
                **parent,
                "status": audit.DIAGNOSTIC_STATUS,
                "formal_evidence": False,
                "registerable": False,
                "diagnostic_parent": {
                    "run_status": {
                        "path": parent_path.name,
                        "sha256": _sha(parent_path),
                        "status": parent["status"],
                    }
                },
            }
            _write_json(diagnostic_path, diagnostic)
            paths["status_path"] = diagnostic_path

            report = audit.build_audit(workspace=root, **paths)

            self.assertEqual(
                "diagnostic_replay", report["evaluation_provenance"]["kind"]
            )
            self.assertFalse(report["evaluation_provenance"]["registerable"])

            parent_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagnostic parent run status SHA"):
                audit.build_audit(workspace=root, **paths)

    def test_manifest_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _fixture(root)
            with paths["component_manifest_path"].open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(ValueError, "component_manifest/summary SHA"):
                audit.build_audit(workspace=root, **paths)

    def test_tiled_diagnostic_is_hash_bound_and_reports_sample_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _tiled_fixture(root)
            report = audit.build_audit(
                workspace=root, score_threshold=0.2, **paths
            )

        self.assertEqual(
            "pcbparse_component_tiled_frozen_val_stratified_audit",
            report["kind"],
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["registerable"])
        self.assertEqual(
            "tiled_diagnostic_replay", report["evaluation_provenance"]["kind"]
        )
        self.assertEqual(0, report["selection_boundary"]["test_rows_used"])
        self.assertEqual(0, report["selection_boundary"]["test_images_decoded"])
        self.assertEqual(1.0, report["overall"]["map50"])
        self.assertEqual(1.0, report["by_sample"]["v"]["ground_truth_coverage"])
        self.assertEqual(1, report["by_source_domain"]["kicad"]["sample_count"])
        self.assertEqual([], report["zero_recall_supported_classes"])
        self.assertEqual(
            "original source-image bbox area in pixels squared",
            report["contract"]["size_basis"],
        )
        self.assertTrue(
            all(report["lineage_checks"]["run_metric_reproduction"].values())
        )

    def test_tiled_test_sample_is_rejected_even_when_status_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _tiled_fixture(root, include_test_prediction=True)
            with self.assertRaisesRegex(
                ValueError, "exact frozen-val stitched population"
            ):
                audit.build_audit(
                    workspace=root, score_threshold=0.2, **paths
                )

    def test_tiled_prediction_byte_drift_is_rejected_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _tiled_fixture(root)
            with paths["predictions_path"].open("a", encoding="utf-8") as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(ValueError, "tiled predictions SHA-256"):
                audit.build_audit(
                    workspace=root, score_threshold=0.2, **paths
                )


if __name__ == "__main__":
    unittest.main()
