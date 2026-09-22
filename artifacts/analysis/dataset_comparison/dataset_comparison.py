from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TASK_FILES = {
    "component_detection": "component_detection.jsonl",
    "pin_keypoints": "pin_keypoints.jsonl",
    "ocr_text": "ocr_text.jsonl",
    "wire_paths": "wire_paths.jsonl",
    "junction_patches": "junction_patches.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def _cohort_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "source_classes": dict(sorted(Counter(row["source_class"] for row in rows).items())),
    }
    for key in ("components", "pins", "texts", "edges"):
        values = [int(row["counts"][key]) for row in rows]
        result[key] = {
            "total": sum(values),
            "mean_per_sample": round(statistics.mean(values), 3),
            "median_per_sample": statistics.median(values),
            "p90_per_sample": _quantile(values, 0.9),
        }
    areas = [int(row["canvas"]["width"]) * int(row["canvas"]["height"]) for row in rows]
    result["image_area_pixels"] = {
        "mean": round(statistics.mean(areas), 1),
        "median": statistics.median(areas),
        "p90": _quantile(areas, 0.9),
    }
    return result


def _raw_label_profile(dataset_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    text_details: Counter[str] = Counter()
    component_states: Counter[str] = Counter()
    for row in rows:
        raw = json.loads((dataset_root / row["files"]["json"]["path"]).read_text(encoding="utf-8"))
        for component in raw["components"].values():
            categories[str(component.get("category") or "<blank>")] += 1
            component_states[str(component.get("state") or "<blank>")] += 1
        for text in raw["texts"].values():
            text_details[str(text.get("details") or "<blank>")] += 1
    return {
        "component_categories": dict(categories.most_common()),
        "text_details": dict(text_details.most_common()),
        "component_states": dict(component_states.most_common()),
    }


def _derived_task_counts(derived_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task, filename in TASK_FILES.items():
        counts = Counter()
        with (derived_root / filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cohort = "first_200" if int(row["sample_id"]) <= 200 else "new_100"
                counts[cohort] += 1
        first = counts["first_200"]
        new = counts["new_100"]
        results[task] = {
            "first_200": first,
            "new_100": new,
            "all_300": first + new,
            "increment_vs_first_200_pct": round(100.0 * new / first, 2) if first else None,
            "new_100_share_of_all_pct": round(100.0 * new / (first + new), 2) if first + new else None,
        }
    return results


def analyze(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_root = repo_root / "Dataset300" / "Dataset300"
    public_root = repo_root / "赛题六公开数据集"
    public_train = public_root / "200_train_cases"
    manifest_path = repo_root / "data" / "manifests" / "dataset300.jsonl"
    audit_path = repo_root / "artifacts" / "audits" / "dataset300_audit.json"
    derived_root = repo_root / "data" / "derived" / "dataset300-v2"

    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in manifest}
    first_200 = manifest[:200]
    new_100 = manifest[200:]
    first_200_datasheet = [row for row in first_200 if row["source_class"] == "datasheet"]

    verified_images = 0
    verified_json = 0
    for row in manifest:
        verified_images += _sha256(dataset_root / row["files"]["image"]["path"]) == row["files"]["image"]["sha256"]
        verified_json += _sha256(dataset_root / row["files"]["json"]["path"]) == row["files"]["json"]["sha256"]

    public_image_matches = 0
    component_count_equal_cases = 0
    public_counts = Counter()
    raw_first_counts = Counter()
    public_target_hashes: dict[str, str] = {}
    for sample_number in range(1, 201):
        sample_id = f"{sample_number:04d}"
        manifest_row = by_id[sample_id]
        public_image = next((public_train / sample_id).glob("*.png"))
        dataset_image = dataset_root / manifest_row["files"]["image"]["path"]
        public_image_matches += _sha256(public_image) == _sha256(dataset_image)

        target_path = next((public_train / sample_id).glob("*_target_named.json"))
        public_target_hashes[sample_id] = _sha256(target_path)
        target = json.loads(target_path.read_text(encoding="utf-8"))
        component_count = len(target["components"])
        pin_count = sum(len(component_pins) for component_pins in target["pins"].values())
        net_count = len(target["nets"])
        segment_count = sum(len(net.get("edges", {})) for net in target["nets"].values())
        public_counts.update(
            components=component_count,
            pins=pin_count,
            nets=net_count,
            output_segments=segment_count,
        )
        raw_first_counts.update(manifest_row["counts"])
        component_count_equal_cases += component_count == manifest_row["counts"]["components"]

    gt_rows: list[dict[str, Any]] = []
    for case_dir in sorted((public_root / "10_GTcase").iterdir()):
        if not case_dir.is_dir():
            continue
        sample_id = case_dir.name[:4]
        gt_image = next(case_dir.glob("*.png"))
        train_image = next((public_train / sample_id).glob("*.png"))
        gt_target = next(case_dir.glob("*_target.json"))
        gt_rows.append(
            {
                "sample_id": sample_id,
                "image_matches_public_train": _sha256(gt_image) == _sha256(train_image),
                "target_json_matches_public_train": _sha256(gt_target) == public_target_hashes[sample_id],
            }
        )

    first_groups = {row["duplicate_group"] for row in first_200}
    new_groups = {row["duplicate_group"] for row in new_100}
    cross_groups = sorted(first_groups & new_groups)
    cross_group_members = {
        group: [row["sample_id"] for row in manifest if row["duplicate_group"] == group]
        for group in cross_groups
    }

    old_image_hashes = {row["files"]["image"]["sha256"]: row["sample_id"] for row in first_200}
    old_json_hashes = {row["files"]["json"]["sha256"]: row["sample_id"] for row in first_200}
    exact_cross_image_pairs = [
        [old_image_hashes[row["files"]["image"]["sha256"]], row["sample_id"]]
        for row in new_100
        if row["files"]["image"]["sha256"] in old_image_hashes
    ]
    exact_cross_json_pairs = [
        [old_json_hashes[row["files"]["json"]["sha256"]], row["sample_id"]]
        for row in new_100
        if row["files"]["json"]["sha256"] in old_json_hashes
    ]

    label_profiles = {
        "first_200": _raw_label_profile(dataset_root, first_200),
        "new_100": _raw_label_profile(dataset_root, new_100),
    }
    first_categories = Counter(label_profiles["first_200"]["component_categories"])
    new_categories = Counter(label_profiles["new_100"]["component_categories"])
    category_gain_rows = []
    for category in sorted(set(first_categories) | set(new_categories)):
        old = first_categories[category]
        new = new_categories[category]
        category_gain_rows.append(
            {
                "category": category,
                "first_200": old,
                "new_100": new,
                "all_300": old + new,
                "increment_vs_first_200_pct": round(100.0 * new / old, 2) if old else None,
                "new_100_share_of_all_pct": round(100.0 * new / (old + new), 2) if old + new else None,
            }
        )

    training_configs = sorted((repo_root / "configs" / "train").glob("dataset300*.json"))
    public_training_references = []
    for path in list((repo_root / "configs").rglob("*")) + list((repo_root / "src").rglob("*.py")) + list((repo_root / "data_tools").rglob("*.py")):
        if not path.is_file():
            continue
        try:
            if "赛题六公开数据集" in path.read_text(encoding="utf-8"):
                public_training_references.append(path.relative_to(repo_root).as_posix())
        except UnicodeDecodeError:
            pass

    summary = {
        "as_of": "2026-09-12",
        "sources": {
            "public_dataset": "赛题六公开数据集",
            "dataset300": "Dataset300/Dataset300",
            "dataset300_manifest": "data/manifests/dataset300.jsonl",
            "dataset300_audit": "artifacts/audits/dataset300_audit.json",
            "derived_training_data": "data/derived/dataset300-v2",
        },
        "identity_and_overlap": {
            "public_train_samples": 200,
            "dataset300_samples": len(manifest),
            "corresponding_images_exact_sha256_matches": public_image_matches,
            "public_component_count_equal_to_dataset300_cases": component_count_equal_cases,
            "public_10_gt_samples": len(gt_rows),
            "public_10_gt_images_also_in_public_train": sum(row["image_matches_public_train"] for row in gt_rows),
            "public_10_gt_target_json_byte_matches_train_target_named": sum(row["target_json_matches_public_train"] for row in gt_rows),
            "dataset300_manifest_hash_checks": {
                "images_verified": verified_images,
                "json_annotations_verified": verified_json,
                "expected_each": len(manifest),
            },
        },
        "annotation_shape": {
            "public_train_files_per_sample": ["image", "competition_target_json"],
            "dataset300_files_per_sample": ["image", "CVJsonStd_json", "CVJsonStd_eds", "SPICE_netlist"],
            "public_first_200_totals": dict(public_counts),
            "dataset300_first_200_raw_totals": dict(raw_first_counts),
            "interpretation": "Same first 200 images, but different annotation representations and coordinate/output semantics; do not concatenate as independent samples.",
        },
        "cohorts": {
            "first_200": _cohort_profile(first_200),
            "first_200_datasheet_only": _cohort_profile(first_200_datasheet),
            "new_100": _cohort_profile(new_100),
        },
        "effective_increment": {
            "first_200_effective_groups": len(first_groups),
            "new_100_effective_groups_before_overlap_removal": len(new_groups),
            "groups_shared_between_cohorts": len(cross_groups),
            "new_independent_groups_added": len(new_groups - first_groups),
            "all_300_effective_groups": len(first_groups | new_groups),
            "cross_cohort_group_members": cross_group_members,
            "exact_cross_cohort_image_pairs": exact_cross_image_pairs,
            "exact_cross_cohort_json_pairs": exact_cross_json_pairs,
        },
        "derived_training_records": _derived_task_counts(derived_root),
        "label_profiles": label_profiles,
        "quality": {
            "audit_ok": audit["ok"],
            "strict_ok": audit["strict_ok"],
            "warnings": audit["summary"]["warnings"],
            "duplicate_groups": audit["summary"]["duplicate_groups"],
            "image_extension_mismatches": audit["summary"]["issues_by_code"].get("image_extension_mismatch", 0),
            "notes": [
                "Duplicate and near-duplicate groups must stay in one split.",
                "Image decoding must use detected content, not only filename extension.",
                "CVJsonStd, EDS, and SPICE are co-packaged/generated views, not statistically independent labels.",
            ],
        },
        "current_project_usage": {
            "dataset300_training_config_count": len(training_configs),
            "dataset300_training_configs": [path.relative_to(repo_root).as_posix() for path in training_configs],
            "public_dataset_references_in_training_code_or_configs": public_training_references,
        },
        "public_gt_overlap_rows": gt_rows,
    }
    return summary, category_gain_rows


def write_outputs(repo_root: Path, output_dir: Path | None = None) -> tuple[Path, Path, Path]:
    if output_dir is None:
        output_dir = repo_root / "artifacts" / "analysis" / "dataset_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, category_gain_rows = analyze(repo_root)
    summary_path = output_dir / "dataset_comparison_summary.json"
    category_path = output_dir / "component_category_gain.csv"
    task_path = output_dir / "training_record_increment.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with category_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_gain_rows[0]))
        writer.writeheader()
        writer.writerows(category_gain_rows)
    task_rows = [
        {
            "task": task,
            "first_200": values["first_200"],
            "new_100": values["new_100"],
            "all_300": values["all_300"],
            "increment_pct": values["increment_vs_first_200_pct"],
            "new_share_pct": values["new_100_share_of_all_pct"],
        }
        for task, values in summary["derived_training_records"].items()
    ]
    with task_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    return summary_path, category_path, task_path


if __name__ == "__main__":
    here = Path(__file__).resolve()
    repository = next(parent for parent in here.parents if (parent / "Dataset300" / "Dataset300").is_dir())
    summary_file, category_file, task_file = write_outputs(repository)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    print(json.dumps({
        "summary": summary_file.as_posix(),
        "category_table": category_file.as_posix(),
        "task_table": task_file.as_posix(),
        "identity_and_overlap": summary["identity_and_overlap"],
        "effective_increment": summary["effective_increment"],
        "derived_training_records": summary["derived_training_records"],
    }, ensure_ascii=False, indent=2))
