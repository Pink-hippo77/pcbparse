from __future__ import annotations

"""Build and strictly validate a human-review pack for real junction labels.

The junction truth audit contains *review proposals*, not labels.  It also
contains multiple pixel records for some physical intersections.  This module
turns those records into one review item per physical candidate cluster,
creates source-image context crops, and binds every immutable field to the
frozen audit and derived manifests.

Nothing in this module edits the materialized annotations or promotes a model,
heuristic, topology channel, or synthetic sample to ground truth.  A label is
accepted only when a human explicitly supplies a valid six-class label,
reviewer identity, and timezone-aware review timestamp.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from ._bootstrap import REPOSITORY_ROOT, default_dataset_root


PACK_SCHEMA_VERSION = 1
PACK_VERSION = "junction-human-review-pack-v1"
PHYSICAL_ID_VERSION = "junction-physical-cluster-v1"
IMPORT_VERSION = "junction-human-review-import-v1"
JUNCTION_CLASSES: tuple[str, ...] = (
    "endpoint",
    "t_junction",
    "bend",
    "connected_cross",
    "non_connected_cross",
    "noise",
)
REVIEW_STATUSES = frozenset({"pending", "reviewed", "needs_adjudication"})
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
_CONFIDENCE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
_REVIEWER_RE = re.compile(r"^[^\s].{0,126}[^\s]$|^[^\s]$")

CSV_FIELDS: tuple[str, ...] = (
    "schema_version",
    "package_id",
    "physical_candidate_id",
    "integrity_sha256",
    "batch_id",
    "batch_index",
    "global_index",
    "sample_id",
    "split",
    "source_domain",
    "member_record_count",
    "member_record_ids",
    "edge_keys",
    "center_x",
    "center_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "source_image",
    "source_image_sha256",
    "context_crop_path",
    "overlay_crop_path",
    "representative_patch_path",
    "context_x1",
    "context_y1",
    "context_x2",
    "context_y2",
    "current_proposal",
    "current_proposal_confidence",
    "proposal_consensus",
    "proposal_distribution_json",
    "proposal_reference_only",
    "proposal_accepted_as_ground_truth",
    "leakage_guard",
    "review_status",
    "human_label",
    "reviewer",
    "reviewed_at",
    "review_notes",
)
HUMAN_FIELDS = frozenset(
    {"review_status", "human_label", "reviewer", "reviewed_at", "review_notes"}
)
IMMUTABLE_CSV_FIELDS = tuple(field for field in CSV_FIELDS if field not in HUMAN_FIELDS)


class JunctionReviewPackError(ValueError):
    """The review pack or a human-review import failed closed validation."""


@dataclass(frozen=True, slots=True)
class ReviewPackResult:
    package_id: str
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReviewValidationResult:
    package_id: str
    batch_ids: tuple[str, ...]
    total: int
    reviewed: int
    pending: int
    needs_adjudication: int
    records: tuple[dict[str, Any], ...]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JunctionReviewPackError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_bytes(row) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JunctionReviewPackError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JunctionReviewPackError(f"{path}: root must be an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise JunctionReviewPackError(
                        f"{path}:{line_number}: row must be an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise JunctionReviewPackError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _read_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in output:
            raise JunctionReviewPackError(
                f"{path}: missing or duplicate sample_id {sample_id!r}"
            )
        output[sample_id] = row
    return output


def _read_selected_derived_rows(
    path: Path, required_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], str]:
    digest = hashlib.sha256()
    selected: dict[str, dict[str, Any]] = {}
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise JunctionReviewPackError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise JunctionReviewPackError(
                        f"{path}:{line_number}: row must be an object"
                    )
                record_id = str(row.get("record_id", ""))
                if record_id in required_ids:
                    if record_id in selected:
                        raise JunctionReviewPackError(
                            f"{path}: duplicate required record_id {record_id!r}"
                        )
                    selected[record_id] = row
    except OSError as exc:
        raise JunctionReviewPackError(f"cannot read derived manifest {path}: {exc}") from exc
    missing = sorted(required_ids - selected.keys())
    if missing:
        raise JunctionReviewPackError(
            f"derived manifest is missing {len(missing)} audited records; first={missing[0]!r}"
        )
    return selected, digest.hexdigest()


def _safe_child(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise JunctionReviewPackError(f"path escapes its root: {relative!r}") from exc
    if must_exist and not candidate.is_file():
        raise JunctionReviewPackError(f"required file does not exist: {candidate}")
    return candidate


def physical_candidate_id(cluster: Mapping[str, Any]) -> str:
    """Return a stable ID derived only from the physical cluster identity."""

    bbox = cluster.get("bbox")
    if not isinstance(bbox, Mapping):
        raise JunctionReviewPackError("cluster bbox must be an object")
    identity = {
        "version": PHYSICAL_ID_VERSION,
        "sample_id": str(cluster.get("sample_id", "")),
        "edge_keys": sorted(str(value) for value in cluster.get("edge_keys", [])),
        "bbox": {key: int(bbox[key]) for key in ("x1", "y1", "x2", "y2")},
        "record_ids": sorted(str(value) for value in cluster.get("record_ids", [])),
    }
    if not identity["sample_id"] or not identity["edge_keys"] or not identity["record_ids"]:
        raise JunctionReviewPackError("cluster identity is incomplete")
    return "JPC1-" + _sha256_bytes(_canonical_bytes(identity))[:24]


def _validate_audit(
    audit: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if audit.get("kind") != "junction_truth_audit":
        raise JunctionReviewPackError("input is not a junction truth audit")
    policy = audit.get("policy")
    if not isinstance(policy, Mapping) or policy.get("no_ground_truth_promotion") is not True:
        raise JunctionReviewPackError("audit does not assert the no-ground-truth-promotion policy")
    if policy.get("human_review_required") is not True:
        raise JunctionReviewPackError("audit does not require human review")
    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        raise JunctionReviewPackError("audit summary is missing")
    if int(summary.get("accepted_as_ground_truth_count", -1)) != 0:
        raise JunctionReviewPackError("audit already contains ground-truth promotions")
    if int(summary.get("may_enter_training_without_human_review_count", -1)) != 0:
        raise JunctionReviewPackError("audit permits unreviewed proposals to enter training")

    raw_candidates = audit.get("candidates")
    raw_clusters = audit.get("clusters")
    if not isinstance(raw_candidates, list) or not isinstance(raw_clusters, list):
        raise JunctionReviewPackError("audit candidates/clusters must be arrays")
    candidates: dict[str, dict[str, Any]] = {}
    for value in raw_candidates:
        if not isinstance(value, dict):
            raise JunctionReviewPackError("audit candidate must be an object")
        record_id = str(value.get("record_id", ""))
        if not record_id or record_id in candidates:
            raise JunctionReviewPackError(f"missing or duplicate candidate record_id {record_id!r}")
        proposal = value.get("review_proposal")
        if not isinstance(proposal, Mapping):
            raise JunctionReviewPackError(f"{record_id}: review_proposal is missing")
        if proposal.get("accepted_as_ground_truth") is not False:
            raise JunctionReviewPackError(f"{record_id}: proposal was promoted to ground truth")
        if proposal.get("may_enter_training_without_human_review") is not False:
            raise JunctionReviewPackError(f"{record_id}: proposal bypasses human review")
        candidates[record_id] = value

    clusters: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    seen_physical_ids: set[str] = set()
    for raw in raw_clusters:
        if not isinstance(raw, dict):
            raise JunctionReviewPackError("audit cluster must be an object")
        cluster = dict(raw)
        cluster_id = physical_candidate_id(cluster)
        if cluster_id in seen_physical_ids:
            raise JunctionReviewPackError(f"duplicate physical cluster ID {cluster_id}")
        seen_physical_ids.add(cluster_id)
        record_ids = [str(value) for value in cluster.get("record_ids", [])]
        if len(record_ids) != len(set(record_ids)):
            raise JunctionReviewPackError(f"{cluster_id}: duplicate member record IDs")
        bbox = cluster.get("bbox")
        if not isinstance(bbox, Mapping):
            raise JunctionReviewPackError(f"{cluster_id}: missing bbox")
        x1, y1, x2, y2 = (int(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
        if x2 < x1 or y2 < y1:
            raise JunctionReviewPackError(f"{cluster_id}: invalid inclusive bbox")
        for record_id in record_ids:
            if record_id in seen_records:
                raise JunctionReviewPackError(f"candidate appears in multiple clusters: {record_id}")
            seen_records.add(record_id)
            candidate = candidates.get(record_id)
            if candidate is None:
                raise JunctionReviewPackError(f"cluster references unknown candidate: {record_id}")
            if str(candidate.get("sample_id")) != str(cluster.get("sample_id")):
                raise JunctionReviewPackError(f"{record_id}: sample mismatch between cluster and candidate")
            if str(candidate.get("split")) != str(cluster.get("split")):
                raise JunctionReviewPackError(f"{record_id}: split mismatch between cluster and candidate")
            center = candidate.get("center")
            if not isinstance(center, list) or len(center) != 2:
                raise JunctionReviewPackError(f"{record_id}: invalid center")
            cx, cy = int(center[0]), int(center[1])
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                raise JunctionReviewPackError(f"{record_id}: center is outside cluster bbox")
            if sorted(str(value) for value in candidate.get("edge_keys", [])) != sorted(
                str(value) for value in cluster.get("edge_keys", [])
            ):
                raise JunctionReviewPackError(f"{record_id}: edge-key mismatch")
        declared_count = int(cluster.get("candidate_pixel_count", -1))
        if declared_count != len(record_ids):
            raise JunctionReviewPackError(f"{cluster_id}: candidate_pixel_count mismatch")
        cluster["physical_candidate_id"] = cluster_id
        clusters.append(cluster)

    missing = sorted(candidates.keys() - seen_records)
    if missing:
        raise JunctionReviewPackError(
            f"audit clusters do not partition all candidates; first missing={missing[0]!r}"
        )
    return candidates, clusters


def _validate_derived_join(
    candidates: Mapping[str, Mapping[str, Any]],
    derived: Mapping[str, Mapping[str, Any]],
) -> None:
    for record_id, candidate in candidates.items():
        row = derived[record_id]
        for field in ("sample_id", "split", "source_domain"):
            if str(row.get(field)) != str(candidate.get(field)):
                raise JunctionReviewPackError(
                    f"{record_id}: derived/audit {field} mismatch"
                )
        center = row.get("center")
        if center != candidate.get("center"):
            raise JunctionReviewPackError(f"{record_id}: derived/audit center mismatch")
        if sorted(str(value) for value in row.get("edge_keys", [])) != sorted(
            str(value) for value in candidate.get("edge_keys", [])
        ):
            raise JunctionReviewPackError(f"{record_id}: derived/audit edge-key mismatch")


def _proposal_summary(
    members: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, dict[str, int]]:
    labels: list[str] = []
    confidences: list[str] = []
    for member in members:
        proposal = member["review_proposal"]
        raw_label = proposal.get("proposed_label")
        label = str(raw_label) if raw_label in JUNCTION_CLASSES else "UNKNOWN"
        labels.append(label)
        raw_confidence = str(proposal.get("confidence", "none")).lower()
        confidences.append(raw_confidence if raw_confidence in _CONFIDENCE_ORDER else "none")
    distribution = dict(sorted(Counter(labels).items()))
    unique = set(labels)
    if len(unique) == 1 and "UNKNOWN" not in unique:
        label = labels[0]
        confidence = min(confidences, key=lambda value: _CONFIDENCE_ORDER[value])
        return label, confidence, "unanimous", distribution
    if unique == {"UNKNOWN"}:
        return "", "none", "all_unknown", distribution
    return "", "none", "mixed", distribution


def _review_sort_key(cluster: Mapping[str, Any]) -> tuple[Any, ...]:
    bbox = cluster["bbox"]
    return (
        _SPLIT_ORDER.get(str(cluster["split"]), 99),
        str(cluster["source_domain"]),
        str(cluster["sample_id"]),
        int(bbox["y1"]),
        int(bbox["x1"]),
        str(cluster["physical_candidate_id"]),
    )


def _context_bbox(
    *, image_width: int, image_height: int, bbox: Mapping[str, Any], context_size: int
) -> dict[str, int]:
    cluster_width = int(bbox["x2"]) - int(bbox["x1"]) + 1
    cluster_height = int(bbox["y2"]) - int(bbox["y1"]) + 1
    width = min(image_width, max(context_size, cluster_width + 64))
    height = min(image_height, max(context_size, cluster_height + 64))
    center_x = (int(bbox["x1"]) + int(bbox["x2"])) / 2.0
    center_y = (int(bbox["y1"]) + int(bbox["y2"])) / 2.0
    x1 = max(0, min(image_width - width, int(math.floor(center_x - width / 2.0))))
    y1 = max(0, min(image_height - height, int(math.floor(center_y - height / 2.0))))
    return {"x1": x1, "y1": y1, "x2": x1 + width, "y2": y1 + height}


def _representative_member(
    cluster: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    bbox = cluster["bbox"]
    center_x = (int(bbox["x1"]) + int(bbox["x2"])) / 2.0
    center_y = (int(bbox["y1"]) + int(bbox["y2"])) / 2.0
    return min(
        (candidates[str(record_id)] for record_id in cluster["record_ids"]),
        key=lambda value: (
            (float(value["center"][0]) - center_x) ** 2
            + (float(value["center"][1]) - center_y) ** 2,
            str(value["record_id"]),
        ),
    )


def _draw_overlay(image: Image.Image, source_bbox: Mapping[str, int], cluster_bbox: Mapping[str, Any]) -> Image.Image:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    left = int(cluster_bbox["x1"]) - source_bbox["x1"]
    top = int(cluster_bbox["y1"]) - source_bbox["y1"]
    right = int(cluster_bbox["x2"]) - source_bbox["x1"]
    bottom = int(cluster_bbox["y2"]) - source_bbox["y1"]
    if left == right and top == bottom:
        radius = 7
        draw.ellipse((left - radius, top - radius, right + radius, bottom + radius), outline=(220, 20, 60), width=2)
        draw.line((left - 15, top, left - 9, top), fill=(220, 20, 60), width=2)
        draw.line((left + 9, top, left + 15, top), fill=(220, 20, 60), width=2)
        draw.line((left, top - 15, left, top - 9), fill=(220, 20, 60), width=2)
        draw.line((left, top + 9, left, top + 15), fill=(220, 20, 60), width=2)
    else:
        draw.rectangle((left - 4, top - 4, right + 4, bottom + 4), outline=(220, 20, 60), width=2)
    return overlay


def _row_integrity(row: Mapping[str, Any]) -> str:
    immutable = {key: value for key, value in row.items() if key not in HUMAN_FIELDS and key != "integrity_sha256"}
    return _sha256_bytes(_canonical_bytes(immutable))


def _csv_row(row: Mapping[str, Any]) -> dict[str, str]:
    bbox = row["physical_bbox"]
    context = row["context_source_bbox"]
    center = row["review_center"]
    values: dict[str, Any] = {
        "schema_version": row["schema_version"],
        "package_id": row["package_id"],
        "physical_candidate_id": row["physical_candidate_id"],
        "integrity_sha256": row["integrity_sha256"],
        "batch_id": row["batch_id"],
        "batch_index": row["batch_index"],
        "global_index": row["global_index"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "source_domain": row["source_domain"],
        "member_record_count": row["member_record_count"],
        "member_record_ids": "|".join(row["member_record_ids"]),
        "edge_keys": "|".join(row["edge_keys"]),
        "center_x": center[0],
        "center_y": center[1],
        "bbox_x1": bbox["x1"],
        "bbox_y1": bbox["y1"],
        "bbox_x2": bbox["x2"],
        "bbox_y2": bbox["y2"],
        "source_image": row["source_image"],
        "source_image_sha256": row["source_image_sha256"],
        "context_crop_path": row["context_crop_path"],
        "overlay_crop_path": row["overlay_crop_path"],
        "representative_patch_path": row["representative_patch_path"],
        "context_x1": context["x1"],
        "context_y1": context["y1"],
        "context_x2": context["x2"],
        "context_y2": context["y2"],
        "current_proposal": row["current_proposal"],
        "current_proposal_confidence": row["current_proposal_confidence"],
        "proposal_consensus": row["proposal_consensus"],
        "proposal_distribution_json": json.dumps(row["proposal_distribution"], sort_keys=True, separators=(",", ":")),
        "proposal_reference_only": "true",
        "proposal_accepted_as_ground_truth": "false",
        "leakage_guard": row["leakage_guard"],
        "review_status": row["review_status"],
        "human_label": row["human_label"],
        "reviewer": row["reviewer"],
        "reviewed_at": row["reviewed_at"],
        "review_notes": row["review_notes"],
    }
    return {field: str(values[field]) for field in CSV_FIELDS}


def _batch_html(batch_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        proposal = row["current_proposal"] or "无一致建议"
        cards.append(
            "<article>"
            f"<h3>{html.escape(row['physical_candidate_id'])}</h3>"
            f"<img src=\"../{html.escape(row['context_crop_path'])}\" alt=\"raw context\">"
            f"<img src=\"../{html.escape(row['overlay_crop_path'])}\" alt=\"marked context\">"
            f"<p>样本 {html.escape(row['sample_id'])} · {html.escape(row['split'])} · "
            f"中心 {row['review_center'][0]}, {row['review_center'][1]}</p>"
            f"<p>审计建议（非真值）：{html.escape(proposal)} · "
            f"{html.escape(row['current_proposal_confidence'])}</p>"
            "</article>"
        )
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Junction review %s</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;background:#f5f7fb;color:#172033}
.warning{padding:12px;background:#fff3cd;border:1px solid #ffe69c;border-radius:8px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px}
article{background:white;border:1px solid #dfe5ef;border-radius:9px;padding:10px}
h3{font-size:14px;margin:0 0 8px}img{width:190px;height:190px;object-fit:contain;background:white;border:1px solid #ddd;margin-right:6px}
p{font-size:13px;margin:6px 0}</style></head><body>
<h1>%s</h1><p class="warning">审计建议仅用于排队，绝不是标签。请在对应 CSV 中独立填写六类人工标签、审核人和带时区时间戳。</p>
<main>%s</main></body></html>
""" % (html.escape(batch_id), html.escape(batch_id), "".join(cards))


def _root_html(batch_summaries: Sequence[Mapping[str, Any]]) -> str:
    links = "".join(
        f"<li><a href=\"{html.escape(str(item['visual_path']))}\">{html.escape(str(item['batch_id']))}</a>"
        f" — {item['records']} 个物理候选，{html.escape(str(item['split']))}</li>"
        for item in batch_summaries
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Junction 真实标签人审包</title></head><body style="font-family:system-ui,sans-serif;max-width:920px;margin:30px auto">
<h1>Junction 真实六类标签人审包</h1>
<p><strong>安全边界：</strong>模型、拓扑和视觉规则只提供建议；任何建议都不是 Ground Truth。只有通过严格导入校验的人工标签才会形成独立审核产物。</p>
<p>六类：{', '.join(JUNCTION_CLASSES)}</p><ul>{links}</ul></body></html>"""


def _instructions() -> str:
    return f"""# Junction 真实六类人工审核说明

本包将候选像素合并成物理候选簇。每个 `physical_candidate_id` 只能标一次。

## 操作

1. 优先按 `train-0001`、`train-0002` 的顺序打开 `visual/` 页面查看原图与红框定位图。
2. 在对应 `batches/<batch>.review.csv` 中，只修改以下五列：
   `review_status`、`human_label`、`reviewer`、`reviewed_at`、`review_notes`。
3. 确认时填写 `review_status=reviewed`；`human_label` 必须是六类之一：
   `{', '.join(JUNCTION_CLASSES)}`。
4. `reviewer` 填可追溯的人工审核人；`reviewed_at` 填带时区 RFC3339 时间，
   例如 `2026-08-24T10:30:00+08:00`。
5. 无法确认时填写 `review_status=needs_adjudication`，标签留空，并在备注中说明歧义。
6. 不要根据 `current_proposal` 直接抄标签。它是审计建议，`proposal_reference_only=true`。
7. 使用 `python -m data_tools.junction_review_pack validate --package-dir <run_dir> --reviews <csv>`
   检查进度；批次全部确认后加 `--require-complete`。
8. 使用 `python -m data_tools.junction_review_pack import --package-dir <run_dir> --reviews <csv> --output-dir <new_dir>`
   生成独立的人审导入产物。该命令不会改写现有真值。

`val` 和 `test` 标签始终标记为 `never_use_for_training`。合成样本没有进入本包，
也不会被导入器视为真实标签。
"""


def build_review_pack(
    *,
    audit_path: str | Path,
    derived_manifest: str | Path,
    source_manifest: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    batch_size: int = 100,
    context_size: int = 192,
) -> ReviewPackResult:
    """Build a content-addressed, self-contained human-review package."""

    if type(batch_size) is not int or batch_size <= 0:
        raise JunctionReviewPackError("batch_size must be a positive integer")
    if type(context_size) is not int or context_size < 96:
        raise JunctionReviewPackError("context_size must be an integer >= 96")
    audit_file = Path(audit_path).expanduser().resolve()
    derived_file = Path(derived_manifest).expanduser().resolve()
    source_file = Path(source_manifest).expanduser().resolve()
    dataset_dir = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_root).expanduser().resolve(strict=False)
    for path in (audit_file, derived_file, source_file):
        if not path.is_file():
            raise JunctionReviewPackError(f"required input not found: {path}")
    if not dataset_dir.is_dir():
        raise JunctionReviewPackError(f"dataset root not found: {dataset_dir}")

    audit_sha256 = _sha256_file(audit_file)
    source_manifest_sha256 = _sha256_file(source_file)
    audit = _read_json(audit_file)
    candidates, clusters = _validate_audit(audit)
    derived, derived_sha256 = _read_selected_derived_rows(derived_file, set(candidates))
    _validate_derived_join(candidates, derived)
    source_rows = _read_source_manifest(source_file)

    lineage = audit.get("lineage", {})
    if lineage.get("source_manifest_sha256") != source_manifest_sha256:
        raise JunctionReviewPackError("source manifest hash does not match the audit lineage")
    materialized_run_dir = Path(str(audit.get("scope", {}).get("materialized_run_dir", ""))).resolve()
    if not materialized_run_dir.is_dir():
        raise JunctionReviewPackError(
            f"materialized run referenced by audit does not exist: {materialized_run_dir}"
        )

    used_samples = sorted({str(cluster["sample_id"]) for cluster in clusters})
    sample_assets: dict[str, dict[str, Any]] = {}
    audit_sample_hashes = lineage.get("sample_source_sha256", {})
    for sample_id in used_samples:
        manifest = source_rows.get(sample_id)
        if manifest is None:
            raise JunctionReviewPackError(f"source manifest is missing sample {sample_id}")
        image_meta = manifest.get("files", {}).get("image", {})
        relative = str(image_meta.get("path", ""))
        expected_hash = str(image_meta.get("sha256", ""))
        image_path = _safe_child(dataset_dir, relative)
        actual_hash = _sha256_file(image_path)
        if actual_hash != expected_hash:
            raise JunctionReviewPackError(f"{sample_id}: source image hash mismatch")
        audit_hash = str(audit_sample_hashes.get(sample_id, {}).get("image", ""))
        if audit_hash != actual_hash:
            raise JunctionReviewPackError(f"{sample_id}: source image differs from audit lineage")
        sample_assets[sample_id] = {
            "relative": relative,
            "path": image_path,
            "sha256": actual_hash,
        }

    package_basis = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pack_version": PACK_VERSION,
        "physical_id_version": PHYSICAL_ID_VERSION,
        "audit_sha256": audit_sha256,
        "derived_manifest_sha256": derived_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "materialized_run_id": audit.get("scope", {}).get("materialized_run_id"),
        "cluster_ids": sorted(str(cluster["physical_candidate_id"]) for cluster in clusters),
        "batch_size": batch_size,
        "context_size": context_size,
        "classes": list(JUNCTION_CLASSES),
    }
    package_id = _sha256_bytes(_canonical_bytes(package_basis))
    runs_root = output_dir / "runs"
    final_dir = runs_root / package_id
    manifest_path = final_dir / "package_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("package_id") != package_id:
            raise JunctionReviewPackError("existing package manifest has the wrong package ID")
        return ReviewPackResult(package_id, final_dir, manifest_path, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".junction-review-", dir=str(output_dir)))
    try:
        rows: list[dict[str, Any]] = []
        csv_rows: list[dict[str, str]] = []
        clusters_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cluster in sorted(clusters, key=_review_sort_key):
            split = str(cluster["split"])
            if split not in _SPLIT_ORDER:
                raise JunctionReviewPackError(f"unsupported split {split!r}")
            clusters_by_split[split].append(cluster)

        batch_assignments: dict[str, tuple[str, int]] = {}
        batch_members: dict[str, list[str]] = defaultdict(list)
        for split in ("train", "val", "test"):
            for split_index, cluster in enumerate(clusters_by_split.get(split, [])):
                batch_id = f"{split}-{split_index // batch_size + 1:04d}"
                batch_index = split_index % batch_size + 1
                candidate_id = str(cluster["physical_candidate_id"])
                batch_assignments[candidate_id] = (batch_id, batch_index)
                batch_members[batch_id].append(candidate_id)

        open_images: dict[str, Image.Image] = {}
        try:
            for global_index, cluster in enumerate(sorted(clusters, key=_review_sort_key), 1):
                candidate_id = str(cluster["physical_candidate_id"])
                sample_id = str(cluster["sample_id"])
                split = str(cluster["split"])
                batch_id, batch_index = batch_assignments[candidate_id]
                members = [candidates[str(record_id)] for record_id in cluster["record_ids"]]
                representative = _representative_member(cluster, candidates)
                proposal, confidence, consensus, distribution = _proposal_summary(members)
                source_asset = sample_assets[sample_id]
                source_image = open_images.get(sample_id)
                if source_image is None:
                    with Image.open(source_asset["path"]) as loaded:
                        source_image = loaded.convert("RGB")
                    open_images[sample_id] = source_image
                context_bbox = _context_bbox(
                    image_width=source_image.width,
                    image_height=source_image.height,
                    bbox=cluster["bbox"],
                    context_size=context_size,
                )
                context = source_image.crop(
                    (
                        context_bbox["x1"],
                        context_bbox["y1"],
                        context_bbox["x2"],
                        context_bbox["y2"],
                    )
                )
                context_rel = f"crops/context/{candidate_id}.png"
                overlay_rel = f"crops/overlay/{candidate_id}.png"
                detail_rel = f"crops/detail/{candidate_id}.png"
                context_path = staging / context_rel
                overlay_path = staging / overlay_rel
                detail_path = staging / detail_rel
                context_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                context.save(context_path, format="PNG", optimize=False)
                _draw_overlay(context.copy(), context_bbox, cluster["bbox"]).save(
                    overlay_path, format="PNG", optimize=False
                )
                source_patch = _safe_child(materialized_run_dir, str(representative["patch_path"]))
                try:
                    with Image.open(source_patch) as patch_image:
                        patch_image.verify()
                except Exception as exc:
                    raise JunctionReviewPackError(
                        f"invalid representative patch {source_patch}: {exc}"
                    ) from exc
                shutil.copyfile(source_patch, detail_path)

                bbox = {key: int(cluster["bbox"][key]) for key in ("x1", "y1", "x2", "y2")}
                row: dict[str, Any] = {
                    "schema_version": PACK_SCHEMA_VERSION,
                    "package_id": package_id,
                    "physical_candidate_id": candidate_id,
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "global_index": global_index,
                    "sample_id": sample_id,
                    "split": split,
                    "source_domain": str(cluster["source_domain"]),
                    "member_record_count": len(cluster["record_ids"]),
                    "member_record_ids": sorted(str(value) for value in cluster["record_ids"]),
                    "edge_keys": sorted(str(value) for value in cluster["edge_keys"]),
                    "physical_bbox": bbox,
                    "coordinate_convention": "source_pixel_xy_inclusive_physical_bbox",
                    "review_center": [
                        round((bbox["x1"] + bbox["x2"]) / 2.0, 3),
                        round((bbox["y1"] + bbox["y2"]) / 2.0, 3),
                    ],
                    "source_image": str(source_asset["relative"]),
                    "source_image_sha256": str(source_asset["sha256"]),
                    "context_crop_path": context_rel,
                    "overlay_crop_path": overlay_rel,
                    "representative_patch_path": detail_rel,
                    "representative_record_id": str(representative["record_id"]),
                    "context_source_bbox": context_bbox,
                    "context_coordinate_convention": "source_pixel_xy_half_open_bbox",
                    "current_proposal": proposal,
                    "current_proposal_confidence": confidence,
                    "proposal_consensus": consensus,
                    "proposal_distribution": distribution,
                    "proposal_reference_only": True,
                    "proposal_accepted_as_ground_truth": False,
                    "synthetic_source": False,
                    "leakage_guard": (
                        "human_review_required_before_separate_train_import"
                        if split == "train"
                        else "never_use_for_training"
                    ),
                    "review_status": "pending",
                    "human_label": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "review_notes": "",
                }
                row["integrity_sha256"] = _row_integrity(row)
                rows.append(row)
                csv_rows.append(_csv_row(row))
        finally:
            for image in open_images.values():
                image.close()

        _write_jsonl(staging / "review_queue.jsonl", rows)
        _write_csv(staging / "review_queue.csv", csv_rows)
        row_by_id = {str(row["physical_candidate_id"]): row for row in rows}
        csv_by_id = {str(row["physical_candidate_id"]): row for row in csv_rows}
        batch_summaries: list[dict[str, Any]] = []
        for batch_id in sorted(batch_members, key=lambda value: (_SPLIT_ORDER[value.split("-", 1)[0]], value)):
            ids = batch_members[batch_id]
            batch_rows = [row_by_id[value] for value in ids]
            batch_csv_rows = [csv_by_id[value] for value in ids]
            jsonl_rel = f"batches/{batch_id}.review.jsonl"
            csv_rel = f"batches/{batch_id}.review.csv"
            visual_rel = f"visual/{batch_id}.html"
            _write_jsonl(staging / jsonl_rel, batch_rows)
            _write_csv(staging / csv_rel, batch_csv_rows)
            (staging / visual_rel).parent.mkdir(parents=True, exist_ok=True)
            (staging / visual_rel).write_text(
                _batch_html(batch_id, batch_rows), encoding="utf-8", newline="\n"
            )
            batch_summaries.append(
                {
                    "batch_id": batch_id,
                    "split": batch_id.split("-", 1)[0],
                    "records": len(batch_rows),
                    "jsonl_path": jsonl_rel,
                    "jsonl_sha256": _sha256_file(staging / jsonl_rel),
                    "csv_path": csv_rel,
                    "csv_sha256": _sha256_file(staging / csv_rel),
                    "visual_path": visual_rel,
                }
            )
        (staging / "index.html").write_text(
            _root_html(batch_summaries), encoding="utf-8", newline="\n"
        )
        (staging / "INSTRUCTIONS.md").write_text(
            _instructions(), encoding="utf-8", newline="\n"
        )

        asset_rows: list[dict[str, Any]] = []
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            relative = path.relative_to(staging).as_posix()
            if relative in {"asset_manifest.jsonl", "package_manifest.json"}:
                continue
            # Batch CSV/JSONL files are deliberate editable working copies.
            # Their immutable fields are checked against the hash-bound
            # authoritative queue during validation, so editing human fields
            # must not invalidate the package asset manifest.
            if relative.startswith("batches/") and relative.endswith(
                (".review.csv", ".review.jsonl")
            ):
                continue
            asset_rows.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )
        _write_jsonl(staging / "asset_manifest.jsonl", asset_rows)
        counts_by_split = dict(sorted(Counter(str(row["split"]) for row in rows).items()))
        proposal_counts = dict(
            sorted(Counter(str(row["current_proposal"] or "UNKNOWN") for row in rows).items())
        )
        manifest = {
            "schema_version": PACK_SCHEMA_VERSION,
            "pack_version": PACK_VERSION,
            "kind": "junction_human_review_pack",
            "package_id": package_id,
            "package_basis_sha256": _sha256_bytes(_canonical_bytes(package_basis)),
            "physical_id_version": PHYSICAL_ID_VERSION,
            "classes": list(JUNCTION_CLASSES),
            "review_statuses": sorted(REVIEW_STATUSES),
            "policy": {
                "one_label_per_physical_cluster": True,
                "proposals_are_ground_truth": False,
                "synthetic_is_real_truth": False,
                "human_identity_and_timezone_timestamp_required": True,
                "existing_truth_is_modified": False,
                "val_test_may_enter_training": False,
            },
            "inputs": {
                "audit": {"path": str(audit_file), "sha256": audit_sha256},
                "derived_manifest": {"path": str(derived_file), "sha256": derived_sha256},
                "source_manifest": {"path": str(source_file), "sha256": source_manifest_sha256},
                "dataset_root": str(dataset_dir),
                "materialized_run_dir": str(materialized_run_dir),
                "materialized_run_id": audit.get("scope", {}).get("materialized_run_id"),
            },
            "settings": {"batch_size": batch_size, "context_size": context_size},
            "counts": {
                "candidate_pixel_records": len(candidates),
                "physical_candidate_clusters": len(rows),
                "samples": len(used_samples),
                "batches": len(batch_summaries),
                "by_split": counts_by_split,
                "current_proposals_reference_only": proposal_counts,
                "accepted_ground_truth": 0,
                "synthetic_records": 0,
            },
            "authoritative_queue": {
                "jsonl_path": "review_queue.jsonl",
                "jsonl_sha256": _sha256_file(staging / "review_queue.jsonl"),
                "csv_path": "review_queue.csv",
                "csv_sha256": _sha256_file(staging / "review_queue.csv"),
            },
            "asset_manifest": {
                "path": "asset_manifest.jsonl",
                "sha256": _sha256_file(staging / "asset_manifest.jsonl"),
                "records": len(asset_rows),
            },
            "batches": batch_summaries,
        }
        _write_json(staging / "package_manifest.json", manifest)
        try:
            os.replace(staging, final_dir)
        except OSError:
            if not manifest_path.is_file():
                raise
            shutil.rmtree(staging, ignore_errors=True)
        latest = {
            "schema_version": 1,
            "package_id": package_id,
            "run_dir": f"runs/{package_id}",
            "package_manifest_sha256": _sha256_file(final_dir / "package_manifest.json"),
        }
        temporary_latest = output_dir / ".latest.json.tmp"
        _write_json(temporary_latest, latest)
        os.replace(temporary_latest, output_dir / "latest.json")
        final_manifest = _read_json(final_dir / "package_manifest.json")
        return ReviewPackResult(
            package_id, final_dir, final_dir / "package_manifest.json", final_manifest
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_review_package(package_dir: str | Path, *, verify_assets: bool = True) -> dict[str, Any]:
    root = Path(package_dir).expanduser().resolve()
    manifest = _read_json(root / "package_manifest.json")
    if manifest.get("kind") != "junction_human_review_pack" or manifest.get("pack_version") != PACK_VERSION:
        raise JunctionReviewPackError("unsupported review package")
    if manifest.get("classes") != list(JUNCTION_CLASSES):
        raise JunctionReviewPackError("review package class schema mismatch")
    policy = manifest.get("policy", {})
    for field, expected in (
        ("proposals_are_ground_truth", False),
        ("synthetic_is_real_truth", False),
        ("existing_truth_is_modified", False),
        ("val_test_may_enter_training", False),
    ):
        if policy.get(field) is not expected:
            raise JunctionReviewPackError(f"unsafe package policy: {field}")
    inputs = manifest.get("inputs", {})
    for field in ("audit", "derived_manifest", "source_manifest"):
        item = inputs.get(field, {})
        source = Path(str(item.get("path", "")))
        if not source.is_file() or _sha256_file(source) != item.get("sha256"):
            raise JunctionReviewPackError(f"package input lineage mismatch: {field}")
    queue = manifest.get("authoritative_queue", {})
    for field in ("jsonl", "csv"):
        path = _safe_child(root, str(queue.get(f"{field}_path", "")))
        if _sha256_file(path) != queue.get(f"{field}_sha256"):
            raise JunctionReviewPackError(f"authoritative {field} queue hash mismatch")
    asset_item = manifest.get("asset_manifest", {})
    asset_path = _safe_child(root, str(asset_item.get("path", "")))
    if _sha256_file(asset_path) != asset_item.get("sha256"):
        raise JunctionReviewPackError("asset manifest hash mismatch")
    asset_rows = _read_jsonl(asset_path)
    if len(asset_rows) != int(asset_item.get("records", -1)):
        raise JunctionReviewPackError("asset manifest record count mismatch")
    if verify_assets:
        seen: set[str] = set()
        for row in asset_rows:
            relative = str(row.get("path", ""))
            if not relative or relative in seen:
                raise JunctionReviewPackError(f"missing or duplicate asset path {relative!r}")
            seen.add(relative)
            path = _safe_child(root, relative)
            if path.stat().st_size != int(row.get("bytes", -1)):
                raise JunctionReviewPackError(f"asset byte count mismatch: {relative}")
            if _sha256_file(path) != row.get("sha256"):
                raise JunctionReviewPackError(f"asset hash mismatch: {relative}")
    return manifest


def _load_review_rows(path: Path) -> tuple[str, list[dict[str, Any]], tuple[str, ...] | None]:
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = tuple(reader.fieldnames or ())
                rows = [dict(row) for row in reader]
        except OSError as exc:
            raise JunctionReviewPackError(f"cannot read review CSV {path}: {exc}") from exc
        return "csv", rows, fields
    if path.suffix.lower() == ".jsonl":
        return "jsonl", _read_jsonl(path), None
    raise JunctionReviewPackError("review input must be .csv or .jsonl")


def _validate_reviewer(value: str) -> None:
    if not _REVIEWER_RE.fullmatch(value) or value.lower() in {"reviewer", "unknown", "n/a"}:
        raise JunctionReviewPackError("reviewer must be a non-placeholder human identity")


def _validate_timestamp(value: str) -> None:
    if not value or value.endswith("Z"):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    else:
        normalized = value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise JunctionReviewPackError(f"reviewed_at is not RFC3339/ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise JunctionReviewPackError("reviewed_at must contain an explicit timezone offset")


def validate_review_file(
    *,
    package_dir: str | Path,
    reviews_path: str | Path,
    require_complete: bool = False,
    verify_assets: bool = True,
) -> ReviewValidationResult:
    """Validate edited CSV/JSONL rows against the immutable package queue."""

    root = Path(package_dir).expanduser().resolve()
    manifest = verify_review_package(root, verify_assets=verify_assets)
    queue_path = _safe_child(root, str(manifest["authoritative_queue"]["jsonl_path"]))
    templates = _read_jsonl(queue_path)
    template_by_id = {str(row["physical_candidate_id"]): row for row in templates}
    if len(template_by_id) != len(templates):
        raise JunctionReviewPackError("authoritative queue contains duplicate candidate IDs")
    csv_template_by_id = {
        candidate_id: _csv_row(row) for candidate_id, row in template_by_id.items()
    }
    review_file = Path(reviews_path).expanduser().resolve()
    kind, rows, fields = _load_review_rows(review_file)
    if not rows:
        raise JunctionReviewPackError("review file is empty")
    if kind == "csv" and fields != CSV_FIELDS:
        raise JunctionReviewPackError("review CSV columns or order differ from the authoritative template")

    seen: set[str] = set()
    batch_ids: set[str] = set()
    canonical: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for line_number, raw in enumerate(rows, 2 if kind == "csv" else 1):
        candidate_id = str(raw.get("physical_candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise JunctionReviewPackError(
                f"{review_file}:{line_number}: missing or duplicate physical_candidate_id"
            )
        seen.add(candidate_id)
        template = template_by_id.get(candidate_id)
        if template is None:
            raise JunctionReviewPackError(f"unknown physical candidate ID {candidate_id!r}")
        if kind == "csv":
            authoritative_csv = csv_template_by_id[candidate_id]
            for field in IMMUTABLE_CSV_FIELDS:
                if str(raw.get(field, "")) != authoritative_csv[field]:
                    raise JunctionReviewPackError(
                        f"{candidate_id}: immutable field {field!r} was changed"
                    )
        else:
            missing_fields = set(template) - set(raw)
            extra = set(raw) - set(template)
            if missing_fields or extra:
                raise JunctionReviewPackError(
                    f"{candidate_id}: JSONL fields differ from the authoritative template; "
                    f"missing={sorted(missing_fields)!r}, extra={sorted(extra)!r}"
                )
            for field, expected in template.items():
                if field in HUMAN_FIELDS:
                    continue
                if raw.get(field) != expected:
                    raise JunctionReviewPackError(
                        f"{candidate_id}: immutable field {field!r} was changed"
                    )
        status = str(raw.get("review_status", "")).strip()
        label = str(raw.get("human_label", "")).strip()
        reviewer = str(raw.get("reviewer", "")).strip()
        reviewed_at = str(raw.get("reviewed_at", "")).strip()
        notes = str(raw.get("review_notes", "")).strip()
        if status not in REVIEW_STATUSES:
            raise JunctionReviewPackError(f"{candidate_id}: invalid review_status {status!r}")
        if status == "pending":
            if any((label, reviewer, reviewed_at, notes)):
                raise JunctionReviewPackError(
                    f"{candidate_id}: pending row must not contain human review fields"
                )
        elif status == "needs_adjudication":
            if label:
                raise JunctionReviewPackError(
                    f"{candidate_id}: needs_adjudication row must leave human_label empty"
                )
            _validate_reviewer(reviewer)
            _validate_timestamp(reviewed_at)
            if not notes:
                raise JunctionReviewPackError(
                    f"{candidate_id}: needs_adjudication requires review_notes"
                )
        else:
            if label not in JUNCTION_CLASSES:
                raise JunctionReviewPackError(f"{candidate_id}: invalid human_label {label!r}")
            _validate_reviewer(reviewer)
            _validate_timestamp(reviewed_at)
        batch_id = str(template["batch_id"])
        batch_ids.add(batch_id)
        counts[status] += 1
        canonical.append(
            {
                **template,
                "review_status": status,
                "human_label": label,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "review_notes": notes,
            }
        )

    expected_ids = {
        candidate_id
        for candidate_id, template in template_by_id.items()
        if str(template["batch_id"]) in batch_ids
    }
    if seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise JunctionReviewPackError(
            "review input must contain complete referenced batches; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    if require_complete and (counts["pending"] or counts["needs_adjudication"]):
        raise JunctionReviewPackError(
            "review input is incomplete: "
            f"pending={counts['pending']}, needs_adjudication={counts['needs_adjudication']}"
        )
    canonical.sort(key=lambda row: int(row["global_index"]))
    return ReviewValidationResult(
        package_id=str(manifest["package_id"]),
        batch_ids=tuple(sorted(batch_ids)),
        total=len(canonical),
        reviewed=counts["reviewed"],
        pending=counts["pending"],
        needs_adjudication=counts["needs_adjudication"],
        records=tuple(canonical),
    )


def import_human_reviews(
    *, package_dir: str | Path, reviews_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Write a separate, hash-bound human label artifact; never edit truth in place."""

    validation = validate_review_file(
        package_dir=package_dir,
        reviews_path=reviews_path,
        require_complete=True,
        verify_assets=True,
    )
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists() and any(destination.iterdir()):
        raise JunctionReviewPackError(f"import output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, Any]] = []
    for row in validation.records:
        split = str(row["split"])
        imported.append(
            {
                "schema_version": 1,
                "import_version": IMPORT_VERSION,
                "package_id": validation.package_id,
                "physical_candidate_id": row["physical_candidate_id"],
                "member_record_ids": row["member_record_ids"],
                "sample_id": row["sample_id"],
                "split": split,
                "source_domain": row["source_domain"],
                "physical_bbox": row["physical_bbox"],
                "label": row["human_label"],
                "label_source": "human_visual_review",
                "reviewer": row["reviewer"],
                "reviewed_at": row["reviewed_at"],
                "review_notes": row["review_notes"],
                "synthetic_source": False,
                "proposal_used_as_ground_truth": False,
                "training_eligibility": (
                    "separate_train_supervision_candidate"
                    if split == "train"
                    else "never_use_for_training"
                ),
                "source_integrity_sha256": row["integrity_sha256"],
            }
        )
    annotations_path = destination / "junction_human_reviews.jsonl"
    _write_jsonl(annotations_path, imported)
    receipt = {
        "schema_version": 1,
        "import_version": IMPORT_VERSION,
        "kind": "junction_human_review_import_receipt",
        "package_id": validation.package_id,
        "batch_ids": list(validation.batch_ids),
        "records": len(imported),
        "by_split": dict(sorted(Counter(str(row["split"]) for row in imported).items())),
        "by_label": dict(sorted(Counter(str(row["label"]) for row in imported).items())),
        "annotations_path": annotations_path.name,
        "annotations_sha256": _sha256_file(annotations_path),
        "existing_truth_modified": False,
        "synthetic_records": 0,
        "val_test_training_exclusion_enforced": True,
    }
    _write_json(destination / "import_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a physical-cluster review pack")
    build.add_argument(
        "--audit",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "audits" / "junction_truth_audit.json",
    )
    build.add_argument(
        "--derived-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "derived" / "dataset300-v2" / "junction_patches.jsonl",
    )
    build.add_argument(
        "--source-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "manifests" / "dataset300.jsonl",
    )
    build.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    build.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "reviews" / "junction-real-v1",
    )
    build.add_argument("--batch-size", type=int, default=100)
    build.add_argument("--context-size", type=int, default=192)

    validate = subparsers.add_parser("validate", help="strictly validate an edited batch")
    validate.add_argument("--package-dir", type=Path, required=True)
    validate.add_argument("--reviews", type=Path, required=True)
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--skip-asset-verification", action="store_true")

    import_parser = subparsers.add_parser(
        "import", help="create a separate human-label artifact from a completed batch"
    )
    import_parser.add_argument("--package-dir", type=Path, required=True)
    import_parser.add_argument("--reviews", type=Path, required=True)
    import_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_review_pack(
            audit_path=args.audit,
            derived_manifest=args.derived_manifest,
            source_manifest=args.source_manifest,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            batch_size=args.batch_size,
            context_size=args.context_size,
        )
        payload = {
            "ok": True,
            "package_id": result.package_id,
            "run_dir": str(result.run_dir),
            "manifest": str(result.manifest_path),
            "counts": result.manifest["counts"],
        }
    elif args.command == "validate":
        result = validate_review_file(
            package_dir=args.package_dir,
            reviews_path=args.reviews,
            require_complete=args.require_complete,
            verify_assets=not args.skip_asset_verification,
        )
        payload = {
            "ok": True,
            "package_id": result.package_id,
            "batch_ids": list(result.batch_ids),
            "total": result.total,
            "reviewed": result.reviewed,
            "pending": result.pending,
            "needs_adjudication": result.needs_adjudication,
        }
    else:
        payload = import_human_reviews(
            package_dir=args.package_dir,
            reviews_path=args.reviews,
            output_dir=args.output_dir,
        )
        payload = {"ok": True, **payload}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
