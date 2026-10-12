#!/usr/bin/env python3
"""opt90 wire materialization with explicit negative-patch injection (goal line B).

Reuses the production ``wire_junction_materialize`` publication protocol
(content-addressed runs/<run_id> + latest.json + summary hash bindings) with an
*extended, honestly hashed* materializer config.  Two injection mechanisms,
both purely data-layer:

A. supervision flip  : inside NEG_eff = NEG - dilate(GT wire, 3px) the ignore
   mask is cleared and the target stays 0 -> those pixels become explicit
   negative supervision for the first time (symbol/decoration strokes used to
   live inside ignored component/text bboxes).
B. explicit negpatch : per sample, K offset 512 tiles are cropped around the
   NEG_eff hot spots (greedy on the NEG_eff integral image), record ids
   ``negp_*``.  They up-sample hard negatives and act as shift augmentation.

Everything outside NEG_eff keeps the exact original semantics (target drawn
from keyPointPaths+skeleton, ignore = component/text bbox + pin line, ignore
wins).  The NEG masks are copied into the run as hashed assets, so the run is
self-contained and lineage (materializer_config -> negative_label_tree_sha256)
is verifiable.

Usage:
  python opt90_materialize.py --neg-label-dir work/opt90-wire/neg-labels \
      --output-root work/opt90-wire/materialized --workers 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from pcbparse.training import wire_junction_materialize as wjm  # noqa: E402

PROTECT_MARGIN_PX = 3
NEG_DILATE_PX = 2


@dataclass(frozen=True, slots=True)
class Opt90MaterializeConfig(wjm.WireJunctionMaterializeConfig):
    """Base materializer config + opt90 explicit negative-patch injection."""

    negative_label_dir: str = ""
    negative_label_tree_sha256: str = ""
    neg_dilate_px: int = NEG_DILATE_PX
    protect_margin_px: int = PROTECT_MARGIN_PX
    flip_policy: str = "neg_eff_unignore_target_zero"
    neg_patch_per_sample: int = 6
    neg_patch_stride_px: int = 128
    neg_patch_min_gap_px: int = 256
    materializer_script: str = "work/opt90-wire/opt90_materialize.py"

    def __post_init__(self) -> None:
        # slots=True dataclasses rebuild the class; call the base validator
        # explicitly instead of the zero-arg super()
        wjm.WireJunctionMaterializeConfig.__post_init__(self)
        if not self.negative_label_dir:
            raise ValueError("negative_label_dir is required")
        if self.neg_dilate_px < 0 or self.protect_margin_px < 0:
            raise ValueError("opt90 margins must be non-negative")
        if self.neg_patch_per_sample < 0:
            raise ValueError("neg_patch_per_sample must be non-negative")
        if self.neg_patch_stride_px < 8 or self.neg_patch_min_gap_px < 0:
            raise ValueError("neg patch stride/gap are invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_NEG_CACHE: dict[str, np.ndarray] = {}


def _load_neg(sample_id: str, cfg: Opt90MaterializeConfig, size: tuple[int, int]) -> np.ndarray | None:
    """NEG masks exist for train samples only (the supervised population).

    Val/test samples return ``None``: their tiles keep the exact original
    semantics (no injection is derived from evaluation data).
    """
    cached = _NEG_CACHE.get(sample_id)
    if cached is not None:
        return cached
    path = (ROOT / cfg.negative_label_dir / f"NEG_{sample_id}.png").resolve()
    if not path.is_file():
        _NEG_CACHE[sample_id] = None
        return None
    with Image.open(path) as opened:
        opened.load()
        image = opened.convert("L")
    if image.size != size:
        raise wjm.MaterializationError(
            f"opt90 NEG mask size {image.size} differs from canvas {size} for {sample_id!r}"
        )
    mask = np.asarray(image) > 0
    _NEG_CACHE[sample_id] = mask
    return mask


def _full_target(geometries, width: int, height: int, cfg) -> Image.Image:
    """Exactly the materializer's target rasterisation (line width 1 + points)."""
    target = Image.new("L", (width, height), cfg.mask_background)
    draw = ImageDraw.Draw(target)
    for geometry in geometries:
        if geometry.bbox is None:
            continue
        for path in geometry.paths:
            pts = [(wjm._pixel_x(px), wjm._pixel_y(py)) for px, py in path]
            if len(pts) == 1:
                draw.point(pts[0], fill=cfg.mask_foreground)
            else:
                draw.line(pts, fill=cfg.mask_foreground, width=1)
        for px, py in geometry.skeleton:
            draw.point((wjm._pixel_x(px), wjm._pixel_y(py)), fill=cfg.mask_foreground)
    return target


def _pick_neg_patches(neg_eff: np.ndarray, cfg: Opt90MaterializeConfig) -> list[tuple[int, int, int, int]]:
    """Greedy top-K 512 windows on the NEG_eff integral image."""
    if cfg.neg_patch_per_sample <= 0:
        return []
    height, width = neg_eff.shape
    if width <= cfg.tile_size and height <= cfg.tile_size:
        return []  # single grid tile already covers the canvas
    integral = np.pad(neg_eff.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def window_count(x1: int, y1: int, x2: int, y2: int) -> int:
        return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    candidates: list[tuple[int, int, int]] = []
    for y1 in range(0, max(1, height - cfg.tile_size + 1), cfg.neg_patch_stride_px):
        for x1 in range(0, max(1, width - cfg.tile_size + 1), cfg.neg_patch_stride_px):
            x2 = min(width, x1 + cfg.tile_size)
            y2 = min(height, y1 + cfg.tile_size)
            count = window_count(x1, y1, x2, y2)
            if count > 0:
                candidates.append((count, x1, y1))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen: list[tuple[int, int, int, int]] = []
    for count, x1, y1 in candidates:
        x2 = min(width, x1 + cfg.tile_size)
        y2 = min(height, y1 + cfg.tile_size)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if any(
            abs(cx - (a + c) / 2.0) < cfg.neg_patch_min_gap_px
            and abs(cy - (b + d) / 2.0) < cfg.neg_patch_min_gap_px
            for a, b, c, d in chosen
        ):
            continue
        chosen.append((x1, y1, x2, y2))
        if len(chosen) >= cfg.neg_patch_per_sample:
            break
    return chosen


def opt90_write_wire_tiles(
    *,
    staging: Path,
    sample_id: str,
    split_name: str,
    source_row: dict,
    json_sha: str,
    image_sha: str,
    canvas: dict[str, int],
    geometries: list,
    unusable_sparse_record_ids: list[str],
    ignore_regions: list[dict],
    run_id: str,
    cfg: Opt90MaterializeConfig,
):
    """Drop-in replacement of wjm._write_wire_tiles with opt90 injection."""
    width, height = canvas["width"], canvas["height"]
    neg_full = _load_neg(sample_id, cfg, (width, height))
    if neg_full is None:
        if split_name == "train":
            raise wjm.MaterializationError(
                f"opt90 NEG mask missing for train sample {sample_id!r}"
            )
        neg_full = np.zeros((height, width), dtype=bool)
    injection_active = bool(neg_full.any())
    gt_full = np.asarray(_full_target(geometries, width, height, cfg)) > 0
    gt_dil = np.asarray(
        Image.fromarray((gt_full * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(2 * cfg.protect_margin_px + 1)
        )
    ) > 0
    neg_eff_full = neg_full & ~gt_dil

    records: list[dict] = []
    assets: list[dict] = []
    unusable = bool(unusable_sparse_record_ids)

    neg_asset_rel = Path("wire") / "neg_masks" / f"{sample_id}.png"
    if injection_active:
        neg_asset_path = staging / neg_asset_rel
        neg_asset_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((neg_full * 255).astype(np.uint8)).save(neg_asset_path)
        assets.append(
            {
                "path": neg_asset_rel.as_posix(),
                "sha256": _sha256_file(neg_asset_path),
                "bytes": neg_asset_path.stat().st_size,
                "kind": "opt90_negative_mask",
            }
        )

    grid_boxes = []
    for y1 in range(0, height, cfg.tile_size):
        y2 = min(height, y1 + cfg.tile_size)
        for x1 in range(0, width, cfg.tile_size):
            x2 = min(width, x1 + cfg.tile_size)
            grid_boxes.append((x1, y1, x2, y2))
    patch_boxes = [] if (unusable or not injection_active) else _pick_neg_patches(neg_eff_full, cfg)

    def emit_tile(tile_bbox, tile_id: str, is_negpatch: bool, neg_eff_px: int):
        x1, y1, x2, y2 = tile_bbox
        tw, th = x2 - x1, y2 - y1
        target = Image.new("L", (tw, th), cfg.mask_background)
        target_draw = ImageDraw.Draw(target)
        edge_record_ids: list[str] = []
        for geometry in geometries:
            if geometry.bbox is None or not wjm._bbox_intersects(geometry.bbox, tile_bbox):
                continue
            used = False
            for path in geometry.paths:
                if not wjm._bbox_intersects(wjm._points_bbox(path), tile_bbox):
                    continue
                local = [(wjm._pixel_x(px) - x1, wjm._pixel_y(py) - y1) for px, py in path]
                if len(local) == 1:
                    target_draw.point(local[0], fill=cfg.mask_foreground)
                else:
                    target_draw.line(local, fill=cfg.mask_foreground, width=1)
                used = True
            for px, py in geometry.skeleton:
                ix, iy = wjm._pixel_x(px), wjm._pixel_y(py)
                if x1 <= ix < x2 and y1 <= iy < y2:
                    target_draw.point((ix - x1, iy - y1), fill=cfg.mask_foreground)
                    used = True
            if used:
                edge_record_ids.append(geometry.record_id)

        ignore = Image.new("L", (tw, th), cfg.mask_background)
        ignore_draw = ImageDraw.Draw(ignore)
        tile_regions: list[dict] = []
        if unusable:
            ignore_draw.rectangle((0, 0, tw - 1, th - 1), fill=cfg.mask_foreground)
            tile_regions.append(
                {
                    "kind": "sample_unusable_sparse_geometry",
                    "source_record_ids": sorted(unusable_sparse_record_ids),
                    "image_bbox": [x1, y1, x2, y2],
                    "tile_bbox": [0, 0, tw, th],
                }
            )
        else:
            for region in ignore_regions:
                clipped = wjm._draw_ignore_region(region, ignore_draw, target_draw, tile_bbox, cfg)
                if clipped is not None:
                    tile_regions.append(clipped)
            if cfg.pin_line_policy == "positive":
                for region in ignore_regions:
                    if region["kind"] != "pin_line":
                        continue
                    wjm._draw_positive_pin(region, target_draw, tile_bbox, cfg)

        # ---- opt90 injection: flip NEG_eff pixels to explicit negatives -----
        neg_eff_tile = neg_eff_full[y1:y2, x1:x2]
        gt_drawn = np.asarray(target) > 0  # target after ignore-region clearing
        ignore_orig_arr = np.asarray(ignore) > 0
        # ineligible samples (unusable sparse geometry) must stay fully ignored:
        # no negative flip and no positive restore may apply there.
        if neg_eff_tile.any() and not unusable:
            # invariant 1: NEG_eff is disjoint from GT wire (±protect margin),
            # so no true-wire pixel can ever be flipped to negative supervision
            if bool((gt_drawn & neg_eff_tile).any()):
                raise wjm.MaterializationError(
                    f"opt90 NEG_eff overlaps GT wire pixels in {sample_id}:{tile_id}"
                )
            ignore_arr = ignore_orig_arr & ~neg_eff_tile
            ignore = Image.fromarray((ignore_arr * 255).astype(np.uint8))
            flipped = int((ignore_orig_arr & ~ignore_arr).sum())
        else:
            flipped = 0
        # original semantics outside NEG_eff: ignore always wins over target.
        # invariant 2: the positive target is byte-identical to the original
        # materializer's target (the injection only shrinks the ignore mask).
        target_new_arr = gt_drawn & ~(np.asarray(ignore) > 0)
        target_orig_arr = gt_drawn & ~ignore_orig_arr
        if not np.array_equal(target_new_arr, target_orig_arr):
            raise wjm.MaterializationError(
                f"opt90 injection changed the positive target in {sample_id}:{tile_id}"
            )
        target = Image.fromarray((target_new_arr * 255).astype(np.uint8))

        target_rel = Path("wire") / "target_masks" / sample_id / f"{tile_id}.png"
        ignore_rel = Path("wire") / "ignore_masks" / sample_id / f"{tile_id}.png"
        target_path = staging / target_rel
        ignore_path = staging / ignore_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        ignore_path.parent.mkdir(parents=True, exist_ok=True)
        wjm._save_png(target, target_path)
        wjm._save_png(ignore, ignore_path)
        target_sha = wjm._sha256_file(target_path)
        ignore_sha = wjm._sha256_file(ignore_path)
        for rel, path, sha, kind in (
            (target_rel, target_path, target_sha, "wire_target_mask"),
            (ignore_rel, ignore_path, ignore_sha, "wire_ignore_mask"),
        ):
            assets.append(
                {"path": rel.as_posix(), "sha256": sha, "bytes": path.stat().st_size, "kind": kind}
            )
        files = source_row["files"]
        records.append(
            {
                "schema_version": 1,
                "record_id": f"{sample_id}:wire_tile:{tile_id}",
                "sample_id": sample_id,
                "split": split_name,
                "tile_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "tile_size": [tw, th],
                "target_mask": {
                    "path": target_rel.as_posix(),
                    "sha256": target_sha,
                    "foreground_pixels": wjm._mask_pixel_count(target, cfg.mask_foreground),
                },
                "ignore_mask": {
                    "path": ignore_rel.as_posix(),
                    "sha256": ignore_sha,
                    "foreground_pixels": wjm._mask_pixel_count(ignore, cfg.mask_foreground),
                    "regions": tile_regions,
                },
                "transform": {
                    "coordinate_convention": "pixel_xy_half_open_bbox",
                    "tile_to_image": [[1.0, 0.0, float(x1)], [0.0, 1.0, float(y1)], [0.0, 0.0, 1.0]],
                    "image_to_tile": [[1.0, 0.0, float(-x1)], [0.0, 1.0, float(-y1)], [0.0, 0.0, 1.0]],
                },
                "source": {
                    "image": files["image"]["path"],
                    "image_sha256": image_sha,
                    "json": files["json"]["path"],
                    "json_sha256": json_sha,
                    "image_crop_included": False,
                    "derived_wire_record_ids": sorted(edge_record_ids),
                },
                "supervision": {
                    "eligible": not unusable,
                    "ineligible_reason": (
                        "one_or_more_wire_records_lack_keyPointPaths_and_skeleton"
                        if unusable
                        else None
                    ),
                    "unusable_sparse_record_ids": sorted(unusable_sparse_record_ids),
                },
                "lineage": {
                    "run_id": run_id,
                    "geometry_sources": ["keyPointPaths", "keyPoints/skeleton"],
                    "dense_coordinates_loaded": False,
                },
                "opt90": {
                    "negative_patch": is_negpatch,
                    "injection_active": injection_active,
                    "neg_eff_pixels_in_tile": int(neg_eff_tile.sum()),
                    "ignore_pixels_flipped_to_negative": flipped,
                    "flip_policy": cfg.flip_policy,
                },
            }
        )

    for tile_bbox in grid_boxes:
        x1, y1, x2, y2 = tile_bbox
        neg_eff_px = int(neg_eff_full[y1:y2, x1:x2].sum())
        emit_tile(tile_bbox, f"y{y1:06d}_x{x1:06d}", False, neg_eff_px)
    for x1, y1, x2, y2 in patch_boxes:
        neg_eff_px = int(neg_eff_full[y1:y2, x1:x2].sum())
        emit_tile((x1, y1, x2, y2), f"negp_y{y1:06d}_x{x1:06d}", True, neg_eff_px)
    return records, assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg-label-dir", default="work/opt90-wire/neg-labels")
    parser.add_argument("--stats", default="")
    parser.add_argument("--samples", default="")
    parser.add_argument("--allow-sig-err-over-gate", action="store_true",
                        help="recorded override for smoke/validation runs only")
    parser.add_argument("--output-root", default="work/opt90-wire/materialized")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    neg_dir = (ROOT / args.neg_label_dir).resolve()
    stats_path = (ROOT / args.stats).resolve() if args.stats else neg_dir / "neg_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"missing negative-label stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    totals = stats.get("totals", {})
    if totals.get("errors", 1) != 0:
        raise SystemExit("negative labeler reported errors; refusing to materialize")
    err_rate = float(totals.get("sig_err_rate", totals.get("err_rate", 1.0)))
    # Lead ruling 2026-10-08 (PREREG amendment, work/opt90-wire/DESIGN.md):
    # the <2% 误伤 gate binds the TRAINING supervision (NEG), whose 误伤 rate is
    # 0 by construction and asserted per tile below.  SIG is stratification
    # only (never supervision); its band-metric rate is recorded, not gating.
    sig_gate_override = bool(err_rate >= 0.02 and args.allow_sig_err_over_gate)

    # tree hash over all NEG pngs, cross-checked against the config field
    entries = sorted(neg_dir.glob("NEG_*.png"))
    digest = hashlib.sha256()
    for path in entries:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    tree_sha = digest.hexdigest()

    cfg = Opt90MaterializeConfig(
        negative_label_dir=args.neg_label_dir,
        negative_label_tree_sha256=tree_sha,
    )

    # route the production materializer through the opt90 tile writer
    wjm._write_wire_tiles = opt90_write_wire_tiles  # type: ignore[assignment]
    sample_ids = [value.strip() for value in args.samples.split(",") if value.strip()] or None
    result = wjm.materialize_wire_junction(
        wire_manifest=ROOT / "data" / "derived" / "dataset300-v2" / "wire_paths.jsonl",
        junction_manifest=ROOT / "data" / "derived" / "dataset300-v2" / "junction_patches.jsonl",
        source_manifest=ROOT / "data" / "manifests" / "dataset300.jsonl",
        split_manifest=ROOT / "data" / "splits" / "dataset300_split.jsonl",
        source_root=ROOT / "Dataset300" / "Dataset300",
        output_root=ROOT / args.output_root,
        config=cfg,
        expected_derivation_version="dataset300-v2",
        expected_config_hash="3381e705c06642cacf8d3e9d82afbf2f725d3e925b67b049a57bd9b858632009",
        sample_ids=sample_ids,
    )
    summary = result.summary
    document = {
        "schema_version": 1,
        "kind": "opt90_materialization_receipt",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "output_root": args.output_root,
        "selected_sample_ids": sample_ids or "all",
        "materializer_config_sha256": summary.get("materializer_config_sha256"),
        "negative_label_tree_sha256": tree_sha,
        "negative_label_count": len(entries),
        "negative_labeler_stats": str(stats_path.relative_to(ROOT)),
        "sig_err_rate": err_rate,
        "sig_gate_override": sig_gate_override,
        "gate_binding": "NEG: zero 误伤 by construction, asserted per tile (GT wire pixels never negative); SIG informational only (Lead ruling 2026-10-08)",
        "wire_tile_count": summary.get("materialization", {}).get("wire_tile_count"),
    }
    receipt_name = "materialize_receipt_smoke.json" if sample_ids else "materialize_receipt.json"
    receipt = ROOT / "work" / "opt90-wire" / receipt_name
    receipt.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
