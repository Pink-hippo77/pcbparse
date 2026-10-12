#!/usr/bin/env python3
"""opt90 negative labeler v2 (goal line B) — training negatives for wire retrain.

Two products per train240 sample, both measured against GT wire geometry:

1. **NEG (training negative region)** — the *current production wire model's own
   false-positive pixel mass* on that image: model-fired mask pixels further
   than ``GT_TOL_PX`` from any GT wire centreline.  This is the population that
   must die for wire precision 0.691 -> 0.90 (r12 attribution: the residual
   51k px are symbol/decoration strokes the deterministic rules cannot catch).
   Hard-negative mining: the same pixels stay negatives for the retrained model.

2. **SIG (prescribed four-signature annotation)** — the r12 decorative
   four-signature annotator (production ``filter_decorative_wire_segments``)
   runs on the very same model-fired segments (production vectorizer output,
   r12-replay semantics) and marks "symbol/decoration strokes".  Its 误伤率 vs
   GT wire (tolerance 3px, masked_metrics口径) is the pre-registered <2% gate.
   Signature layer counts (canvas_span / companion / rect_outline / interior)
   stratify the NEG population in the report.

Pipeline per sample (offline; GT is legitimate training-data material):
  production model (models.lock wire entry) -> tiled inference (production
  tiling) -> canvas mask merge -> production ``_vectorize_masks`` -> segments
  -> GT-gated CircuitIR -> production four-signature annotator -> NEG/SIG rasters.

Outputs under work/opt90-wire/neg-labels/:
  NEG_<sample>.png  L-mode, 255 = training negative pixels (protected: disjoint
                    from GT wire +-3px by construction)
  SIG_<sample>.png  L-mode, 255 = four-signature-marked stroke pixels (raw,
                    before GT protection - this is what the 误伤率 gate covers)
  neg_stats.json    aggregate + per-sample rows

Usage:
  python opt90_neg_labeler.py [--workers N] [--limit K] [--samples 0001,0002]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from pcbparse.graph.physical import filter_decorative_wire_segments  # noqa: E402
from pcbparse.ir.models import (  # noqa: E402
    BBox,
    CircuitIR,
    Component,
    ImageInfo,
    NetLabel,
    Point,
    WireSegment,
)
from pcbparse.io.image_loader import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.composite_provider import _vectorize_masks  # noqa: E402
from pcbparse.perception.production_provider import (  # noqa: E402
    _apply_wire_inference_overrides,
    composite_provider_config_from_pipeline,
)
from pcbparse.perception.torch_wire_adapter import TorchWireAdapter  # noqa: E402
from pcbparse.training import wire_junction_materialize as wjm  # noqa: E402

DATASET_ROOT = ROOT / "Dataset300" / "Dataset300"
WIRE_MANIFEST = ROOT / "data" / "derived" / "dataset300-v2" / "wire_paths.jsonl"
OUT_DIR = ROOT / "work" / "opt90-wire" / "neg-labels"
LOCK_PATH = ROOT / "models" / "models.lock.json"
PIPELINE_CFG = ROOT / "work" / "module-opt80-20261007" / "r12-integration" / "pipeline-integration.yaml"

GT_TOL_PX = 3          # masked_metrics tolerance (qna-v1 wire metrics) == protection margin
SIG_DILATE_PX = 2      # segment skeleton -> stroke half width

_RUNTIME: dict = {}
_SIG_PARAMS: dict = {}


def set_sig_params(params: dict) -> None:
    """Four-signature annotator parameters (tightening knob for the 误伤 gate)."""
    _SIG_PARAMS.clear()
    _SIG_PARAMS.update(params)


def set_out_dir(path: str | Path) -> None:
    global OUT_DIR
    OUT_DIR = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime() -> dict:
    """Load the production wire model + resolved pipeline settings once per worker."""
    if _RUNTIME:
        return _RUNTIME
    import torch

    torch.set_num_threads(2)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    entries = [e for e in lock["models"] if e.get("task") == "wire"]
    if len(entries) != 1:
        raise RuntimeError("production lock must contain exactly one wire entry")
    entry = dict(entries[0])
    model_path = (ROOT / "models" / entry["path"]).resolve(strict=True)
    if _sha256_file(model_path) != entry["sha256"]:
        raise RuntimeError(f"wire model hash mismatch: {model_path}")
    adapter = TorchWireAdapter(model_path, entry=entry, device="cpu")

    from pcbparse.cli import _apply_profile, _load_profile
    from pcbparse.config import load_config

    cfg = _apply_profile(load_config(PIPELINE_CFG), _load_profile("competition"))
    composite = composite_provider_config_from_pipeline(cfg)
    _apply_wire_inference_overrides(adapter, composite.wire_inference_overrides)
    _RUNTIME.update(
        adapter=adapter,
        composite=composite,
        model_path=model_path,
        model_sha256=entry["sha256"],
        threshold=float(adapter.threshold),
    )
    return _RUNTIME


def _draw_geometry_target(geometries, width: int, height: int) -> Image.Image:
    """Replicate the materializer target rasterisation (line width 1 + points)."""
    target = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(target)
    for geometry in geometries:
        if geometry.bbox is None:
            continue
        for path in geometry.paths:
            pts = [(round(x), round(y)) for x, y in path]
            if len(pts) == 1:
                draw.point(pts[0], fill=255)
            else:
                draw.line(pts, fill=255, width=1)
        for px, py in geometry.skeleton:
            draw.point((round(px), round(py)), fill=255)
    return target


def _gt_regions_ir(annotation: dict, canvas: tuple[int, int]) -> tuple[list[Component], list[NetLabel]]:
    """GT component bboxes / GT text points as the far-gate geometry."""
    width, height = canvas
    cfg = wjm.WireJunctionMaterializeConfig()
    regions = wjm._collect_ignore_regions(annotation, {"width": width, "height": height}, cfg)
    components: list[Component] = []
    labels: list[NetLabel] = []
    for region in regions:
        if region["kind"] == "component_bbox":
            x1, y1, x2, y2 = region["bbox"]
            components.append(
                Component(
                    internal_id=f"C_{region['source_id']}",
                    category="unknown",
                    bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                )
            )
        elif region["kind"] == "text_bbox":
            x1, y1, x2, y2 = region["bbox"]
            labels.append(
                NetLabel(
                    internal_id=f"T_{region['source_id']}",
                    instance_text="",
                    resolved_text="",
                    point=Point(x=(float(x1) + float(x2)) / 2.0, y=(float(y1) + float(y2)) / 2.0),
                )
            )
    return components, labels


def process_sample(sample_id: str, rows: list[dict]) -> dict:
    started = time.time()
    runtime = _runtime()
    adapter = runtime["adapter"]
    composite = runtime["composite"]

    image_rel = rows[0]["source_image"]
    json_rel = rows[0]["source_json"]
    image_path = DATASET_ROOT / image_rel
    json_path = DATASET_ROOT / json_rel
    annotation, _dense = wjm._load_annotation_without_dense_coordinates(json_path)
    original = load_image(image_path)
    width, height = original.width, original.height

    # 1) GT wire target raster (exact training-target geometry sources)
    geometries = []
    for row in rows:
        if row.get("category") != "wire":
            continue
        geometry = wjm._resolve_wire_geometry(row, annotation, ("wire",))
        if geometry.bbox is not None:
            geometries.append(geometry)
    gt_mask = _draw_geometry_target(geometries, width, height)
    gt_px = int(np.count_nonzero(np.asarray(gt_mask)))
    gt_dil_arr = np.asarray(gt_mask.filter(ImageFilter.MaxFilter(2 * GT_TOL_PX + 1))) > 0

    # 2) production tiled inference + canvas mask merge (r12-replay semantics)
    views = tile_image_view(
        original,
        tile_size=composite.resolved_wire_tile_size,
        overlap=composite.resolved_wire_tile_overlap,
    )
    view_map = {view.view_id: view for view in views}
    masks = adapter.segment(views)
    canvas_mask = np.zeros((height, width), dtype=bool)
    for wire_mask in masks:
        view = view_map[wire_mask.view_id]
        origin = view.to_original(Point(x=0.0, y=0.0))
        x0, y0 = int(round(origin.x)), int(round(origin.y))
        array = np.asarray(wire_mask.mask, dtype=bool)
        canvas_mask[y0 : y0 + view.height, x0 : x0 + view.width] |= array
    model_px = int(canvas_mask.sum())

    # 3) production vectoriser -> segments -> production four-signature annotator
    segments, _junctions, _lineage = _vectorize_masks(masks, view_map, composite)
    components, labels = _gt_regions_ir(annotation, (width, height))
    ir = CircuitIR(
        image=ImageInfo(image_id=sample_id, path=image_rel, width=width, height=height),
        components=components,
        net_labels=labels,
        segments=list(segments),
    )
    report = filter_decorative_wire_segments(ir, **_SIG_PARAMS)
    suppressed_ids = set(report.get("ids", []))
    seg_index = {segment.internal_id: segment for segment in segments}

    # 4) SIG raster: four-signature-marked strokes (raw, pre-protection)
    sig = Image.new("L", (width, height), 0)
    sig_draw = ImageDraw.Draw(sig)
    for seg_id in suppressed_ids:
        segment = seg_index[seg_id]
        sig_draw.line(
            [
                (round(segment.start.x), round(segment.start.y)),
                (round(segment.end.x), round(segment.end.y)),
            ],
            fill=255,
            width=1,
        )
    if suppressed_ids:
        sig = sig.filter(ImageFilter.MaxFilter(2 * SIG_DILATE_PX + 1))
    sig_arr = np.asarray(sig) > 0
    sig_px = int(sig_arr.sum())
    sig_hit_px = int((sig_arr & gt_dil_arr).sum())
    sig_err_rate = (sig_hit_px / sig_px) if sig_px else 0.0

    # 5) NEG = model false-positive pixels (protected from GT wire by construction)
    neg_arr = canvas_mask & ~gt_dil_arr
    near_miss_px = int((canvas_mask & gt_dil_arr).sum())
    neg_px = int(neg_arr.sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray((neg_arr * 255).astype(np.uint8)).save(OUT_DIR / f"NEG_{sample_id}.png")
    sig.save(OUT_DIR / f"SIG_{sample_id}.png")

    return {
        "sample_id": sample_id,
        "canvas": [width, height],
        "image": image_rel,
        "gt_px": gt_px,
        "gt_dilated_px": int(gt_dil_arr.sum()),
        "model_mask_px": model_px,
        "model_near_miss_px": near_miss_px,
        "neg_px": neg_px,
        "tile_count": len(views),
        "segment_count": len(segments),
        "suppressed_segments": len(suppressed_ids),
        "layers": report.get("layers", {}),
        "sig_px": sig_px,
        "sig_hit_gt_px": sig_hit_px,
        "sig_err_rate": round(sig_err_rate, 6),
        "elapsed_s": round(time.time() - started, 2),
    }


def _worker(args):
    sample_id, rows = args
    try:
        row = process_sample(sample_id, rows)
        print(json.dumps({"done": sample_id, "elapsed_s": row.get("elapsed_s"),
                          "neg_px": row.get("neg_px"), "sig_px": row.get("sig_px"),
                          "sig_err_rate": row.get("sig_err_rate")}, ensure_ascii=False), flush=True)
        return row
    except Exception:
        return {"sample_id": sample_id, "error": traceback.format_exc(limit=8)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=str, default="")
    parser.add_argument("--out-dir", default="work/opt90-wire/neg-labels")
    parser.add_argument("--span-fraction", type=float, default=None)
    parser.add_argument("--rect-min-area", type=float, default=None)
    parser.add_argument("--interior-inset", type=float, default=None)
    parser.add_argument("--companion-max-gap", type=float, default=None)
    args = parser.parse_args()
    sig_params = {
        name: value
        for name, value in (
            ("span_fraction", args.span_fraction),
            ("rect_min_area_px", args.rect_min_area),
            ("interior_inset_px", args.interior_inset),
            ("companion_max_gap_px", args.companion_max_gap),
        )
        if value is not None
    }
    set_sig_params(sig_params)  # before Pool creation: forked workers inherit
    set_out_dir(args.out_dir)

    rows_by_sample: dict[str, list[dict]] = {}
    with WIRE_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            rows_by_sample.setdefault(row["sample_id"], []).append(row)
    sample_ids = sorted(rows_by_sample)
    if args.samples:
        wanted = {value.strip() for value in args.samples.split(",") if value.strip()}
        sample_ids = [value for value in sample_ids if value in wanted]
    if args.limit:
        sample_ids = sample_ids[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tasks = [(sample_id, rows_by_sample[sample_id]) for sample_id in sample_ids]
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, tasks, chunksize=1)
    else:
        results = [_worker(task) for task in tasks]

    rows = sorted(results, key=lambda item: item["sample_id"])
    errors = [row for row in rows if "error" in row]
    done = [row for row in rows if "error" not in row]
    totals = {
        "samples": len(sample_ids),
        "ok": len(done),
        "errors": len(errors),
        "gt_px": sum(row["gt_px"] for row in done),
        "model_mask_px": sum(row["model_mask_px"] for row in done),
        "model_near_miss_px": sum(row["model_near_miss_px"] for row in done),
        "neg_px": sum(row["neg_px"] for row in done),
        "segment_count": sum(row["segment_count"] for row in done),
        "suppressed_segments": sum(row["suppressed_segments"] for row in done),
        "sig_px": sum(row["sig_px"] for row in done),
        "sig_hit_gt_px": sum(row["sig_hit_gt_px"] for row in done),
    }
    totals["sig_err_rate"] = (
        round(totals["sig_hit_gt_px"] / totals["sig_px"], 6) if totals["sig_px"] else 0.0
    )
    layer_totals: dict[str, int] = {}
    for row in done:
        for name, value in row.get("layers", {}).items():
            layer_totals[name] = layer_totals.get(name, 0) + int(value)

    runtime = _runtime()
    document = {
        "schema_version": 2,
        "kind": "opt90_wire_negative_labeler",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "stroke_source": "production wire model fired pixels (models.lock wire entry, tiled production inference + production _vectorize_masks)",
            "model_path": str(runtime["model_path"].relative_to(ROOT)),
            "model_sha256": runtime["model_sha256"],
            "models_lock_sha256": _sha256_file(LOCK_PATH),
            "wire_paths_manifest_sha256": _sha256_file(WIRE_MANIFEST),
            "pipeline_config": str(PIPELINE_CFG.relative_to(ROOT)),
            "pipeline_config_sha256": _sha256_file(PIPELINE_CFG),
            "labeler_script_sha256": _sha256_file(Path(__file__).resolve()),
            "reproduce": "PYTHONPATH=src python work/opt90-wire/opt90_neg_labeler.py --workers 24",
            "wire_threshold": runtime["threshold"],
            "annotator": "pcbparse.graph.physical.filter_decorative_wire_segments (production four signatures) on the same model segments, GT-gated",
            "sig_params_overrides": sig_params,
            "neg_definition": "model-fired pixels farther than %dpx from GT wire centrelines (hard negatives; disjoint from GT by construction)" % GT_TOL_PX,
            "sig_definition": "four-signature-marked stroke pixels, dilated %dpx, RAW (pre-protection); 误伤率 gate applies here" % SIG_DILATE_PX,
            "gt_geometry": "wire_paths.jsonl keyPointPaths + skeleton (identical to training target)",
            "gt_tolerance_px": GT_TOL_PX,
        },
        "totals": totals,
        "layer_totals": layer_totals,
        "samples": rows,
    }
    out_path = OUT_DIR / "neg_stats.json"
    out_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": totals, "layer_totals": layer_totals,
                      "elapsed_s": round(time.time() - started, 1)}, ensure_ascii=False))
    if errors:
        print(f"ERRORS: {len(errors)}", file=sys.stderr)
        for row in errors[:5]:
            print(row["sample_id"], row["error"].splitlines()[-1], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
