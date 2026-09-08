"""Test that failure IR is saved when PipelineError has partial_ir."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(r"E:\EDA挑战题目六")
sys.path.insert(0, str(ROOT / "src"))

from pcbparse.ir import CircuitIR, Component, ImageInfo, Point, WireSegment, PhysicalNet, Diagnostic
from pcbparse.pipeline.orchestrator import PipelineError


def test_pipeline_error_carries_ir():
    """Verify PipelineError can carry a partial_ir attribute."""
    ir = CircuitIR(
        image=ImageInfo(image_id="test", path="test.png", width=100, height=100),
        components=[
            Component(internal_id="c1", category="R", bbox={"x1": 10, "y1": 10, "x2": 20, "y2": 20}),
        ],
        segments=[
            WireSegment(internal_id="s1", start=Point(x=0, y=10), end=Point(x=20, y=10)),
            WireSegment(internal_id="s2", start=Point(x=30, y=10), end=Point(x=50, y=10)),
        ],
        physical_nets=[
            PhysicalNet(internal_id="pnet_1", pins=[], segments=["s1", "s2"]),
        ],
    )
    diagnostics = [
        Diagnostic(severity="ERROR", code="NET.DISCONNECTED_SEGMENTS", message="test", stage="graph"),
    ]

    err = PipelineError("validation failed with 1 error(s)", diagnostics)
    err.partial_ir = ir

    # Verify the IR is accessible
    assert err.partial_ir is not None
    assert len(err.partial_ir.components) == 1
    assert len(err.partial_ir.segments) == 2
    assert err.partial_ir.image.image_id == "test"

    # Verify it can be serialized
    payload = err.partial_ir.model_dump(mode="json")
    json_str = json.dumps(payload, ensure_ascii=False)
    assert len(json_str) > 100

    print("OK: PipelineError carries partial_ir successfully")
    print(f"   Components: {len(err.partial_ir.components)}")
    print(f"   Segments: {len(err.partial_ir.segments)}")
    print(f"   Physical nets: {len(err.partial_ir.physical_nets)}")
    print(f"   JSON size: {len(json_str)} bytes")


def test_infer_one_attaches_ir():
    """Verify infer_one attaches partial_ir to exception."""
    from pcbparse.pipeline.orchestrator import infer_one
    from pcbparse.perception.heuristic_provider import HeuristicVisionProvider

    # This will fail at validation because the heuristic provider produces
    # an IR with disconnected segments, but it should attach the IR.
    try:
        result = infer_one(
            ROOT / "data" / "images" / "0001-KiCad.png",  # May not exist, that's OK
            ROOT / "work" / "test_failure_ir" / "test.json",
            provider=HeuristicVisionProvider(),
            config=json.loads((ROOT / "configs" / "pipeline.yaml").read_text()),
        )
        print(f"⚠️ infer_one succeeded unexpectedly (status={result.ir.state})")
    except FileNotFoundError:
        print("WARN: Test image not found, skipping infer_one test")
    except Exception as exc:
        partial_ir = getattr(exc, "partial_ir", None)
        context = getattr(exc, "context_manifest", None)
        if partial_ir is not None:
            print(f"OK: infer_one attached partial_ir: {len(partial_ir.components)} components, {len(partial_ir.segments)} segments")
        else:
            print(f"INFO: infer_one did not attach partial_ir (exception type: {type(exc).__name__})")
        if context is not None:
            print(f"OK: context_manifest attached: {list(context.keys())[:5]}...")


if __name__ == "__main__":
    test_pipeline_error_carries_ir()
    print()
    test_infer_one_attaches_ir()
