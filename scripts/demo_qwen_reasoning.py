"""Demo: Qwen selective reasoning inside the pipeline on a synthetic ambiguous IR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbparse.config import load_config
from pcbparse.ir.models import BBox, CircuitIR, Component, Evidence, IRState, ImageInfo
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.orchestrator import run_selective_reasoning
from pcbparse.reasoning.qwen_client import HTTPQwenReasoner


def main() -> int:
    config = load_config(r"E:\EDA挑战题目六\configs\pipeline.yaml")
    qwen = config["qwen"]
    print("qwen:", qwen.get("enabled"), qwen.get("backend"), qwen.get("model"))

    ir = CircuitIR(
        image=ImageInfo(image_id="demo", path="demo.png", width=1280, height=1024),
        state=IRState.ASSOCIATED,
        components=[
            Component(
                internal_id="component-local:demo",
                category="R",
                key=None,
                bbox=BBox(x1=10, y1=10, x2=40, y2=30),
                score=0.9,
                evidence=[
                    Evidence(
                        evidence_id="ocr:demo:key",
                        source="ocr",
                        score=0.72,
                        metadata={
                            "field": "key",
                            "kind": "component",
                            "decision": "ambiguous",
                            "candidates": [
                                {"value": "R12", "score": 0.72},
                                {"value": "R21", "score": 0.55},
                                {"value": "R1 2", "score": 0.40},
                            ],
                        },
                    )
                ],
            )
        ],
    )

    ctx = RunContext.start("demo.png", config)
    reasoner = HTTPQwenReasoner(
        endpoint=qwen["endpoint"],
        model=qwen["model"],
        timeout_seconds=float(qwen.get("timeout_seconds", 20.0)),
        api_key=qwen.get("api_key"),
        enable_thinking=bool(qwen.get("enable_thinking", False)),
        send_images=bool(qwen.get("send_images", False)),
    )
    run_selective_reasoning(ir, config, ctx, reasoner)
    report = ir.metadata.get("selective_reasoning", {})
    print("generated:", report.get("generated_tasks"), "routed:", report.get("routed_tasks"), "applied:", report.get("applied_tasks"))
    for row in report.get("ledger", []):
        d = row.get("decision") or {}
        print(" ", row.get("task_type"), "|", row.get("status"), "|", d.get("selected_id"), "|", d.get("confidence"))
    component = ir.components[0]
    print("component key after reasoning:", repr(component.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
