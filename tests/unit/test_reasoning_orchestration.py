from __future__ import annotations

import unittest

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    Evidence,
    ImageInfo,
    Junction,
    Pin,
    Point,
)
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.cache import ReasoningCache
from pcbparse.reasoning.orchestrator import run_selective_reasoning
from pcbparse.reasoning.qwen_client import CandidateOnlyReasoner, HTTPQwenReasoner
from pcbparse.reasoning.router import AmbiguityTask, Decision
from pcbparse.reasoning.tasks import generate_ambiguity_tasks


class SelectingReasoner:
    version = "fixture-v1"

    def __init__(self, selected_value: str | None = None) -> None:
        self.selected_value = selected_value
        self.calls = 0

    def choose(self, task: AmbiguityTask, budget: object) -> Decision:
        self.calls += 1
        selected = next(
            (
                candidate
                for candidate in task.candidates
                if self.selected_value is not None and candidate.value == self.selected_value
            ),
            task.candidates[0],
        )
        return Decision(task.task_id, selected.candidate_id, 0.91, "fixture")


class OutOfSetReasoner:
    version = "invalid-v1"

    def __init__(self) -> None:
        self.calls = 0

    def choose(self, task: AmbiguityTask, budget: object) -> Decision:
        self.calls += 1
        return Decision(task.task_id, "fabricated-value", 0.99, "fixture")


class ReasoningOrchestrationTests(unittest.TestCase):
    def test_disabled_is_a_literal_noop(self) -> None:
        ir = _component_ir()
        before = ir.model_dump(mode="json")
        reasoner = SelectingReasoner("10k")

        result = run_selective_reasoning(
            ir,
            {"qwen": {"enabled": False}},
            _context({"enabled": False}),
            reasoner,
        )

        self.assertIs(result, ir)
        self.assertEqual(before, ir.model_dump(mode="json"))
        self.assertEqual(0, reasoner.calls)

    def test_candidate_only_reasoner_can_fill_only_a_supplied_empty_field(self) -> None:
        ir = _component_ir()
        original_bbox = ir.components[0].bbox.model_copy(deep=True)
        config = _config()

        run_selective_reasoning(
            ir,
            config,
            _context(config["qwen"]),
            CandidateOnlyReasoner(minimum_score=0.75, minimum_margin=0.10),
        )

        self.assertEqual("10k", ir.components[0].value)
        self.assertEqual(original_bbox, ir.components[0].bbox)
        self.assertEqual("qwen", ir.components[0].evidence[-1].source)
        row = ir.metadata["selective_reasoning"]["ledger"][0]
        self.assertIn(row["applied_value"], {item["value"] for item in row["candidates"]})

    def test_router_threshold_prevents_call_and_application(self) -> None:
        ir = _component_ir()
        config = _config(route_threshold=1.0)
        reasoner = SelectingReasoner("10k")

        run_selective_reasoning(ir, config, _context(config["qwen"]), reasoner)

        self.assertEqual("", ir.components[0].value)
        self.assertEqual(0, reasoner.calls)
        self.assertEqual("not_routed", ir.metadata["selective_reasoning"]["ledger"][0]["status"])

    def test_out_of_set_decision_is_rejected_without_mutation(self) -> None:
        ir = _component_ir()
        config = _config()
        reasoner = OutOfSetReasoner()

        run_selective_reasoning(ir, config, _context(config["qwen"]), reasoner)

        self.assertEqual("", ir.components[0].value)
        self.assertEqual(1, reasoner.calls)
        self.assertEqual(
            "decision_rejected",
            ir.metadata["selective_reasoning"]["ledger"][0]["status"],
        )
        self.assertIn("REASONING.DECISION_REJECTED", {item.code for item in ir.diagnostics})

    def test_call_budget_routes_only_the_highest_value_task(self) -> None:
        ir = _component_ir(include_key_task=True)
        config = _config(max_calls_per_image=1)
        reasoner = SelectingReasoner()
        ctx = _context(config["qwen"])

        run_selective_reasoning(ir, config, ctx, reasoner)

        ledger = ir.metadata["selective_reasoning"]["ledger"]
        self.assertEqual(1, reasoner.calls)
        self.assertEqual(1, ctx.token_budget.calls)
        self.assertEqual(1, sum(row["status"] == "applied" for row in ledger))
        self.assertEqual(1, sum(row["status"] == "not_routed" for row in ledger))

    def test_ambiguous_junction_can_only_be_resolved_to_crossing_kind(self) -> None:
        ir = _junction_ir()
        config = _config()
        reasoner = SelectingReasoner("connected_cross")
        point_before = ir.junctions[0].point.model_copy(deep=True)

        run_selective_reasoning(ir, config, _context(config["qwen"]), reasoner)

        self.assertEqual("connected_cross", ir.junctions[0].kind)
        self.assertEqual(point_before, ir.junctions[0].point)
        task = generate_ambiguity_tasks(_junction_ir())[0]
        self.assertEqual(
            {"connected_cross", "non_connected_cross"},
            {candidate.value for candidate in task.candidates},
        )

    def test_text_only_http_reasoner_skips_junction_without_crop(self) -> None:
        ir = _junction_ir()
        config = _config()
        reasoner = HTTPQwenReasoner(
            endpoint="https://unused.invalid/v1/chat/completions",
            model="unused",
            timeout_seconds=1.0,
            send_images=False,
        )

        run_selective_reasoning(ir, config, _context(config["qwen"]), reasoner)

        self.assertEqual("ambiguous", ir.junctions[0].kind)
        ledger = ir.metadata["selective_reasoning"]["ledger"]
        self.assertEqual(1, len(ledger))
        self.assertEqual("not_routed", ledger[0]["status"])
        self.assertIn("requires a visual crop", ledger[0]["reason"])

    def test_rerun_is_idempotent_and_equivalent_fresh_ir_hits_cache(self) -> None:
        cache = ReasoningCache()
        config = _config(cache_store=cache)
        reasoner = SelectingReasoner("10k")
        ir = _component_ir()
        ctx = _context(config["qwen"])

        run_selective_reasoning(ir, config, ctx, reasoner)
        first = ir.model_dump(mode="json")
        run_selective_reasoning(ir, config, ctx, reasoner)

        self.assertEqual(first, ir.model_dump(mode="json"))
        self.assertEqual(1, reasoner.calls)

        equivalent = _component_ir()
        equivalent_ctx = _context(config["qwen"])
        run_selective_reasoning(equivalent, config, equivalent_ctx, reasoner)
        self.assertEqual("10k", equivalent.components[0].value)
        self.assertEqual(1, reasoner.calls)
        self.assertTrue(equivalent.metadata["selective_reasoning"]["ledger"][0]["cache_hit"])
        self.assertEqual(0, equivalent_ctx.token_budget.calls)

    def test_existing_field_is_never_exposed_as_a_task_or_overwritten(self) -> None:
        ir = _component_ir()
        ir.components[0].value = "verified"
        ir.components[0].score = 0.99
        reasoner = SelectingReasoner("10k")
        config = _config()

        run_selective_reasoning(ir, config, _context(config["qwen"]), reasoner)

        self.assertEqual("verified", ir.components[0].value)
        self.assertEqual(0, reasoner.calls)
        self.assertEqual(0, ir.metadata["selective_reasoning"]["generated_tasks"])


def _component_ir(*, include_key_task: bool = False) -> CircuitIR:
    evidence = [
        Evidence(
            evidence_id="association:component-field:C1:value",
            source="rule",
            score=0.90,
            metadata={
                "kind": "component-field-resolution",
                "field": "value",
                "decision": "ambiguous",
                "candidates": [
                    {"value": "10k", "score": 0.90, "lexical_candidates": ["10k"]},
                    {"value": "100k", "score": 0.60, "lexical_candidates": ["100k"]},
                ],
            },
        )
    ]
    if include_key_task:
        evidence.append(
            Evidence(
                evidence_id="association:component-field:C1:designator",
                source="rule",
                score=0.80,
                metadata={
                    "kind": "component-field-resolution",
                    "field": "designator",
                    "decision": "ambiguous",
                    "candidates": [
                        {"value": "R1", "score": 0.80},
                        {"value": "R7", "score": 0.72},
                    ],
                },
            )
        )
    return CircuitIR(
        image=ImageInfo(
            image_id="reasoning-fixture",
            path="missing-fixture.png",
            width=128,
            height=96,
            sha256="a" * 64,
        ),
        components=[
            Component(
                internal_id="C1",
                category="resistor",
                bbox=BBox(x1=10, y1=20, x2=30, y2=40),
                evidence=evidence,
            )
        ],
        pins=[
            Pin(
                internal_id="P1",
                component_id="C1",
                point=Point(x=10, y=30),
            )
        ],
    )


def _junction_ir() -> CircuitIR:
    return CircuitIR(
        image=ImageInfo(
            image_id="junction-fixture",
            path="missing-junction.png",
            width=64,
            height=64,
            sha256="b" * 64,
        ),
        junctions=[
            Junction(
                internal_id="J1",
                point=Point(x=32, y=32),
                kind="ambiguous",
                segment_ids=["S1", "S2"],
                score=0.8,
            )
        ],
    )


def _config(**overrides: object) -> dict[str, object]:
    qwen: dict[str, object] = {
        "enabled": True,
        "route_threshold": 0.0,
        "max_calls_per_image": 3,
        "max_input_tokens_per_image": 3000,
        "max_output_tokens_per_call": 64,
        "min_time_reserve_seconds": 0.0,
        "cache": True,
    }
    qwen.update(overrides)
    return {"run": {"max_image_seconds": 120.0}, "qwen": qwen}


def _context(qwen: dict[str, object]) -> RunContext:
    serializable_qwen = {key: value for key, value in qwen.items() if key != "cache_store"}
    return RunContext.start(
        "missing-fixture.png",
        {"run": {"max_image_seconds": 120.0}, "qwen": serializable_qwen},
    )


if __name__ == "__main__":
    unittest.main()
