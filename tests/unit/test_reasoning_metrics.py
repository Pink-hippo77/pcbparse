from __future__ import annotations

import unittest

from pcbparse.evaluation.metrics import connectivity_prf1, set_prf1
from pcbparse.pipeline.budget import BudgetExceeded, TimeBudget, TokenBudget
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.qwen_client import CandidateOnlyReasoner
from pcbparse.reasoning.router import AmbiguityTask, Candidate, ValueAwareRouter


class ReasoningTests(unittest.TestCase):
    def test_time_budget_fails_at_zero_and_after_stage_overrun(self) -> None:
        with self.assertRaises(BudgetExceeded):
            TimeBudget(0).check()
        ctx = RunContext(
            run_id="r",
            image_id="i",
            config_hash="h",
            time_budget=TimeBudget(0.001),
            token_budget=TokenBudget(0, 0, 0),
        )
        ctx.time_budget.started_at -= 1.0
        with self.assertRaises(BudgetExceeded):
            with ctx.stage("slow"):
                pass

    def test_reasoner_cannot_create_values(self) -> None:
        task = AmbiguityTask(
            task_id="t1",
            task_type="designator",
            candidates=(Candidate("c1", "R8", 0.91), Candidate("c2", "R6", 0.40)),
            primary_score=0.4,
            graph_inconsistency_score=1.0,
            downstream_value_score=1.0,
            hardcase_prior=1.0,
        )
        decision = CandidateOnlyReasoner().choose(task, TokenBudget(3, 3000, 64))
        self.assertEqual("c1", decision.selected_id)
        decision.validate_against(task)

    def test_router_obeys_budgets(self) -> None:
        task = AmbiguityTask(
            task_id="t2",
            task_type="crossing",
            candidates=(Candidate("connected", "connected", 0.5), Candidate("open", "open", 0.5)),
            primary_score=0.0,
            graph_inconsistency_score=1.0,
            downstream_value_score=1.0,
            hardcase_prior=1.0,
        )
        router = ValueAwareRouter(threshold=0.5, min_time_reserve_seconds=0.0)
        self.assertTrue(router.should_route(task, TokenBudget(1, 100, 64), TimeBudget(10), estimated_input_tokens=50))
        self.assertFalse(router.should_route(task, TokenBudget(0, 100, 64), TimeBudget(10), estimated_input_tokens=50))


class MetricTests(unittest.TestCase):
    def test_set_prf1(self) -> None:
        metric = set_prf1({"a", "b"}, {"b", "c"})
        self.assertAlmostEqual(0.5, metric.f1)

    def test_connectivity(self) -> None:
        metric = connectivity_prf1([{"A.1", "B.1", "C.1"}], [{"A.1", "B.1"}, {"C.1"}])
        self.assertEqual(1, metric.true_positive)
        self.assertEqual(2, metric.false_positive)


if __name__ == "__main__":
    unittest.main()
