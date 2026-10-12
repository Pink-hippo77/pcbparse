import unittest
from pathlib import Path
from unittest.mock import patch

from pcbparse.pipeline.budget import BudgetExceeded, TimeBudget
from pcbparse.pipeline.context import RunContext


class GlobalDeadlineTests(unittest.TestCase):
    def test_global_deadline_limits_long_single_image_budget(self):
        budget = TimeBudget(600, started_at=100, deadline_at=110)
        with patch('pcbparse.pipeline.budget.time.monotonic', return_value=109):
            self.assertEqual(1, budget.remaining_seconds)
            budget.check()
        with patch('pcbparse.pipeline.budget.time.monotonic', return_value=110):
            with self.assertRaises(BudgetExceeded):
                budget.check()

    def test_case_budget_remains_shorter_than_global_deadline(self):
        budget = TimeBudget(10, started_at=100, deadline_at=700)
        with patch('pcbparse.pipeline.budget.time.monotonic', return_value=111):
            with self.assertRaises(BudgetExceeded):
                budget.check()

    def test_stage_overrun_is_recorded_as_error(self):
        ctx = RunContext.start(Path('fixture.png'), {'run': {'max_image_seconds': 600}})
        with self.assertRaises(BudgetExceeded):
            with ctx.stage('fixture_call'):
                ctx.time_budget.deadline_at = 0
        self.assertEqual('error', ctx.trace[-1].status)


if __name__ == '__main__':
    unittest.main()
