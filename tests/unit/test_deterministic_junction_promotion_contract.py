from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data_tools import model_promotion as promotion


class DeterministicJunctionPromotionContractTests(unittest.TestCase):
    def test_mode_selects_exact_task_sets_and_fails_closed(self) -> None:
        self.assertEqual(
            promotion.REQUIRED_TASKS,
            promotion._tasks_for_junction_mode("learned_model"),
        )
        self.assertEqual(
            promotion.DETERMINISTIC_JUNCTION_TASKS,
            promotion._tasks_for_junction_mode("deterministic_vectorizer"),
        )
        with self.assertRaisesRegex(promotion.ModelPromotionError, "junction_mode"):
            promotion._tasks_for_junction_mode("disabled")

    def test_current_config_binds_four_model_plan_and_vectorizer_fixture(self) -> None:
        contract = promotion._verify_deterministic_junction_contract(
            Path("configs/pipeline.yaml")
        )

        self.assertEqual(
            list(promotion.DETERMINISTIC_JUNCTION_TASKS),
            contract["model_task_plan"]["required_model_tasks"],
        )
        self.assertFalse(contract["model_task_plan"]["learned_junction_model_loaded"])
        self.assertFalse(contract["model_task_plan"]["learned_junction_model_invoked"])
        self.assertIn("t_junction", contract["vectorizer_fixture"]["junction_kinds"])
        self.assertEqual(4, len(contract["runtime_sources"]))
        declared = contract.pop("contract_payload_sha256")
        actual = hashlib.sha256(
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(declared, actual)

    def test_learned_enabled_config_cannot_be_promoted_as_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pipeline.json"
            config.write_text(
                json.dumps(
                    {"graph": {"learned_junctions_enabled": True}},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                promotion.ModelPromotionError,
                "learned_junctions_enabled=false",
            ):
                promotion._verify_deterministic_junction_contract(config)

    def test_bundle_count_is_mode_specific_before_any_candidate_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models.lock.json"
            with self.assertRaisesRegex(
                promotion.ModelPromotionError,
                "exactly 4 candidate manifests",
            ):
                promotion.assemble_model_lock(
                    [],
                    output=output,
                    junction_mode="deterministic_vectorizer",
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
