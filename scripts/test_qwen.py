"""End-to-end smoke for the Qwen reasoner against a real OpenAI-compatible API.

Usage: python scripts/test_qwen.py [--config configs/pipeline.yaml]
Prints the routing/decision result; never prints credentials.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbparse.config import load_config
from pcbparse.pipeline.budget import TokenBudget
from pcbparse.reasoning.qwen_client import HTTPQwenReasoner
from pcbparse.reasoning.router import AmbiguityTask, Candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=r"E:\EDA挑战题目六\configs\pipeline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    qwen = config.get("qwen", {})
    if not qwen.get("enabled"):
        print("qwen not enabled in config")
        return 2
    endpoint = qwen["endpoint"]
    model = qwen["model"]
    has_key = bool(qwen.get("api_key"))
    print(f"endpoint: {endpoint}")
    print(f"model: {model}")
    print(f"api_key: {'<set>' if has_key else '<missing>'}")
    print(f"timeout: {qwen.get('timeout_seconds')}s")

    task = AmbiguityTask(
        task_id="reasoning:component:demo:key",
        task_type="component.key",
        candidates=(
            Candidate(candidate_id="candidate:aaaa", value="R12", score=0.72),
            Candidate(candidate_id="candidate:bbbb", value="R21", score=0.55),
            Candidate(candidate_id="candidate:cccc", value="R1 2", score=0.40),
        ),
        primary_score=0.72,
        graph_inconsistency_score=0.9,
        downstream_value_score=1.0,
        hardcase_prior=0.8,
        context={"owner_kind": "component", "object_id": "component-local:demo", "field": "key", "current_value": ""},
    )
    budget = TokenBudget(
        max_calls=int(qwen.get("max_calls_per_image", 3)),
        max_input_tokens=int(qwen.get("max_input_tokens_per_image", 3000)),
        max_output_tokens_per_call=int(qwen.get("max_output_tokens_per_call", 64)),
    )
    reasoner = HTTPQwenReasoner(
        endpoint=endpoint,
        model=model,
        timeout_seconds=float(qwen.get("timeout_seconds", 20.0)),
        api_key=qwen.get("api_key"),
    )
    decision = reasoner.choose(task, budget)
    print("decision:", json.dumps({
        "task_id": decision.task_id,
        "selected_id": decision.selected_id,
        "confidence": decision.confidence,
        "source": decision.source,
        "reason": decision.reason,
    }, ensure_ascii=False))
    print("budget calls remaining:", budget.remaining_calls)
    if decision.selected_id not in {c.candidate_id for c in task.candidates} | {"UNKNOWN"}:
        print("FAIL: out-of-set decision")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
