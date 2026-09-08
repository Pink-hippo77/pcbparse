from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from pcbparse.pipeline.budget import TokenBudget
from pcbparse.reasoning.router import AmbiguityTask, Decision


class Reasoner(Protocol):
    def choose(self, task: AmbiguityTask, budget: TokenBudget) -> Decision: ...


class DisabledReasoner:
    def choose(self, task: AmbiguityTask, budget: TokenBudget) -> Decision:
        return Decision(task_id=task.task_id, selected_id="UNKNOWN", confidence=0.0, source="disabled")


class CandidateOnlyReasoner:
    """Deterministic offline fallback that never creates a value or coordinate."""

    def __init__(self, minimum_score: float = 0.75, minimum_margin: float = 0.10) -> None:
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin

    def choose(self, task: AmbiguityTask, budget: TokenBudget) -> Decision:
        ranked = sorted(task.candidates, key=lambda item: (-item.score, item.candidate_id))
        if not ranked:
            return Decision(task.task_id, "UNKNOWN", 0.0, "candidate_only", "no candidates")
        second = ranked[1].score if len(ranked) > 1 else 0.0
        if ranked[0].score < self.minimum_score or ranked[0].score - second < self.minimum_margin:
            return Decision(task.task_id, "UNKNOWN", ranked[0].score, "candidate_only", "low score or margin")
        decision = Decision(task.task_id, ranked[0].candidate_id, ranked[0].score, "candidate_only")
        decision.validate_against(task)
        return decision


class HTTPQwenReasoner:
    """Strict candidate-selection adapter for an OpenAI-compatible endpoint.

    The reasoner never generates values or coordinates: it selects one supplied
    candidate id or ``UNKNOWN``.  ``enable_thinking=False`` keeps the Qwen3
    reasoning budget small (the MaaS reports reasoning tokens in
    ``usage.completion_tokens``).  When ``send_images`` is enabled and a task
    carries a local ``crop_path``, the ROI is attached as a base64 data URL so
    vision-capable models can disambiguate visually; text-only models ignore
    the image part gracefully by still receiving the candidate JSON.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout_seconds: float = 12.0,
        api_key: str | None = None,
        enable_thinking: bool = False,
        send_images: bool = False,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.send_images = send_images

    def choose(self, task: AmbiguityTask, budget: TokenBudget) -> Decision:
        prompt = _candidate_prompt(task)
        user_message = _user_message(task, prompt, self.send_images)
        estimated_input = max(1, len(prompt) // 4)
        budget.reserve(estimated_input, min(64, budget.max_output_tokens_per_call))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return JSON only. No extra text.",
                },
                user_message,
            ],
            "temperature": 0,
            "max_tokens": min(64, budget.max_output_tokens_per_call),
            "response_format": {"type": "json_object"},
        }
        if not self.enable_thinking:
            payload["enable_thinking"] = False
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return Decision(task.task_id, "UNKNOWN", 0.0, "qwen_http", f"request failed: {type(exc).__name__}")
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            parsed = json.loads(content)
            decision = Decision(
                task_id=task.task_id,
                selected_id=str(parsed.get("selected_id", "UNKNOWN")),
                confidence=float(parsed.get("confidence", 0.0)),
                source="qwen_http",
            )
            decision.validate_against(task)
        except (TypeError, ValueError, json.JSONDecodeError):
            decision = Decision(task.task_id, "UNKNOWN", 0.0, "qwen_http", "invalid structured response")
        output_tokens = int(data.get("usage", {}).get("completion_tokens", max(1, len(content) // 4)))
        budget.record_output(output_tokens)
        return decision


def _candidate_prompt(task: AmbiguityTask) -> str:
    compact = {
        "task": task.task_type,
        "context": task.context,
        "candidates": [
            {"id": candidate.candidate_id, "value": candidate.value, "score": round(candidate.score, 6)}
            for candidate in task.candidates
        ],
        "instructions": (
            "Pick the single best candidate id, or UNKNOWN if none is clearly "
            "right. Respond with exactly " + '{"selected_id": "<id or UNKNOWN>", '
            '"confidence": <0..1>}.'
        ),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _user_message(task: AmbiguityTask, prompt: str, send_images: bool) -> dict[str, Any]:
    if not send_images or not task.crop_path:
        return {"role": "user", "content": prompt}
    import base64
    from pathlib import Path

    try:
        crop = Path(task.crop_path).expanduser().resolve()
        encoded = base64.b64encode(crop.read_bytes()).decode("ascii")
        suffix = crop.suffix.lower().lstrip(".") or "png"
        if suffix not in {"png", "jpg", "jpeg", "webp", "bmp"}:
            suffix = "png"
        data_url = f"data:image/{suffix};base64,{encoded}"
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    except OSError:
        return {"role": "user", "content": prompt}
