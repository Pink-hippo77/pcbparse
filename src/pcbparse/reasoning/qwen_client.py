from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from pcbparse.config import qwen_chat_endpoint
from pcbparse.pipeline.budget import TokenBudget
from pcbparse.reasoning.router import AmbiguityTask, Decision
from pcbparse.reasoning.http_transport import request_json_with_deadline


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

    ``choose`` and ``choose_many`` select a supplied candidate id or ``UNKNOWN``.
    The separate opt-in ``read_numbers`` transcribes source-image pixels for
    confirmation by the caller. No method generates geometry or IR objects.
    ``enable_thinking=False`` keeps the Qwen3
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
        isolate_http: bool = True,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        self.endpoint = qwen_chat_endpoint(endpoint)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.send_images = send_images
        # Production always uses the deadline worker. The inline option is for
        # deterministic protocol tests that replace urllib without networking.
        self.isolate_http = isolate_http
        self.consecutive_failures = 0
        self.cooldown_until = 0.0
        self.disabled_reason = None
        self.requests = 0
        self.failures = 0

    def choose(self, task: AmbiguityTask, budget: TokenBudget, *, timeout_seconds: float | None = None) -> Decision:
        prompt = _candidate_prompt(task)
        content, error, source = self._request(prompt, _user_message(task, prompt, self.send_images), budget, timeout_seconds)
        if error:
            return Decision(task.task_id, "UNKNOWN", 0.0, source, error)
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError('structured response must be an object')
            decision = Decision(task.task_id, str(parsed.get('selected_id', 'UNKNOWN')),
                                float(parsed.get('confidence', 0.0)), 'qwen_http')
            decision.validate_against(task)
        except (TypeError, ValueError, json.JSONDecodeError):
            self._failed('invalid_structured_response')
            return Decision(task.task_id, 'UNKNOWN', 0.0, 'qwen_http', 'invalid structured response')
        self.consecutive_failures = 0
        return decision

    def choose_many(self, tasks: list[AmbiguityTask], budget: TokenBudget, *, timeout_seconds: float | None = None) -> list[Decision]:
        """Select independently from each supplied set using one labelled image.

        Short aliases reduce output size; an alias is meaningful only within
        its own task. This never creates a candidate value or a pin identity.
        """
        if not 2 <= len(tasks) <= 6 or len({t.task_id for t in tasks}) != len(tasks):
            raise ValueError('visual batch needs two to six distinct tasks')
        if (not self.send_images or not tasks[0].crop_path
                or any(t.crop_path != tasks[0].crop_path or not 2 <= len(t.candidates) <= 5 for t in tasks)):
            return [Decision(t.task_id, 'UNKNOWN', 0., 'qwen_not_called', 'batch_requires_shared_visual_crop') for t in tasks]
        rows = [{'slot': str(i), 'field': t.context.get('field'),
                 'target_point_in_crop': t.context.get('target_point_in_crop'),
                 'terminal_side': t.context.get('terminal_side'),
                 'terminal_label_layout': t.context.get('terminal_label_layout'),
                 'candidates': {'#'+str(j): c.value for j,c in enumerate(t.candidates,1)}}
                for i,t in enumerate(tasks,1)]
        prompt = json.dumps({'tasks': rows, 'instructions':
            'Each numbered image tile is an independent target. Read only the printed field next to its target point. '
            'Select a candidate alias from that slot only if EVERY printed character matches, or 0 if unclear '
            'or if the visible answer is absent from its candidates. A similar spelling or uppercasing is insufficient. '
            'A pin number is the identifier at the symbol boundary; a farther wire/signal name is not its pin number. '
            'Never infer a pin sequence or use model knowledge. '
            'Return exactly {"selections":[["slot","candidate alias or 0",confidence_0_to_1],...]}. '
            'Include every slot once. Do not copy an alias or answer between targets.'}, separators=(',',':'))
        message = _user_message(tasks[0],prompt,True)
        if not isinstance(message['content'],list):
            return [Decision(t.task_id,'UNKNOWN',0.,'qwen_not_called','batch_crop_unavailable') for t in tasks]
        content,error,source = self._request(prompt,message,budget,timeout_seconds)
        if error:
            return [Decision(t.task_id,'UNKNOWN',0.,source,error) for t in tasks]
        try:
            parsed = json.loads(content)
            if not isinstance(parsed,dict) or set(parsed) != {'selections'} or not isinstance(parsed['selections'],list):
                raise ValueError('invalid batch envelope')
            aliases = {str(i): t for i,t in enumerate(tasks,1)}
            selected = {}
            for row in parsed['selections']:
                if not isinstance(row,list) or len(row) != 3:
                    raise ValueError('invalid batch row')
                slot,choice,confidence = row
                if not isinstance(slot,str) or slot not in aliases or slot in selected:
                    raise ValueError('unknown or duplicate batch slot')
                if not isinstance(choice,str) or type(confidence) not in (int,float):
                    raise ValueError('invalid selection types')
                t = aliases[slot]
                candidates = {'#'+str(i): c.candidate_id for i,c in enumerate(t.candidates,1)}
                # Some compatible services return the visible value instead
                # of its alias. Accept only an exact supplied value, with
                # aliases chosen outside the printed-field grammar.
                if choice not in candidates and choice != '0':
                    exact = [c.candidate_id for c in t.candidates if c.value == choice]
                    if len(exact) == 1:
                        candidates[choice] = exact[0]
                if choice not in candidates and choice != '0':
                    raise ValueError('candidate outside its slot')
                decision = Decision(t.task_id,candidates.get(choice,'UNKNOWN'),float(confidence),'qwen_http')
                decision.validate_against(t)
                selected[slot] = decision
        except (TypeError,ValueError,json.JSONDecodeError):
            self._failed('invalid_batch_response')
            return [Decision(t.task_id,'UNKNOWN',0.,'qwen_http','invalid batch response') for t in tasks]
        self.consecutive_failures = 0
        return [selected.get(str(i),Decision(t.task_id,'UNKNOWN',0.,'qwen_http','missing batch selection'))
                for i,t in enumerate(tasks,1)]

    def read_numbers(self, tasks: list[AmbiguityTask], budget: TokenBudget, *,
                     timeout_seconds: float | None = None) -> dict[str, tuple[str, float]]:
        """Transcribe bounded image rows; this adapter cannot edit IR objects.

        Unlike candidate selection, this opt-in operation can propose a visible
        number absent from OCR. The caller must independently confirm its crop
        and reject conflicts before writing an existing terminal's number.
        """
        import base64
        import re
        from pathlib import Path
        if not self.send_images or not 1 <= len(tasks) <= 3:
            return {}
        content = []
        for slot, task in enumerate(tasks, 1):
            if task.task_type != 'pin.transcribe_number' or not task.crop_path:
                return {}
            encoded = base64.b64encode(Path(task.crop_path).read_bytes()).decode('ascii')
            content.extend([
                {'type': 'text', 'text': json.dumps({'slot': str(slot),
                    'target_point_in_crop': task.context.get('target_point_in_crop'),
                    'terminal_side': task.context.get('terminal_side'),
                    'number_format': ('decimal_or_alphanumeric' if task.context.get('allow_alphanumeric_number') is True
                                      else 'decimal'),
                    'label_layout': task.context.get('terminal_label_layout', 'exterior_number_above_lead')})},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+encoded}},
            ])
        prompt = ('Each independent image has a declared label_layout. For interior_numeric_cell, transcribe '
                  'only the decimal digits in that isolated interior terminal-label cell. '
                  'For exterior_number_above_lead, transcribe only the printed terminal number just above the target '
                  'terminal line outside the symbol outline; target_point_in_crop locates that line. '
                  'Do not read a farther signal label or another row. Do not infer a numbering sequence, '
                  'device model or datasheet. Return null if any digit is unclear, clipped or absent. '
                  'Return JSON only: {"readings":[{"slot":"1","number":null,"confidence":0.0}]} '
                  'with one reading per slot. For decimal format use only digits. For decimal_or_alphanumeric '
                  'format an identifier may also have one or two ASCII letters followed by one to three digits '
                  '(for example A6). Preserve printed letter case; never use a signal name as a number.')
        content.insert(0, {'type': 'text', 'text': prompt})
        value, error, _ = self._request(prompt, {'role': 'user', 'content': content}, budget, timeout_seconds)
        if error:
            return {}
        try:
            parsed = json.loads(value)
            readings = parsed['readings']
            if not isinstance(readings, list) or len(readings) != len(tasks):
                raise ValueError('transcription cardinality mismatch')
            result, seen = {}, set()
            for row in readings:
                slot, number, confidence = row['slot'], row['number'], row['confidence']
                if (not isinstance(slot, str) or slot not in {str(i) for i in range(1, len(tasks)+1)}
                        or slot in seen or type(confidence) not in {float, int}
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError('invalid transcription slot or confidence')
                seen.add(slot)
                if number is not None:
                    # Some compatible APIs emit JSON integers despite the
                    # requested string schema. Accept exact bounded integers;
                    # never coerce booleans, floats, signs or arbitrary text.
                    if type(number) is int and 0 <= number <= 9999:
                        number = str(number)
                    target = tasks[int(slot)-1]
                    pattern = (r'(?:0|[1-9][0-9]{0,3}|[A-Za-z]{1,2}[0-9]{1,3})'
                               if target.context.get('allow_alphanumeric_number') is True
                               and target.context.get('terminal_label_layout') != 'interior_numeric_cell'
                               else r'(?:0|[1-9][0-9]{0,3})')
                    if not isinstance(number, str) or not re.fullmatch(pattern, number):
                        raise ValueError('invalid visible decimal number')
                    result[tasks[int(slot)-1].task_id] = (number, float(confidence))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._failed('invalid_transcription_response')
            return {}
        self.consecutive_failures = 0
        return result

    def _request(self, prompt: str, user_message: dict, budget: TokenBudget,
                 timeout_seconds: float | None) -> tuple[Any, str | None, str]:
        if self.disabled_reason or time.monotonic() < self.cooldown_until:
            return None, self.disabled_reason or 'service_cooldown', 'qwen_not_called'
        timeout = self.timeout_seconds if timeout_seconds is None else min(self.timeout_seconds, timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            return None, 'insufficient_time', 'qwen_not_called'
        parts = user_message['content']
        image_count = sum(isinstance(part, dict) and part.get('type') == 'image_url' for part in parts) if isinstance(parts, list) else 0
        estimated_input = max(1, len(prompt) // 4) + 256*image_count
        budget.reserve(estimated_input, min(192, budget.max_output_tokens_per_call))
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
            "max_tokens": min(192, budget.max_output_tokens_per_call),
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
        self.requests += 1
        try:
            if self.isolate_http:
                data = request_json_with_deadline(request, timeout)
            else:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeError) as exc:
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            self._failed(f'http_{status}' if status else type(exc).__name__, permanent=status in {400, 401, 403, 404})
            return None, self.disabled_reason or f'request_failed:{status or type(exc).__name__}', 'qwen_http'
        if not isinstance(data, dict):
            self._failed('invalid_response_envelope')
            return None, 'invalid_response_envelope', 'qwen_http'
        usage = data.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        actual_input = usage.get('prompt_tokens')
        if type(actual_input) is int and actual_input >= 0:
            budget.input_tokens += actual_input-estimated_input
        choices = data.get('choices')
        message = choices[0].get('message') if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, list):
            content = ''.join(part.get('text', '') for part in content
                              if isinstance(part, dict) and isinstance(part.get('text'), str))
        reported_output = usage.get('completion_tokens')
        output_tokens = reported_output if type(reported_output) is int and reported_output >= 0 else (
            max(1, len(content)//4) if isinstance(content, str) else 0)
        budget.record_output(output_tokens)
        return content, None, 'qwen_http'

    def _failed(self, reason: str, *, permanent: bool = False) -> None:
        self.failures += 1
        self.consecutive_failures += 1
        if permanent:
            self.disabled_reason = reason
        elif self.consecutive_failures >= 2:
            self.cooldown_until = time.monotonic()+60.0

    def health_record(self) -> dict[str, Any]:
        return {'requests': self.requests, 'failures': self.failures,
                'disabled_reason': self.disabled_reason,
                'cooldown_active': time.monotonic() < self.cooldown_until}


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
