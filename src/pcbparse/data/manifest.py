from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_text_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write canonical UTF-8 JSONL atomically."""

    lines = [_json_text(record) for record in records]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    _atomic_text_write(Path(path), content)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read manifest JSONL and reject malformed or duplicate records."""

    path = Path(path)
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"manifest line {line_number} lacks sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r} in manifest")
            sample_ids.add(sample_id)
            records.append(record)
    return records


def write_json(value: Any, path: Path) -> None:
    """Write a canonical, human-readable JSON document atomically."""

    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    _atomic_text_write(Path(path), content + "\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")

