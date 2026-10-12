"""Hash-bound static model names for semantic classification, never pin synthesis."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _parse_catalog(payload: bytes) -> frozenset[str]:
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported model catalog schema")
    names = document.get("model_names")
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("model catalog must contain nonempty model_names")
    # Normalization is used only to match visible text. Raw OCR spelling is
    # retained in Circuit-IR and remains the value assigned to searchName.
    return frozenset(unicodedata.normalize("NFKC", name).strip().upper() for name in names)


def load_model_catalog(path: str, expected_sha256: str) -> frozenset[str]:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("model catalog hash mismatch")
    return _parse_catalog(payload)
