from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

from pcbparse.cli import _apply_profile


def _config() -> dict:
    return {
        "run": {},
        "qwen": {"enabled": False, "endpoint": "https://example.invalid/v1"},
        "validation": {"require_zero_errors": True},
    }


def _profile(*, qwen_enabled: bool) -> dict:
    return {
        "name": "test",
        "provider": "production",
        "qwen_enabled": qwen_enabled,
        "artifact_policy": "error",
        "strict_validation": True,
    }


class ProfileQwenOverrideTests(unittest.TestCase):
    def test_profile_disables_qwen_when_no_env_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            effective = _apply_profile(_config(), _profile(qwen_enabled=False))
        self.assertFalse(effective["qwen"]["enabled"])

    def test_profile_enables_qwen_when_configured(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            effective = _apply_profile(_config(), _profile(qwen_enabled=True))
        self.assertTrue(effective["qwen"]["enabled"])

    def test_env_override_can_force_enable(self) -> None:
        with mock.patch.dict(os.environ, {"PCBPARSE_QWEN_ENABLED": "1"}, clear=False):
            effective = _apply_profile(_config(), _profile(qwen_enabled=False))
        self.assertTrue(effective["qwen"]["enabled"])

    def test_env_override_can_force_disable(self) -> None:
        with mock.patch.dict(os.environ, {"PCBPARSE_QWEN_ENABLED": "0"}, clear=False):
            effective = _apply_profile(_config(), _profile(qwen_enabled=True))
        self.assertFalse(effective["qwen"]["enabled"])


if __name__ == "__main__":
    unittest.main()
