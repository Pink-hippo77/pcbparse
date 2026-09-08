"""Installed command modules for Dataset300 preparation.

Some audit/review commands intentionally remain repository tooling because
they consume the checked-out evidence tree rather than an installed runtime.
Expose that directory as the fallback half of this package so imports remain
stable when ``src`` is placed before the repository root on ``sys.path``.
Without this bridge, test order determined whether modules such as
``data_tools.junction_truth_audit`` could be imported.
"""

from pathlib import Path


_REPOSITORY_TOOLS = Path(__file__).resolve().parents[2] / "data_tools"
if _REPOSITORY_TOOLS.is_dir():
    __path__.append(str(_REPOSITORY_TOOLS))
