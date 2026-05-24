"""Project-wide pytest fixtures.

The deferred-tools registry in :mod:`agent_harness.core.toolsets` is a
*process-wide* mutable list (see the ``_DEFERRED_TOOLS`` global) — a Wave-2
placeholder until ``RunContext`` lands in Wave-3 and the catalog migrates to
per-run storage. To prevent cross-test leakage (pytest-xdist or any test
forgetting to clear), this auto-use fixture wipes the registry around every
test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent_harness.core.toolsets import clear_deferred_tools


@pytest.fixture(autouse=True)
def _clear_deferred_tools() -> Iterator[None]:
    """Reset the process-wide deferred-tools registry between tests."""

    clear_deferred_tools()
    try:
        yield
    finally:
        clear_deferred_tools()
