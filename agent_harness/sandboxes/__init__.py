"""Sandbox backends (Layer 2).

Per S8: built-in are `InProcessSandbox`, `ModalSandbox`, `FlySandbox`.
Other backends (E2B, Daytona, Docker, …) are user-implementation via
the `Sandbox` Protocol in :mod:`agent_harness.core.sandbox`.
"""

from .fly import FlySandbox
from .inprocess import InProcessSandbox
from .modal import ModalSandbox

__all__ = ["FlySandbox", "InProcessSandbox", "ModalSandbox"]
