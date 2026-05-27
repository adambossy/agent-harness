"""agent_harness — a simple-but-complete Python agent harness.

The top-level namespace re-exports the stable *core* contract: the
:class:`Agent` entry point, the :func:`tool` decorator, and every public
type, Protocol, event, and error a caller builds against::

    from agent_harness import Agent, tool, Message, ModelSettings

Concrete, swappable backends stay in their own sub-packages so the root
import is light and provider SDKs remain optional::

    from agent_harness.providers.anthropic import AnthropicProvider, AnthropicMessagesModel
    from agent_harness.sessions import SqliteSession
    from agent_harness.sandboxes import ModalSandbox
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core import *  # noqa: F403  (curated re-export; see core.__all__)
from .core import __all__ as _core_all

try:
    __version__ = version("agent-harness")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", *_core_all]
