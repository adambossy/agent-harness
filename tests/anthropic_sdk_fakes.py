"""Stand-ins for the ``anthropic`` SDK's streaming surface.

Distinct from :mod:`tests.fakes`, which fakes the *harness's* own Protocols
(``FakeModel``, ``FakeProvider``). These fake the **vendor SDK** instead, so
provider adapters can be exercised without the SDK installed and without a
network call.

Shared by ``test_anthropic_provider`` and ``test_openrouter_provider``: the
OpenRouter adapter subclasses the Anthropic one and streams the identical
event shapes, so a change to the SDK's event surface must land in one place
rather than drifting between two copies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

__all__ = ["_FakeStream", "_FakeStreamEvent", "_build_fake_client", "_ts"]


def _ts() -> datetime:
    """A fixed timestamp, so message construction is deterministic."""
    return datetime(2026, 1, 1, tzinfo=UTC)


class _FakeStreamEvent:
    """An SDK stream event. Adapters read fields via ``getattr``, so any
    keyword becomes an attribute and no real event class is needed."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeStream:
    """The SDK's async context-manager + async-iterator stream shape."""

    def __init__(self, events: list[_FakeStreamEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[_FakeStreamEvent]:
        async def _gen() -> AsyncIterator[_FakeStreamEvent]:
            for e in self._events:
                yield e

        return _gen()


def _build_fake_client(events: list[_FakeStreamEvent]) -> MagicMock:
    """A client whose ``messages.stream(...)`` replays ``events``.

    Injected via the providers' ``client=`` ctor kwarg, which bypasses
    credential resolution entirely.
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeStream(events))
    return client
